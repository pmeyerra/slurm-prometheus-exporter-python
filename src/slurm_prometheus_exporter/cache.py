"""Thread-safe cache with throttling for API requests.

Provides a generic cache that prevents excessive API calls by throttling
requests and returning cached data when within the throttle window.
"""

import time
from collections.abc import Callable
from threading import Lock
from typing import Generic, TypeVar

import structlog

logger = structlog.get_logger()

T = TypeVar("T")


class AtomicThrottledCache(Generic[T]):
    """Thread-safe cache with throttling to prevent excessive API queries.

    Ensures that expensive data fetching operations are rate-limited,
    returning cached data when requests occur within the throttle window.
    """

    def __init__(self, limit: float):
        """Initialize the cache.

        Args:
            limit: Minimum seconds between cache refreshes.
        """
        self._lock = Lock()
        self._last_fetch: float | None = None
        self._limit = limit
        self._cache: T | None = None

    def fetch_or_throttle(self, fetch_func: Callable[[], T]) -> tuple[T, float | None]:
        """Fetch data or return cached data if within throttle limit.

        Thread-safe operation that either returns cached data (if still fresh)
        or calls fetch_func to retrieve new data and updates the cache.

        Args:
            fetch_func: Function to fetch fresh data.

        Returns:
            Tuple of (data, fetch_duration) where:
            - data: Cached or fresh data of type T
            - fetch_duration: Duration in seconds if fetched, None if cache hit
        """
        with self._lock:
            elapsed: float | None = (
                time.time() - self._last_fetch if self._last_fetch is not None else None
            )
            if (
                self._cache is not None
                and elapsed is not None
                and elapsed < self._limit
            ):
                logger.debug("Using cached data", age_seconds=round(elapsed, 2))
                return self._cache, None

            start = time.time()
            data = fetch_func()
            duration = time.time() - start
            self._cache = data
            self._last_fetch = time.time()
            logger.debug(
                "Fetched fresh data",
                duration_seconds=duration,
            )
            return data, duration
