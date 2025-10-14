"""Prometheus collector implementation using composition pattern.

Provides a reusable collector that separates concerns between data fetching,
metric generation, and caching through dependency injection.
"""

from collections.abc import Callable, Iterator
from typing import Generic, TypeAlias, TypeVar

import structlog
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily
from prometheus_client.metrics_core import Metric
from prometheus_client.registry import Collector

from .cache import AtomicThrottledCache

logger = structlog.get_logger(__name__)

T = TypeVar("T")


Fetcher: TypeAlias = Callable[[], list[T]]
MetricsGenerator: TypeAlias = Callable[[list[T]], Iterator[Metric]]


class SlurmCollector(Collector, Generic[T]):
    """Prometheus collector for SLURM metrics using composition pattern.

    Separates concerns through dependency injection:
    - Data fetching and transformation (via Fetcher function with injected dependencies)
    - Metric generation (via MetricsGenerator function)
    - Caching and error handling (managed internally)

    Each collector instance is configured with specific fetcher and generator
    functions, making it reusable for different metric types (nodes, jobs, etc.).
    Dependencies are injected into the fetcher at construction time.
    """

    def __init__(
        self,
        fetcher: Fetcher[T],
        generator: MetricsGenerator[T],
        metric_prefix: str,
        poll_limit: float,
        scraper_description: str,
    ):
        """Initialize the SLURM collector.

        Args:
            fetcher: Function that fetches and transforms data (with
                dependencies pre-injected).
            generator: Function that generates Prometheus metrics from data.
            metric_prefix: Metric name prefix (e.g., "node", "job").
            poll_limit: Minimum seconds between cache refreshes.
            scraper_description: Description of the scraper for logging
                (e.g., API base URL).
        """
        self._fetcher = fetcher
        self._generator = generator
        self._metric_prefix = metric_prefix

        # Initialize cache
        self._cache = AtomicThrottledCache[list[T]](poll_limit)

        # Track errors manually (no global Counter registration)
        self._error_count = 0

        # Scraper description for logging
        self._scraper_desc = scraper_description

    def fetch_metrics(self) -> tuple[list[T], float | None]:
        """Fetch metrics with caching and throttling.

        Returns cached data if within throttle window, otherwise calls the
        fetcher function to retrieve fresh data from the API.

        Returns:
            Tuple of (data, fetch_duration) where:
            - data: List of metric objects from cache or fresh API call
            - fetch_duration: Duration in seconds if fetched, None if cache hit
        """
        return self._cache.fetch_or_throttle(self._fetcher)

    def collect(self) -> Iterator[Metric]:
        """Collect metrics for Prometheus scrape.

        Called by Prometheus client during each scrape. Yields scrape metadata
        (duration and error count) followed by domain-specific metrics from the
        configured generator function.

        Yields:
            Prometheus Metric objects (metadata + domain metrics).
        """
        # Try to fetch metrics (cache returns duration or None for cache hit)
        data: list[T] | None = None
        try:
            data, fetch_duration = self.fetch_metrics()
            # Convert to -1 for cache hits, keep actual duration for fetches
            duration_value = fetch_duration if fetch_duration is not None else -1.0
        except Exception:
            logger.exception(
                "Failed to fetch metrics for collection",
                metric_prefix=self._metric_prefix,
            )
            self._error_count += 1
            duration_value = -1.0

        # Scrape duration metric (-1 indicates cache hit or error, >= 0 indicates fetch)
        scrape_duration = GaugeMetricFamily(
            f"slurm_{self._metric_prefix}_scrape_duration",
            f"scrape duration from {self._scraper_desc} in seconds, "
            f"-1 indicates cache hit or error",
        )
        scrape_duration.add_metric([], duration_value)
        yield scrape_duration

        # Error counter metric
        error_counter = CounterMetricFamily(
            f"slurm_{self._metric_prefix}_scrape_error",
            f"slurm {self._metric_prefix} info scrape errors",
        )
        error_counter.add_metric([], self._error_count)
        yield error_counter

        # Generate metrics from data using the injected generator (skip if error)
        if data is not None:
            yield from self._generator(data)
