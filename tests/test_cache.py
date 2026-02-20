"""Tests for AtomicThrottledCache behaviours not observable through SlurmCollector.

The throttle-window transition (fresh → cached → expired → re-fetch), the
exact return-tuple semantics (data, duration) vs (data, None), thread safety
under concurrent access, and the None-data edge case cannot be verified
through the high-level collect() API.  Basic caching effects (prevent
redundant fetches, serve cached data on hit, duration sign in scrape
metadata) are already exercised by the tests in test_collector.py.
"""

import threading
from unittest.mock import MagicMock, patch

from slurm_prometheus_exporter import cache

# ---------------------------------------------------------------------------
# First fetch
# ---------------------------------------------------------------------------


def test_first_fetch_invokes_fetch_func():
    """A fresh cache always invokes fetch_func."""
    fetch_func = MagicMock(return_value=["data"])
    c = cache.AtomicThrottledCache(limit=9999.0)

    c.fetch_or_throttle(fetch_func)

    fetch_func.assert_called_once()


def test_first_fetch_returns_data_and_non_negative_duration():
    """Fresh fetch returns (data, duration >= 0) tuple."""
    expected_data = ["item-a", "item-b"]
    fetch_func = MagicMock(return_value=expected_data)
    c = cache.AtomicThrottledCache(limit=9999.0)

    data, duration = c.fetch_or_throttle(fetch_func)

    assert data is expected_data
    assert isinstance(duration, float)
    assert duration >= 0.0


# ---------------------------------------------------------------------------
# Cache hit (within throttle window)
# ---------------------------------------------------------------------------


def test_cache_hit_returns_none_duration():
    """Within the throttle window, the second call returns None as duration."""
    fetch_func = MagicMock(return_value=["data"])
    c = cache.AtomicThrottledCache(limit=9999.0)

    c.fetch_or_throttle(fetch_func)
    _data, duration = c.fetch_or_throttle(fetch_func)

    assert duration is None


def test_cache_hit_returns_same_data():
    """Cache hit returns the same data object produced by the original fetch."""
    expected_data = [{"id": 1}]
    fetch_func = MagicMock(return_value=expected_data)
    c = cache.AtomicThrottledCache(limit=9999.0)

    c.fetch_or_throttle(fetch_func)
    data, _ = c.fetch_or_throttle(fetch_func)

    assert data is expected_data


def test_cache_hit_skips_fetch_func():
    """fetch_func is not called on a cache hit."""
    fetch_func = MagicMock(return_value=["data"])
    c = cache.AtomicThrottledCache(limit=9999.0)

    c.fetch_or_throttle(fetch_func)
    c.fetch_or_throttle(fetch_func)
    c.fetch_or_throttle(fetch_func)

    fetch_func.assert_called_once()


# ---------------------------------------------------------------------------
# Cache expiry
# ---------------------------------------------------------------------------


@patch("slurm_prometheus_exporter.cache.time")
def test_cache_expires_and_refetches(mock_time):
    """After the throttle window passes, fetch_func is called again with new data."""
    # First fetch: time calls are start, duration, _last_fetch
    # Second fetch (expired): elapsed check, start, duration, _last_fetch
    mock_time.time.side_effect = [
        100.0,  # start of first fetch
        100.01,  # end of first fetch (duration)
        100.02,  # _last_fetch assignment
        200.0,  # elapsed check: 200.0 - 100.02 = 99.98 >> 10.0
        200.0,  # start of second fetch
        200.01,  # end of second fetch (duration)
        200.02,  # _last_fetch assignment
    ]

    first_data = ["old"]
    second_data = ["new"]
    fetch_func = MagicMock(side_effect=[first_data, second_data])
    c = cache.AtomicThrottledCache(limit=10.0)

    data_1, dur_1 = c.fetch_or_throttle(fetch_func)
    data_2, dur_2 = c.fetch_or_throttle(fetch_func)

    assert data_1 is first_data
    assert data_2 is second_data
    assert dur_1 is not None
    assert dur_2 is not None
    assert fetch_func.call_count == 2


@patch("slurm_prometheus_exporter.cache.time")
def test_cache_stays_fresh_within_limit(mock_time):
    """Cached data is returned without calling fetch_func while within the window."""
    # First fetch: start, duration, _last_fetch
    # Second call (cache hit): elapsed check only
    mock_time.time.side_effect = [
        100.0,  # start of first fetch
        100.01,  # end of first fetch (duration)
        100.02,  # _last_fetch assignment
        100.05,  # elapsed check: 100.05 - 100.02 = 0.03 < 10.0 → hit
    ]

    expected_data = ["cached"]
    fetch_func = MagicMock(return_value=expected_data)
    c = cache.AtomicThrottledCache(limit=10.0)

    c.fetch_or_throttle(fetch_func)
    data, duration = c.fetch_or_throttle(fetch_func)

    assert data is expected_data
    assert duration is None
    fetch_func.assert_called_once()


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


def test_concurrent_access_single_fetch():
    """Multiple threads racing on a fresh cache result in only one fetch_func call."""
    fetch_func = MagicMock(return_value=["data"])
    c = cache.AtomicThrottledCache(limit=9999.0)
    thread_count = 20

    barrier = threading.Barrier(thread_count)

    def worker():
        barrier.wait()
        c.fetch_or_throttle(fetch_func)

    threads = [threading.Thread(target=worker) for _ in range(thread_count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    fetch_func.assert_called_once()


# ---------------------------------------------------------------------------
# Edge case: None data
# ---------------------------------------------------------------------------


def test_none_data_not_treated_as_cache_hit():
    """fetch_func returning None leaves the cache empty; the next call fetches again."""
    fetch_func = MagicMock(side_effect=[None, ["real-data"]])
    c = cache.AtomicThrottledCache(limit=9999.0)

    data_1, _ = c.fetch_or_throttle(fetch_func)
    data_2, dur_2 = c.fetch_or_throttle(fetch_func)

    assert data_1 is None
    assert data_2 == ["real-data"]
    assert dur_2 is not None
    assert fetch_func.call_count == 2
