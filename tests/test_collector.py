"""Tests for SlurmCollector wired with real collector pipelines.

These tests exercise the full pipeline through the public collect() method:
raw API data → transform → Prometheus metric families. By testing at this
level we also cover the internal behavior of the jobs and nodes collector
modules (_transform_job, _transform_node, generate_metrics, fetch, and the
total_memory_mb property) without calling private functions directly.

SlurmCollector-specific concerns (scrape metadata, caching, and error
handling) are tested here as well, using the real collector wiring rather
than synthetic stubs.
"""

from unittest.mock import MagicMock

import pytest

from slurm_prometheus_exporter import collector
from slurm_prometheus_exporter.collectors import jobs, nodes
from slurm_prometheus_exporter.slurmrestapi import client, types

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_client() -> MagicMock:
    """Mock SlurmRestApiClient with no pre-configured return values."""
    return MagicMock(spec=client.SlurmRestApiClient)


@pytest.fixture
def job_collector(mock_client: MagicMock) -> collector.SlurmCollector:
    """SlurmCollector wired with the real jobs pipeline (always fetches)."""
    return collector.SlurmCollector(
        fetcher=lambda: jobs.fetch(mock_client),
        generator=jobs.generate_metrics,
        metric_prefix="job",
        poll_limit=0.0,
        scraper_description="test-cluster",
    )


@pytest.fixture
def node_collector(mock_client: MagicMock) -> collector.SlurmCollector:
    """SlurmCollector wired with the real nodes pipeline (always fetches)."""
    return collector.SlurmCollector(
        fetcher=lambda: nodes.fetch(mock_client),
        generator=nodes.generate_metrics,
        metric_prefix="node",
        poll_limit=0.0,
        scraper_description="test-cluster",
    )


# ---------------------------------------------------------------------------
# Scrape metadata
# ---------------------------------------------------------------------------


def test_collect_always_yields_scrape_duration(
    mock_client: MagicMock,
    job_collector: collector.SlurmCollector,
):
    """Scrape duration gauge is always present in output."""
    mock_client.get_jobs.return_value = []
    metrics = {m.name: m for m in job_collector.collect()}
    assert "slurm_job_scrape_duration" in metrics


def test_collect_always_yields_scrape_error(
    mock_client: MagicMock,
    job_collector: collector.SlurmCollector,
):
    """Scrape error counter is always present in output."""
    mock_client.get_jobs.return_value = []
    metrics = {m.name: m for m in job_collector.collect()}
    assert "slurm_job_scrape_error" in metrics


def test_collect_scrape_duration_non_negative_on_fresh_fetch(
    mock_client: MagicMock,
    job_collector: collector.SlurmCollector,
):
    """Scrape duration is >= 0 when data is freshly fetched."""
    mock_client.get_jobs.return_value = []
    metrics = {m.name: m for m in job_collector.collect()}
    actual_duration = metrics["slurm_job_scrape_duration"].samples[0].value
    assert actual_duration >= 0.0


def test_collect_error_count_starts_at_zero(
    mock_client: MagicMock,
    job_collector: collector.SlurmCollector,
):
    """Error counter is 0 on a healthy scrape."""
    mock_client.get_jobs.return_value = []
    metrics = {m.name: m for m in job_collector.collect()}
    actual_errors = metrics["slurm_job_scrape_error"].samples[0].value
    assert actual_errors == 0


def test_collect_metric_prefix_in_metadata_names(
    mock_client: MagicMock,
    node_collector: collector.SlurmCollector,
):
    """Metadata metric names incorporate the configured metric_prefix."""
    mock_client.get_nodes.return_value = []
    metrics = {m.name: m for m in node_collector.collect()}
    assert "slurm_node_scrape_duration" in metrics
    assert "slurm_node_scrape_error" in metrics


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_collect_increments_error_on_fetch_failure(
    mock_client: MagicMock,
    job_collector: collector.SlurmCollector,
):
    """Error count increases by one for each failed fetch."""
    mock_client.get_jobs.side_effect = RuntimeError("connection refused")

    metrics_after_first = {m.name: m for m in job_collector.collect()}
    assert metrics_after_first["slurm_job_scrape_error"].samples[0].value == 1

    metrics_after_second = {m.name: m for m in job_collector.collect()}
    assert metrics_after_second["slurm_job_scrape_error"].samples[0].value == 2


def test_collect_omits_domain_metrics_on_fetch_failure(
    mock_client: MagicMock,
    job_collector: collector.SlurmCollector,
):
    """Domain metrics are not yielded when the fetcher raises."""
    mock_client.get_jobs.side_effect = RuntimeError("timeout")
    metrics = {m.name: m for m in job_collector.collect()}
    assert "slurm_job_info" not in metrics


def test_collect_duration_negative_on_fetch_failure(
    mock_client: MagicMock,
    job_collector: collector.SlurmCollector,
):
    """Scrape duration is -1 when the fetcher raises."""
    mock_client.get_jobs.side_effect = RuntimeError("timeout")
    metrics = {m.name: m for m in job_collector.collect()}
    actual_duration = metrics["slurm_job_scrape_duration"].samples[0].value
    assert actual_duration == -1.0


def test_collect_recovers_after_fetch_failure(
    mock_client: MagicMock,
    job_collector: collector.SlurmCollector,
):
    """Domain metrics reappear once the fetcher stops failing."""
    mock_client.get_jobs.side_effect = RuntimeError("down")
    list(job_collector.collect())  # first call fails

    raw_job = types.RawJobData(job_id=1, job_state="RUNNING")
    mock_client.get_jobs.side_effect = None
    mock_client.get_jobs.return_value = [raw_job]

    metrics = {m.name: m for m in job_collector.collect()}
    assert "slurm_job_info" in metrics
    assert len(metrics["slurm_job_info"].samples) == 1


def test_collect_error_count_persists_after_recovery(
    mock_client: MagicMock,
    job_collector: collector.SlurmCollector,
):
    """The error count is not reset when a fetch succeeds after a failure."""
    mock_client.get_jobs.side_effect = RuntimeError("blip")
    list(job_collector.collect())

    mock_client.get_jobs.side_effect = None
    mock_client.get_jobs.return_value = []
    metrics = {m.name: m for m in job_collector.collect()}
    actual_errors = metrics["slurm_job_scrape_error"].samples[0].value
    assert actual_errors == 1


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def test_collect_cache_hit_returns_negative_duration(mock_client: MagicMock):
    """Scrape duration is -1 on a cache hit (second call within poll_limit)."""
    cached = collector.SlurmCollector(
        fetcher=lambda: jobs.fetch(mock_client),
        generator=jobs.generate_metrics,
        metric_prefix="job",
        poll_limit=9999.0,
        scraper_description="test",
    )
    mock_client.get_jobs.return_value = []

    first = {m.name: m for m in cached.collect()}
    assert first["slurm_job_scrape_duration"].samples[0].value >= 0.0

    second = {m.name: m for m in cached.collect()}
    assert second["slurm_job_scrape_duration"].samples[0].value == -1.0


def test_collect_cache_hit_still_yields_domain_metrics(mock_client: MagicMock):
    """Cached data produces domain metrics on subsequent scrapes."""
    cached = collector.SlurmCollector(
        fetcher=lambda: jobs.fetch(mock_client),
        generator=jobs.generate_metrics,
        metric_prefix="job",
        poll_limit=9999.0,
        scraper_description="test",
    )
    raw_job = types.RawJobData(job_id=42, job_state="RUNNING")
    mock_client.get_jobs.return_value = [raw_job]
    list(cached.collect())  # populate cache

    metrics = {m.name: m for m in cached.collect()}
    actual_job_id = metrics["slurm_job_info"].samples[0].labels["job_id"]
    assert actual_job_id == str(raw_job.job_id)


def test_collect_cache_prevents_redundant_fetches(mock_client: MagicMock):
    """The fetcher is called only once while the cache is still fresh."""
    cached = collector.SlurmCollector(
        fetcher=lambda: jobs.fetch(mock_client),
        generator=jobs.generate_metrics,
        metric_prefix="job",
        poll_limit=9999.0,
        scraper_description="test",
    )
    mock_client.get_jobs.return_value = []

    list(cached.collect())
    list(cached.collect())
    list(cached.collect())

    mock_client.get_jobs.assert_called_once()


# ---------------------------------------------------------------------------
# Jobs pipeline integration
# ---------------------------------------------------------------------------


def test_collect_jobs_produces_job_info(
    mock_client: MagicMock,
    job_collector: collector.SlurmCollector,
):
    """The full jobs pipeline yields a slurm_job_info metric family."""
    raw_job = types.RawJobData(job_id=100, job_state="RUNNING")
    mock_client.get_jobs.return_value = [raw_job]
    metrics = {m.name: m for m in job_collector.collect()}
    assert "slurm_job_info" in metrics
    assert len(metrics["slurm_job_info"].samples) == 1


def test_collect_jobs_all_labels_from_raw_data(
    mock_client: MagicMock,
    job_collector: collector.SlurmCollector,
):
    """Raw job fields appear as correct labels after the full pipeline."""
    raw_job = types.RawJobData(
        job_id=200,
        partition="gpu",
        priority=500,
        nodes="node[001-004]",
        cpus=32,
        user_id=1000,
        user_name="alice",
        qos="high",
        memory_per_node=128000,
        job_state="RUNNING",
    )
    mock_client.get_jobs.return_value = [raw_job]

    metrics = {m.name: m for m in job_collector.collect()}
    labels = metrics["slurm_job_info"].samples[0].labels

    assert labels["job_id"] == str(raw_job.job_id)
    assert labels["partition"] == raw_job.partition
    assert labels["priority"] == str(raw_job.priority)
    assert labels["nodes"] == raw_job.nodes
    assert "user_id" not in labels
    assert labels["user_name"] == raw_job.user_name
    assert labels["qos"] == raw_job.qos
    assert labels["cpus"] == str(raw_job.cpus)
    assert labels["memory_mb"] == str(raw_job.memory_per_node)
    assert labels["state"] == raw_job.job_state


def test_collect_jobs_memory_per_node_preferred(
    mock_client: MagicMock,
    job_collector: collector.SlurmCollector,
):
    """memory_per_node is used for the memory_mb label when both fields are set."""
    raw_job = types.RawJobData(
        job_id=1,
        memory_per_node=64000,
        memory_per_cpu=2000,
        cpus=8,
    )
    mock_client.get_jobs.return_value = [raw_job]
    metrics = {m.name: m for m in job_collector.collect()}
    actual_memory = metrics["slurm_job_info"].samples[0].labels["memory_mb"]
    assert actual_memory == str(raw_job.memory_per_node)


def test_collect_jobs_memory_per_cpu_fallback(
    mock_client: MagicMock,
    job_collector: collector.SlurmCollector,
):
    """memory_per_cpu * cpus is used when memory_per_node is absent."""
    raw_job = types.RawJobData(
        job_id=1,
        memory_per_node=None,
        memory_per_cpu=4000,
        cpus=16,
    )
    assert raw_job.memory_per_cpu is not None
    assert raw_job.cpus is not None
    expected_memory = raw_job.memory_per_cpu * raw_job.cpus
    mock_client.get_jobs.return_value = [raw_job]
    metrics = {m.name: m for m in job_collector.collect()}
    actual_memory = metrics["slurm_job_info"].samples[0].labels["memory_mb"]
    assert actual_memory == str(expected_memory)


def test_collect_jobs_memory_zero_when_unset(
    mock_client: MagicMock,
    job_collector: collector.SlurmCollector,
):
    """memory_mb is '0' when neither memory field is set."""
    mock_client.get_jobs.return_value = [
        types.RawJobData(job_id=1),
    ]
    metrics = {m.name: m for m in job_collector.collect()}
    actual_memory = metrics["slurm_job_info"].samples[0].labels["memory_mb"]
    assert actual_memory == "0"


def test_collect_jobs_none_nodes_normalised(
    mock_client: MagicMock,
    job_collector: collector.SlurmCollector,
):
    """None nodes from the API appear as an empty string in the label."""
    mock_client.get_jobs.return_value = [
        types.RawJobData(job_id=1, nodes=None, job_state="PENDING"),
    ]
    metrics = {m.name: m for m in job_collector.collect()}
    actual_nodes = metrics["slurm_job_info"].samples[0].labels["nodes"]
    assert actual_nodes == ""


def test_collect_jobs_none_cpus_label_is_zero(
    mock_client: MagicMock,
    job_collector: collector.SlurmCollector,
):
    """Cpus label is '0' when the raw job has cpus=None."""
    mock_client.get_jobs.return_value = [
        types.RawJobData(job_id=1, cpus=None),
    ]
    metrics = {m.name: m for m in job_collector.collect()}
    actual_cpus = metrics["slurm_job_info"].samples[0].labels["cpus"]
    assert actual_cpus == "0"


def test_collect_jobs_empty_api_response(
    mock_client: MagicMock,
    job_collector: collector.SlurmCollector,
):
    """An empty API response yields a job_info family with no samples."""
    mock_client.get_jobs.return_value = []
    metrics = {m.name: m for m in job_collector.collect()}
    assert metrics["slurm_job_info"].samples == []


def test_collect_jobs_multiple_jobs_distinct_ids(
    mock_client: MagicMock,
    job_collector: collector.SlurmCollector,
):
    """Multiple jobs each appear as a distinct sample."""
    raw_jobs = [
        types.RawJobData(job_id=1, job_state="RUNNING"),
        types.RawJobData(job_id=2, job_state="PENDING"),
        types.RawJobData(job_id=3, job_state="COMPLETED"),
    ]
    mock_client.get_jobs.return_value = raw_jobs
    metrics = {m.name: m for m in job_collector.collect()}
    expected_ids = {str(j.job_id) for j in raw_jobs}
    actual_ids = {s.labels["job_id"] for s in metrics["slurm_job_info"].samples}
    assert actual_ids == expected_ids


def test_collect_jobs_sample_value_is_one(
    mock_client: MagicMock,
    job_collector: collector.SlurmCollector,
):
    """Every job info sample has value 1 (info-metric convention)."""
    mock_client.get_jobs.return_value = [
        types.RawJobData(job_id=1),
        types.RawJobData(job_id=2),
    ]
    metrics = {m.name: m for m in job_collector.collect()}
    for sample in metrics["slurm_job_info"].samples:
        assert sample.value == 1


def test_collect_jobs_count_per_state(
    mock_client: MagicMock,
    job_collector: collector.SlurmCollector,
):
    """Job state counts are accurate for jobs in different states."""
    running_jobs = [
        types.RawJobData(job_id=1, job_state="RUNNING"),
        types.RawJobData(job_id=2, job_state="RUNNING"),
        types.RawJobData(job_id=3, job_state="RUNNING"),
    ]
    pending_jobs = [
        types.RawJobData(job_id=4, job_state="PENDING"),
    ]
    completed_jobs = [
        types.RawJobData(job_id=5, job_state="COMPLETED"),
        types.RawJobData(job_id=6, job_state="COMPLETED"),
    ]
    mock_client.get_jobs.return_value = [*running_jobs, *pending_jobs, *completed_jobs]
    metrics = {m.name: m for m in job_collector.collect()}
    state_counts = {
        tuple(s.labels.values()): s.value
        for s in metrics["slurm_job_count_per_state"].samples
    }
    assert state_counts[("RUNNING",)] == len(running_jobs)
    assert state_counts[("PENDING",)] == len(pending_jobs)
    assert state_counts[("COMPLETED",)] == len(completed_jobs)


def test_collect_jobs_count_per_state_empty(
    mock_client: MagicMock,
    job_collector: collector.SlurmCollector,
):
    """An empty API response yields a job count per state family with no samples."""
    mock_client.get_jobs.return_value = []
    metrics = {m.name: m for m in job_collector.collect()}
    assert metrics["slurm_job_count_per_state"].samples == []


def test_collect_jobs_info_includes_state_label(
    mock_client: MagicMock,
    job_collector: collector.SlurmCollector,
):
    """The state label is present on slurm_job_info samples."""
    raw_job = types.RawJobData(job_id=1, job_state="PENDING")
    mock_client.get_jobs.return_value = [raw_job]
    metrics = {m.name: m for m in job_collector.collect()}
    labels = metrics["slurm_job_info"].samples[0].labels
    assert "state" in labels
    assert labels["state"] == "PENDING"


def test_collect_jobs_restart_count(
    mock_client: MagicMock,
    job_collector: collector.SlurmCollector,
):
    """Restart count metric reports the correct value per job."""
    raw_jobs = [
        types.RawJobData(
            job_id=1,
            job_state="RUNNING",
            restart_cnt=3,
            user_name="alice",
        ),
        types.RawJobData(
            job_id=2,
            job_state="PENDING",
            restart_cnt=0,
            user_name="bob",
        ),
        types.RawJobData(
            job_id=3,
            job_state="RUNNING",
            restart_cnt=7,
            user_name="alice",
        ),
    ]
    mock_client.get_jobs.return_value = raw_jobs
    metrics = {m.name: m for m in job_collector.collect()}
    restart_samples = {
        s.labels["job_id"]: s for s in metrics["slurm_job_restart_count"].samples
    }
    assert restart_samples["1"].value == 3
    assert restart_samples["1"].labels["user_name"] == "alice"
    assert restart_samples["2"].value == 0
    assert restart_samples["2"].labels["user_name"] == "bob"
    assert restart_samples["3"].value == 7
    assert restart_samples["3"].labels["user_name"] == "alice"


def test_collect_jobs_restart_count_empty(
    mock_client: MagicMock,
    job_collector: collector.SlurmCollector,
):
    """An empty API response yields a restart count family with no samples."""
    mock_client.get_jobs.return_value = []
    metrics = {m.name: m for m in job_collector.collect()}
    assert metrics["slurm_job_restart_count"].samples == []


def test_collect_jobs_restart_count_defaults_to_zero(
    mock_client: MagicMock,
    job_collector: collector.SlurmCollector,
):
    """Restart count defaults to 0 when not set in raw data."""
    mock_client.get_jobs.return_value = [
        types.RawJobData(job_id=1, job_state="RUNNING"),
    ]
    metrics = {m.name: m for m in job_collector.collect()}
    assert metrics["slurm_job_restart_count"].samples[0].value == 0


def test_collect_jobs_all_domain_families_present(
    mock_client: MagicMock,
    job_collector: collector.SlurmCollector,
):
    """The full jobs pipeline produces all expected metric families."""
    mock_client.get_jobs.return_value = [
        types.RawJobData(job_id=1, job_state="RUNNING"),
    ]
    metrics = {m.name: m for m in job_collector.collect()}
    expected = {
        "slurm_job_count_per_state",
        "slurm_job_restart_count",
        "slurm_job_info",
    }
    assert expected <= set(metrics.keys())


# ---------------------------------------------------------------------------
# Nodes pipeline integration
# ---------------------------------------------------------------------------


def test_collect_nodes_all_domain_families_present(
    mock_client: MagicMock,
    node_collector: collector.SlurmCollector,
):
    """The full nodes pipeline produces all expected metric families."""
    mock_client.get_nodes.return_value = [
        types.RawNodeData(
            name="n1",
            state="idle",
            cpus=64,
            gres="gpu:a100:4",
            gres_used="gpu:a100:0",
        ),
    ]
    metrics = {m.name: m for m in node_collector.collect()}
    expected = {
        "slurm_node_count_per_state",
        "slurm_node_cpus_total",
        "slurm_node_cpus_idle",
        "slurm_node_cpus_allocated",
        "slurm_node_cpu_load",
        "slurm_node_gpus_total",
        "slurm_node_gpus_idle",
        "slurm_node_gpus_allocated",
    }
    assert expected <= set(metrics.keys())


def test_collect_nodes_cpu_aggregation(
    mock_client: MagicMock,
    node_collector: collector.SlurmCollector,
):
    """CPU totals are aggregated across nodes through the full pipeline."""
    raw_n1 = types.RawNodeData(
        name="n1",
        state="idle",
        cpus=64,
        alloc_cpus=0,
    )
    raw_n2 = types.RawNodeData(
        name="n2",
        state="mixed",
        cpus=128,
        alloc_cpus=96,
    )
    mock_client.get_nodes.return_value = [raw_n1, raw_n2]

    expected_total = float(raw_n1.cpus + raw_n2.cpus)
    expected_alloc = float(raw_n1.alloc_cpus + raw_n2.alloc_cpus)
    expected_idle = expected_total - expected_alloc

    metrics = {m.name: m for m in node_collector.collect()}

    actual_total = metrics["slurm_node_cpus_total"].samples[0].value
    actual_alloc = metrics["slurm_node_cpus_allocated"].samples[0].value
    actual_idle = metrics["slurm_node_cpus_idle"].samples[0].value

    assert actual_total == pytest.approx(expected_total)
    assert actual_alloc == pytest.approx(expected_alloc)
    assert actual_idle == pytest.approx(expected_idle)


def test_collect_nodes_cpu_load_scaled(
    mock_client: MagicMock,
    node_collector: collector.SlurmCollector,
):
    """The raw cpu_load integer (value * 100) is scaled to a float."""
    raw_node = types.RawNodeData(
        name="n1",
        state="mixed",
        cpu_load=8750,
    )
    mock_client.get_nodes.return_value = [raw_node]
    expected_load = raw_node.cpu_load / 100.0

    metrics = {m.name: m for m in node_collector.collect()}
    actual_load = metrics["slurm_node_cpu_load"].samples[0].value
    assert actual_load == pytest.approx(expected_load)


def test_collect_nodes_gpu_parsed_and_aggregated(
    mock_client: MagicMock,
    node_collector: collector.SlurmCollector,
):
    """GPU GRES strings are parsed and idle GPUs are computed correctly."""
    raw_node = types.RawNodeData(
        name="gpu-node",
        state="mixed",
        gres="gpu:a100:4,gpu:v100:2",
        gres_used="gpu:a100:1,gpu:v100:0",
    )
    mock_client.get_nodes.return_value = [raw_node]
    metrics = {m.name: m for m in node_collector.collect()}

    expected_a100_total = 4.0
    expected_v100_total = 2.0
    expected_a100_alloc = 1.0
    expected_v100_alloc = 0.0

    total = {
        tuple(s.labels.values()): s.value
        for s in metrics["slurm_node_gpus_total"].samples
    }
    assert total[("a100",)] == pytest.approx(expected_a100_total)
    assert total[("v100",)] == pytest.approx(expected_v100_total)

    idle = {
        tuple(s.labels.values()): s.value
        for s in metrics["slurm_node_gpus_idle"].samples
    }
    expected_a100_idle = expected_a100_total - expected_a100_alloc
    expected_v100_idle = expected_v100_total - expected_v100_alloc
    assert idle[("a100",)] == pytest.approx(expected_a100_idle)
    assert idle[("v100",)] == pytest.approx(expected_v100_idle)

    alloc = {
        tuple(s.labels.values()): s.value
        for s in metrics["slurm_node_gpus_allocated"].samples
    }
    assert alloc[("a100",)] == pytest.approx(expected_a100_alloc)
    assert alloc[("v100",)] == pytest.approx(expected_v100_alloc)


def test_collect_nodes_gpu_summed_across_nodes(
    mock_client: MagicMock,
    node_collector: collector.SlurmCollector,
):
    """GPU counts for the same type are summed across multiple nodes."""
    raw_n1 = types.RawNodeData(
        name="n1",
        state="idle",
        gres="gpu:a100:4",
        gres_used="gpu:a100:0",
    )
    raw_n2 = types.RawNodeData(
        name="n2",
        state="idle",
        gres="gpu:a100:2",
        gres_used="gpu:a100:1",
    )
    mock_client.get_nodes.return_value = [raw_n1, raw_n2]

    # Expected values must be stated explicitly since they're parsed
    # from GRES strings — we can't derive them from the raw fields.
    expected_total = 4.0 + 2.0
    expected_alloc = 0.0 + 1.0

    metrics = {m.name: m for m in node_collector.collect()}
    total = {
        tuple(s.labels.values()): s.value
        for s in metrics["slurm_node_gpus_total"].samples
    }
    alloc = {
        tuple(s.labels.values()): s.value
        for s in metrics["slurm_node_gpus_allocated"].samples
    }
    assert total[("a100",)] == pytest.approx(expected_total)
    assert alloc[("a100",)] == pytest.approx(expected_alloc)


def test_collect_nodes_state_counting(
    mock_client: MagicMock,
    node_collector: collector.SlurmCollector,
):
    """Node state counts are accurate for a heterogeneous cluster."""
    idle_nodes = [
        types.RawNodeData(name="n1", state="idle"),
        types.RawNodeData(name="n2", state="idle"),
    ]
    down_nodes = [
        types.RawNodeData(name="n3", state="down"),
    ]
    mock_client.get_nodes.return_value = [*idle_nodes, *down_nodes]
    metrics = {m.name: m for m in node_collector.collect()}
    state_counts = {
        tuple(s.labels.values()): s.value
        for s in metrics["slurm_node_count_per_state"].samples
    }
    assert state_counts[("idle",)] == len(idle_nodes)
    assert state_counts[("down",)] == len(down_nodes)


def test_collect_nodes_no_gpu_node(
    mock_client: MagicMock,
    node_collector: collector.SlurmCollector,
):
    """A node without GPUs contributes no GPU samples."""
    mock_client.get_nodes.return_value = [
        types.RawNodeData(name="cpu-only", state="idle", cpus=64),
    ]
    metrics = {m.name: m for m in node_collector.collect()}
    assert metrics["slurm_node_gpus_total"].samples == []
    assert metrics["slurm_node_gpus_idle"].samples == []
    assert metrics["slurm_node_gpus_allocated"].samples == []


def test_collect_nodes_empty_api_response(
    mock_client: MagicMock,
    node_collector: collector.SlurmCollector,
):
    """An empty API response yields zero-value CPU metrics and no GPU samples."""
    mock_client.get_nodes.return_value = []
    metrics = {m.name: m for m in node_collector.collect()}
    assert metrics["slurm_node_cpus_total"].samples[0].value == 0.0
    assert metrics["slurm_node_gpus_total"].samples == []
