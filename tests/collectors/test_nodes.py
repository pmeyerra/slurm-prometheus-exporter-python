"""Tests for the nodes collector module."""

from unittest.mock import MagicMock

import pytest
from prometheus_client.core import GaugeMetricFamily

from slurm_prometheus_exporter.collectors import nodes
from slurm_prometheus_exporter.slurmrestapi import types


@pytest.fixture
def raw_node_idle() -> types.RawNodeData:
    """A fully idle node with GPUs and no allocations."""
    return types.RawNodeData(
        name="node001",
        hostname="node001.cluster.local",
        state="idle",
        cpus=64,
        alloc_cpus=0,
        cpu_load=0,
        real_memory=256000,
        free_memory=256000,
        alloc_memory=0,
        gres="gpu:a100:4",
        gres_used="gpu:a100:0",
        partitions=["batch", "gpu"],
    )


@pytest.fixture
def raw_node_mixed() -> types.RawNodeData:
    """A partially allocated node with mixed GPU types."""
    return types.RawNodeData(
        name="node002",
        hostname="node002.cluster.local",
        state="mixed",
        cpus=128,
        alloc_cpus=96,
        cpu_load=8750,
        real_memory=512000,
        free_memory=128000,
        alloc_memory=384000,
        gres="gpu:a100:4,gpu:v100:2",
        gres_used="gpu:a100:2,gpu:v100:1",
        partitions=["batch"],
    )


@pytest.fixture
def raw_node_down() -> types.RawNodeData:
    """A down node with no resources available."""
    return types.RawNodeData(
        name="node003",
        hostname="",
        state="down",
        cpus=32,
        alloc_cpus=0,
        cpu_load=0,
        real_memory=128000,
        free_memory=0,
        alloc_memory=0,
        gres="",
        gres_used="",
        partitions=None,
    )


@pytest.fixture
def node_idle(raw_node_idle: types.RawNodeData) -> nodes.NodeMetric:
    """Transformed idle node metric."""
    return nodes._transform_node(raw_node_idle)


@pytest.fixture
def node_mixed(raw_node_mixed: types.RawNodeData) -> nodes.NodeMetric:
    """Transformed mixed-state node metric."""
    return nodes._transform_node(raw_node_mixed)


@pytest.fixture
def node_down(raw_node_down: types.RawNodeData) -> nodes.NodeMetric:
    """Transformed down node metric."""
    return nodes._transform_node(raw_node_down)


# ---------------------------------------------------------------------------
# transform_node
# ---------------------------------------------------------------------------


def test_transform_node_hostname_preferred(raw_node_idle: types.RawNodeData):
    """Hostname is used as the node name when available."""
    metric = nodes._transform_node(raw_node_idle)
    assert metric.name == raw_node_idle.name


def test_transform_node_falls_back_to_name(raw_node_down: types.RawNodeData):
    """Node name is used when hostname is empty."""
    metric = nodes._transform_node(raw_node_down)
    assert metric.name == "node003"


def test_transform_node_cpu_load_scaled(raw_node_mixed: types.RawNodeData):
    """CPU load integer (load * 100) is converted to a float."""
    metric = nodes._transform_node(raw_node_mixed)
    assert metric.cpu_load == pytest.approx(87.5)


def test_transform_node_zero_cpu_load(raw_node_idle: types.RawNodeData):
    """Zero CPU load stays zero after scaling."""
    metric = nodes._transform_node(raw_node_idle)
    assert metric.cpu_load == 0.0


def test_transform_node_memory_to_bytes(raw_node_mixed: types.RawNodeData):
    """Memory values are converted from MB to bytes (x 1e6)."""
    metric = nodes._transform_node(raw_node_mixed)
    assert metric.real_memory == pytest.approx(512000 * 1e6)
    assert metric.free_memory == pytest.approx(128000 * 1e6)
    assert metric.alloc_memory == pytest.approx(384000 * 1e6)


def test_transform_node_idle_cpus(raw_node_mixed: types.RawNodeData):
    """Idle CPUs equal total minus allocated."""
    metric = nodes._transform_node(raw_node_mixed)
    assert metric.idle_cpus == pytest.approx(128.0 - 96.0)


def test_transform_node_all_cpus_idle(raw_node_idle: types.RawNodeData):
    """Idle CPUs equal total when nothing is allocated."""
    metric = nodes._transform_node(raw_node_idle)
    assert metric.idle_cpus == pytest.approx(64.0)


def test_transform_node_gpu_parsing(raw_node_mixed: types.RawNodeData):
    """Total and allocated GPUs are parsed from GRES strings."""
    metric = nodes._transform_node(raw_node_mixed)
    assert metric.gpus_by_type == {"a100": 4.0, "v100": 2.0}
    assert metric.alloc_gpus_by_type == {"a100": 2.0, "v100": 1.0}


def test_transform_node_idle_gpus(raw_node_mixed: types.RawNodeData):
    """Idle GPUs per type equal total minus allocated."""
    metric = nodes._transform_node(raw_node_mixed)
    assert metric.idle_gpus_by_type == {"a100": 2.0, "v100": 1.0}


def test_transform_node_no_gres(raw_node_down: types.RawNodeData):
    """Empty GRES strings result in empty GPU dictionaries."""
    metric = nodes._transform_node(raw_node_down)
    assert metric.gpus_by_type == {}
    assert metric.alloc_gpus_by_type == {}
    assert metric.idle_gpus_by_type == {}


def test_transform_node_gres_socket_info_stripped():
    """Parenthetical socket binding info in GRES is stripped during parsing."""
    raw = types.RawNodeData(
        name="node-socket",
        state="idle",
        gres="gpu:tesla:2(S:0-1)",
        gres_used="gpu:tesla:0(S:0-1)",
    )
    metric = nodes._transform_node(raw)
    assert metric.gpus_by_type == {"tesla": 2.0}


def test_transform_node_gres_non_gpu_ignored():
    """Non-GPU GRES resources are ignored during transformation."""
    raw = types.RawNodeData(
        name="node-mps",
        state="idle",
        gres="mps:100,gpu:a100:4,shard:8",
        gres_used="mps:50,gpu:a100:1,shard:4",
    )
    metric = nodes._transform_node(raw)
    assert metric.gpus_by_type == {"a100": 4.0}
    assert metric.alloc_gpus_by_type == {"a100": 1.0}


def test_transform_node_gres_whitespace():
    """Whitespace around comma-separated GRES entries is handled."""
    raw = types.RawNodeData(
        name="node-ws",
        state="idle",
        gres="gpu:a100:4 , gpu:v100:2",
        gres_used="gpu:a100:0 , gpu:v100:0",
    )
    metric = nodes._transform_node(raw)
    assert metric.gpus_by_type == {"a100": 4.0, "v100": 2.0}


def test_transform_node_gres_invalid_count_skipped():
    """A GRES entry with a non-numeric GPU count is skipped gracefully."""
    raw = types.RawNodeData(
        name="node-bad",
        state="idle",
        gres="gpu:a100:abc,gpu:v100:2",
        gres_used="gpu:v100:0",
    )
    metric = nodes._transform_node(raw)
    assert metric.gpus_by_type == {"v100": 2.0}


def test_transform_node_none_partitions(raw_node_down: types.RawNodeData):
    """None partitions field is normalised to an empty list."""
    metric = nodes._transform_node(raw_node_down)
    assert metric.partitions == []


def test_transform_node_partitions_preserved(raw_node_idle: types.RawNodeData):
    """Partition list is passed through unchanged."""
    metric = nodes._transform_node(raw_node_idle)
    assert metric.partitions == ["batch", "gpu"]


def test_transform_node_state_preserved(raw_node_mixed: types.RawNodeData):
    """State string is passed through unchanged."""
    metric = nodes._transform_node(raw_node_mixed)
    assert metric.state == "mixed"


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------


def test_fetch_transforms_all_nodes():
    """Each raw node returned by the client is transformed into a NodeMetric."""
    raw_nodes = [
        types.RawNodeData(name="n1", state="idle", cpus=8),
        types.RawNodeData(name="n2", state="down", cpus=16),
    ]
    mock_client = MagicMock()
    mock_client.get_nodes.return_value = raw_nodes

    result = nodes.fetch(mock_client)

    assert len(result) == 2
    assert all(isinstance(n, nodes.NodeMetric) for n in result)
    mock_client.get_nodes.assert_called_once()


def test_fetch_empty_response():
    """An empty API response produces an empty list."""
    mock_client = MagicMock()
    mock_client.get_nodes.return_value = []

    result = nodes.fetch(mock_client)

    assert result == []


# ---------------------------------------------------------------------------
# generate_metrics
# ---------------------------------------------------------------------------


def test_generate_metrics_all_families_present(node_mixed: nodes.NodeMetric):
    """All documented metric families are yielded."""
    metrics = {m.name: m for m in nodes.generate_metrics([node_mixed])}
    expected_names = {
        "slurm_node_count_per_state",
        "slurm_cpus_total",
        "slurm_cpus_idle",
        "slurm_cpus_allocated",
        "slurm_cpu_load",
        "slurm_gpus_total",
        "slurm_gpus_idle",
        "slurm_gpus_allocated",
    }
    assert set(metrics.keys()) == expected_names


def test_generate_metrics_node_count_per_state(
    node_idle: nodes.NodeMetric,
    node_mixed: nodes.NodeMetric,
    node_down: nodes.NodeMetric,
):
    """State counts are accurate for a heterogeneous cluster."""
    metrics = {
        m.name: m for m in nodes.generate_metrics([node_idle, node_mixed, node_down])
    }
    samples = {
        tuple(s.labels.values()): s.value
        for s in metrics["slurm_node_count_per_state"].samples
    }
    assert samples[("idle",)] == 1
    assert samples[("mixed",)] == 1
    assert samples[("down",)] == 1


def test_generate_metrics_duplicate_state_counted(node_idle: nodes.NodeMetric):
    """Nodes sharing a state have their counts summed."""
    metrics = {m.name: m for m in nodes.generate_metrics([node_idle, node_idle])}
    samples = {
        tuple(s.labels.values()): s.value
        for s in metrics["slurm_node_count_per_state"].samples
    }
    assert samples[("idle",)] == 2


def test_generate_metrics_cpu_values(
    node_idle: nodes.NodeMetric,
    node_mixed: nodes.NodeMetric,
):
    """Aggregated CPU gauges reflect the sum across all nodes."""
    metrics = {m.name: m for m in nodes.generate_metrics([node_idle, node_mixed])}

    total = metrics["slurm_cpus_total"].samples[0].value
    idle = metrics["slurm_cpus_idle"].samples[0].value
    allocated = metrics["slurm_cpus_allocated"].samples[0].value
    load = metrics["slurm_cpu_load"].samples[0].value

    assert total == pytest.approx(64.0 + 128.0)
    assert idle == pytest.approx(64.0 + 32.0)
    assert allocated == pytest.approx(0.0 + 96.0)
    assert load == pytest.approx(0.0 + 87.5)


def test_generate_metrics_gpu_values(node_mixed: nodes.NodeMetric):
    """GPU gauges have correct per-type values."""
    metrics = {m.name: m for m in nodes.generate_metrics([node_mixed])}

    total_samples = {
        tuple(s.labels.values()): s.value for s in metrics["slurm_gpus_total"].samples
    }
    assert total_samples[("a100",)] == pytest.approx(4.0)
    assert total_samples[("v100",)] == pytest.approx(2.0)

    idle_samples = {
        tuple(s.labels.values()): s.value for s in metrics["slurm_gpus_idle"].samples
    }
    assert idle_samples[("a100",)] == pytest.approx(2.0)
    assert idle_samples[("v100",)] == pytest.approx(1.0)

    alloc_samples = {
        tuple(s.labels.values()): s.value
        for s in metrics["slurm_gpus_allocated"].samples
    }
    assert alloc_samples[("a100",)] == pytest.approx(2.0)
    assert alloc_samples[("v100",)] == pytest.approx(1.0)


def test_generate_metrics_gpu_summed_across_nodes(
    node_idle: nodes.NodeMetric,
    node_mixed: nodes.NodeMetric,
):
    """GPU counts for the same type are summed across nodes."""
    metrics = {m.name: m for m in nodes.generate_metrics([node_idle, node_mixed])}
    total_samples = {
        tuple(s.labels.values()): s.value for s in metrics["slurm_gpus_total"].samples
    }
    assert total_samples[("a100",)] == pytest.approx(4.0 + 4.0)


def test_generate_metrics_empty_list():
    """An empty node list produces zero-value CPU metrics and no GPU samples."""
    metrics = {m.name: m for m in nodes.generate_metrics([])}

    assert metrics["slurm_cpus_total"].samples[0].value == 0.0
    assert metrics["slurm_cpus_idle"].samples[0].value == 0.0
    assert metrics["slurm_cpus_allocated"].samples[0].value == 0.0
    assert metrics["slurm_cpu_load"].samples[0].value == 0.0
    assert metrics["slurm_gpus_total"].samples == []


def test_generate_metrics_all_gauge_families(node_idle: nodes.NodeMetric):
    """Every yielded metric is a GaugeMetricFamily instance."""
    for metric in nodes.generate_metrics([node_idle]):
        assert isinstance(metric, GaugeMetricFamily)


def test_generate_metrics_no_gpu_node(node_down: nodes.NodeMetric):
    """A node with no GPUs contributes no GPU samples."""
    metrics = {m.name: m for m in nodes.generate_metrics([node_down])}
    assert metrics["slurm_gpus_total"].samples == []
    assert metrics["slurm_gpus_idle"].samples == []
    assert metrics["slurm_gpus_allocated"].samples == []
