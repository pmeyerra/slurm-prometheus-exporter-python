"""Tests for node transform behaviour not observable through SlurmCollector.

The generate_metrics output aggregates across nodes and does not expose
per-node fields like hostname, memory (in bytes), or partitions.  GRES
parsing edge cases (socket info, non-GPU resources, whitespace, invalid
counts) are also difficult to verify through the aggregated output.  The
remaining transform and metric-generation logic is covered by the
high-level tests in test_collector.py.
"""

import pytest

from slurm_prometheus_exporter.collectors import nodes
from slurm_prometheus_exporter.slurmrestapi import types

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Hostname / name selection
# ---------------------------------------------------------------------------


def test_transform_node_hostname_preferred(raw_node_idle: types.RawNodeData):
    """Hostname is used as the node name when available."""
    metric = nodes._transform_node(raw_node_idle)
    assert metric.name == raw_node_idle.hostname


def test_transform_node_falls_back_to_name(raw_node_down: types.RawNodeData):
    """Node name is used when hostname is empty."""
    metric = nodes._transform_node(raw_node_down)
    expected_name = "node003"
    assert metric.name == expected_name


# ---------------------------------------------------------------------------
# Memory conversion
# ---------------------------------------------------------------------------


def test_transform_node_memory_to_bytes(raw_node_mixed: types.RawNodeData):
    """Memory values are converted from MB to bytes (x 1e6)."""
    metric = nodes._transform_node(raw_node_mixed)
    assert metric.real_memory == pytest.approx(512000 * 1e6)
    assert metric.free_memory == pytest.approx(128000 * 1e6)
    assert metric.alloc_memory == pytest.approx(384000 * 1e6)


# ---------------------------------------------------------------------------
# Partitions
# ---------------------------------------------------------------------------


def test_transform_node_none_partitions(raw_node_down: types.RawNodeData):
    """None partitions field is normalised to an empty list."""
    metric = nodes._transform_node(raw_node_down)
    assert metric.partitions == []


def test_transform_node_partitions_preserved(raw_node_idle: types.RawNodeData):
    """Partition list is passed through unchanged."""
    metric = nodes._transform_node(raw_node_idle)
    expected_partitions = ["batch", "gpu"]
    assert metric.partitions == expected_partitions


# ---------------------------------------------------------------------------
# GRES parsing edge cases
# ---------------------------------------------------------------------------


def test_transform_node_gres_socket_info_stripped():
    """Parenthetical socket binding info in GRES is stripped."""
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
    expected_gpus = {"a100": 4.0, "v100": 2.0}
    assert metric.gpus_by_type == expected_gpus


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
