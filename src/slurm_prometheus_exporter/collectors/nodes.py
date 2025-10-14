"""Node metrics collector for SLURM.

Fetches node information from the SLURM REST API and generates Prometheus
metrics for CPU, memory, GPU resources, and node states. Handles GPU type
extraction from GRES strings and aggregates metrics across all nodes.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field

import structlog
from prometheus_client.core import GaugeMetricFamily
from prometheus_client.metrics_core import Metric

from .. import slurmrestapi

logger = structlog.get_logger(__name__)


@dataclass
class NodeMetric:
    """Represents metrics for a single SLURM node.

    Contains normalized node data with computed fields for idle resources
    and parsed GPU information. Memory values are in bytes, CPU counts are
    floats to support fractional allocations.
    """

    name: str
    state: str
    cpus: float = 0.0
    alloc_cpus: float = 0.0
    idle_cpus: float = 0.0
    cpu_load: float = 0.0
    real_memory: float = 0.0
    free_memory: float = 0.0
    alloc_memory: float = 0.0
    gpus_by_type: dict[str, float] = field(default_factory=dict)
    alloc_gpus_by_type: dict[str, float] = field(default_factory=dict)
    idle_gpus_by_type: dict[str, float] = field(default_factory=dict)
    partitions: list[str] = field(default_factory=list)


def _parse_gres_gpus(gres_string: str) -> dict[str, float]:
    """Parse GPU types and counts from GRES string.

    Extracts GPU information from SLURM Generic Resource (GRES) strings,
    handling various formats including typed GPUs, socket bindings, and
    multiple GPU types. Non-GPU resources are ignored.

    GRES format examples:
        "gpu:2" -> {"generic": 2.0}
        "gpu:tesla:2" -> {"tesla": 2.0}
        "gpu:a100:4,gpu:v100:2" -> {"a100": 4.0, "v100": 2.0}
        "gpu:tesla:2(S:0-1)" -> {"tesla": 2.0}  (socket info ignored)

    Args:
        gres_string: GRES string from SLURM API.

    Returns:
        Dictionary mapping GPU type to total count.
    """
    if not gres_string:
        return {}

    gpu_counts: dict[str, float] = {}
    min_parts_for_gpu = 2  # Minimum parts needed: "gpu:count"

    # Split by comma to handle multiple GRES types
    for gres_item in gres_string.split(","):
        gres_cleaned = gres_item.strip()
        if not gres_cleaned.startswith("gpu"):
            continue

        # Remove parenthetical socket/core info if present
        if "(" in gres_cleaned:
            gres_cleaned = gres_cleaned.split("(")[0]

        # Parse gpu:model:count or gpu:count format
        parts = gres_cleaned.split(":")
        if len(parts) >= min_parts_for_gpu:
            # Last part is the count
            try:
                count = float(parts[-1])
            except ValueError:
                logger.warning(
                    "Failed to parse GPU count from GRES string",
                    gres_string=gres_string,
                )
                continue

            # Determine GPU type (generic if only gpu:count, else join middle)
            gpu_type = (
                "generic" if len(parts) == min_parts_for_gpu else ":".join(parts[1:-1])
            )

            # Accumulate counts for the same GPU type
            gpu_counts[gpu_type] = gpu_counts.get(gpu_type, 0.0) + count

    return gpu_counts


def _transform_node(raw: slurmrestapi.types.RawNodeData) -> NodeMetric:
    """Transform raw node data from API into NodeMetric.

    Applies business logic transformations including unit conversions
    (MB to bytes, cpu_load scaling), GRES parsing for GPU info, and
    computation of idle resources.

    Args:
        raw: Raw node data from SLURM REST API.

    Returns:
        Transformed NodeMetric with all derived fields.
    """
    # Convert cpu_load from integer (load * 100) to float
    cpu_load = float(raw.cpu_load) / 100.0 if raw.cpu_load else 0.0

    # Convert memory from MB to bytes
    real_memory = float(raw.real_memory) * 1e6
    free_memory = float(raw.free_memory) * 1e6
    alloc_memory = float(raw.alloc_memory) * 1e6

    # Calculate idle CPUs
    cpus = float(raw.cpus)
    alloc_cpus = float(raw.alloc_cpus)
    idle_cpus = cpus - alloc_cpus

    # Parse GPU information from GRES fields
    gpus_by_type = _parse_gres_gpus(raw.gres)
    alloc_gpus_by_type = _parse_gres_gpus(raw.gres_used)

    # Calculate idle GPUs by type
    idle_gpus_by_type = {}
    all_gpu_types = set(gpus_by_type.keys()) | set(alloc_gpus_by_type.keys())
    for gpu_type in all_gpu_types:
        total = gpus_by_type.get(gpu_type, 0.0)
        allocated = alloc_gpus_by_type.get(gpu_type, 0.0)
        idle_gpus_by_type[gpu_type] = total - allocated

    # Use hostname if available, otherwise fall back to name
    node_name = raw.hostname if raw.hostname else raw.name

    return NodeMetric(
        name=node_name,
        state=raw.state,
        cpus=cpus,
        alloc_cpus=alloc_cpus,
        idle_cpus=idle_cpus,
        cpu_load=cpu_load,
        real_memory=real_memory,
        free_memory=free_memory,
        alloc_memory=alloc_memory,
        gpus_by_type=gpus_by_type,
        alloc_gpus_by_type=alloc_gpus_by_type,
        idle_gpus_by_type=idle_gpus_by_type,
        partitions=raw.partitions if raw.partitions else [],
    )


@dataclass
class CPUSummaryMetric:
    """Aggregated CPU metrics across all nodes."""

    total: float = 0.0
    idle: float = 0.0
    allocated: float = 0.0
    load: float = 0.0


@dataclass
class GPUSummaryMetric:
    """Aggregated GPU metrics across all nodes, grouped by type."""

    total_by_type: dict[str, float] = field(default_factory=dict)
    idle_by_type: dict[str, float] = field(default_factory=dict)
    allocated_by_type: dict[str, float] = field(default_factory=dict)


def _aggregate_cpu_metrics(nodes: list[NodeMetric]) -> CPUSummaryMetric:
    """Aggregate CPU metrics across all nodes.

    Args:
        nodes: List of node metrics.

    Returns:
        Aggregated CPU metrics.
    """
    summary = CPUSummaryMetric()

    for node in nodes:
        summary.total += node.cpus
        summary.idle += node.idle_cpus
        summary.allocated += node.alloc_cpus
        summary.load += node.cpu_load

    return summary


def _count_nodes_by_state(nodes: list[NodeMetric]) -> dict[str, int]:
    """Count nodes grouped by state.

    Args:
        nodes: List of node metrics.

    Returns:
        Dictionary mapping state name to node count.
    """
    node_count_per_state: dict[str, int] = {}

    for node in nodes:
        node_count_per_state[node.state] = node_count_per_state.get(node.state, 0) + 1

    return node_count_per_state


def _aggregate_gpu_metrics(nodes: list[NodeMetric]) -> GPUSummaryMetric:
    """Aggregate GPU metrics across all nodes, grouped by GPU type.

    Args:
        nodes: List of node metrics.

    Returns:
        Aggregated GPU metrics by type.
    """
    summary = GPUSummaryMetric()

    for node in nodes:
        # Aggregate total GPUs by type
        for gpu_type, count in node.gpus_by_type.items():
            summary.total_by_type[gpu_type] = (
                summary.total_by_type.get(gpu_type, 0.0) + count
            )

        # Aggregate idle GPUs by type
        for gpu_type, count in node.idle_gpus_by_type.items():
            summary.idle_by_type[gpu_type] = (
                summary.idle_by_type.get(gpu_type, 0.0) + count
            )

        # Aggregate allocated GPUs by type
        for gpu_type, count in node.alloc_gpus_by_type.items():
            summary.allocated_by_type[gpu_type] = (
                summary.allocated_by_type.get(gpu_type, 0.0) + count
            )

    return summary


def fetch(client: slurmrestapi.SlurmRestApiClient) -> list[NodeMetric]:
    """Fetch node metrics from the SLURM REST API.

    Args:
        client: REST API client to use for fetching.

    Returns:
        List of node metrics.
    """
    raw_nodes = client.get_nodes()
    return [_transform_node(node) for node in raw_nodes]


def generate_metrics(nodes: list[NodeMetric]) -> Iterator[Metric]:
    """Generate Prometheus metrics from node data.

    Creates aggregate metrics for node counts by state, total/idle/allocated
    CPUs, CPU load, and GPUs by type.

    Args:
        nodes: List of node metrics.

    Yields:
        Prometheus Metric objects.
    """
    # Export node count per state metric
    node_count_by_state = _count_nodes_by_state(nodes)
    node_count_per_state = GaugeMetricFamily(
        "slurm_node_count_per_state",
        "nodes per state",
        labels=["state"],
    )
    for state, count in node_count_by_state.items():
        node_count_per_state.add_metric([state], count)
    yield node_count_per_state

    # Aggregate CPU metrics
    cpu_metrics = _aggregate_cpu_metrics(nodes)

    # Export total CPU metrics
    total_cpus = GaugeMetricFamily("slurm_cpus_total", "Total cpus")
    total_cpus.add_metric([], cpu_metrics.total)
    yield total_cpus

    total_idle_cpus = GaugeMetricFamily("slurm_cpus_idle", "Total idle cpus")
    total_idle_cpus.add_metric([], cpu_metrics.idle)
    yield total_idle_cpus

    total_allocated_cpus = GaugeMetricFamily(
        "slurm_cpus_allocated",
        "Total allocated cpus",
    )
    total_allocated_cpus.add_metric([], cpu_metrics.allocated)
    yield total_allocated_cpus

    total_cpu_load = GaugeMetricFamily("slurm_cpu_load", "Total cpu load")
    total_cpu_load.add_metric([], cpu_metrics.load)
    yield total_cpu_load

    # Aggregate GPU metrics
    gpu_metrics = _aggregate_gpu_metrics(nodes)

    # Export total GPU metrics with gpu_type label
    total_gpus = GaugeMetricFamily(
        "slurm_gpus_total",
        "Total gpus",
        labels=["gpu_type"],
    )
    for gpu_type, gpu_count in gpu_metrics.total_by_type.items():
        total_gpus.add_metric([gpu_type], gpu_count)
    yield total_gpus

    # Export idle GPU metrics with gpu_type label
    total_idle_gpus = GaugeMetricFamily(
        "slurm_gpus_idle",
        "Total idle gpus",
        labels=["gpu_type"],
    )
    for gpu_type, gpu_count in gpu_metrics.idle_by_type.items():
        total_idle_gpus.add_metric([gpu_type], gpu_count)
    yield total_idle_gpus

    # Export allocated GPU metrics with gpu_type label
    total_allocated_gpus = GaugeMetricFamily(
        "slurm_gpus_allocated",
        "Total allocated gpus",
        labels=["gpu_type"],
    )
    for gpu_type, gpu_count in gpu_metrics.allocated_by_type.items():
        total_allocated_gpus.add_metric([gpu_type], gpu_count)
    yield total_allocated_gpus
