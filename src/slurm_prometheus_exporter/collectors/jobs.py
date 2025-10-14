"""Job metrics collector for SLURM.

Fetches job information from the SLURM REST API and generates Prometheus
metrics including job info with labels for partition, user, QoS, resources, etc.
"""

from collections.abc import Iterator
from dataclasses import dataclass

from prometheus_client.core import GaugeMetricFamily
from prometheus_client.metrics_core import Metric

from .. import slurmrestapi


@dataclass
class JobMetric:
    """Represents metrics for a single SLURM job.

    Normalized job data from the SLURM API with computed fields.
    """

    job_id: int
    partition: str = ""
    priority: int = 0
    nodes: str = ""
    user_id: int = 0
    user_name: str = ""
    qos: str = ""
    cpus: int | None = None
    memory_per_node: int | None = None
    memory_per_cpu: int | None = None
    job_state: str = ""

    @property
    def total_memory_mb(self) -> int:
        """Calculate total memory in MiB.

        Returns memory_per_node if set, otherwise memory_per_cpu * cpus,
        or 0 if neither is available.
        """
        if self.memory_per_node is not None and self.memory_per_node > 0:
            return self.memory_per_node
        if (
            self.memory_per_cpu is not None
            and self.memory_per_cpu > 0
            and self.cpus is not None
            and self.cpus > 0
        ):
            return self.memory_per_cpu * self.cpus
        return 0


def _transform_job(raw: slurmrestapi.types.RawJobData) -> JobMetric:
    """Transform raw job data from API into JobMetric.

    Args:
        raw: Raw job data from SLURM REST API.

    Returns:
        Transformed JobMetric with normalized fields.
    """
    return JobMetric(
        job_id=raw.job_id,
        partition=raw.partition,
        priority=raw.priority,
        nodes=raw.nodes or "",
        user_id=raw.user_id,
        user_name=raw.user_name,
        qos=raw.qos,
        cpus=raw.cpus,
        memory_per_node=raw.memory_per_node,
        memory_per_cpu=raw.memory_per_cpu,
        job_state=raw.job_state,
    )


def fetch(client: slurmrestapi.SlurmRestApiClient) -> list[JobMetric]:
    """Fetch job metrics from the SLURM REST API.

    Args:
        client: REST API client to use for fetching.

    Returns:
        List of job metrics.
    """
    raw_jobs = client.get_jobs()
    return [_transform_job(job) for job in raw_jobs]


def generate_metrics(jobs: list[JobMetric]) -> Iterator[Metric]:
    """Generate Prometheus metrics from job data.

    Creates a single slurm_job_info gauge metric with labels for each job's
    attributes (partition, user, QoS, resources, etc.).

    Args:
        jobs: List of job metrics.

    Yields:
        Prometheus Metric objects.
    """
    # Export job info metric
    job_info = GaugeMetricFamily(
        "slurm_job_info",
        "Information about Slurm jobs",
        labels=[
            "job_id",
            "partition",
            "priority",
            "nodes",
            "user_id",
            "user_name",
            "qos",
            "cpus",
            "memory_mb",
        ],
    )

    for job in jobs:
        job_info.add_metric(
            [
                str(job.job_id),
                job.partition,
                str(job.priority),
                job.nodes,
                str(job.user_id),
                job.user_name,
                job.qos,
                str(job.cpus or 0),
                str(job.total_memory_mb),
            ],
            1,
        )

    yield job_info
