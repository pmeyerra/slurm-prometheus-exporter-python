"""Tests for total_memory_mb edge cases in the jobs collector.

These cover boundary conditions in the memory calculation that are hard to
exercise through the high-level SlurmCollector tests because all the
zero-result paths produce the same '0' label.  The happy paths (preference
for memory_per_node, fallback to memory_per_cpu * cpus, and the
nothing-set case) are tested via test_collector.py.
"""

from slurm_prometheus_exporter.collectors import jobs


def test_total_memory_mb_zero_memory_per_node_skipped():
    """memory_per_node of 0 is treated as unset and falls through."""
    memory_per_cpu = 1000
    cpus = 2
    expected_memory = memory_per_cpu * cpus
    metric = jobs.JobMetric(
        job_id=1,
        memory_per_node=0,
        memory_per_cpu=memory_per_cpu,
        cpus=cpus,
    )
    assert metric.total_memory_mb == expected_memory


def test_total_memory_mb_zero_memory_per_cpu_returns_zero():
    """memory_per_cpu of 0 with no memory_per_node returns 0."""
    metric = jobs.JobMetric(
        job_id=1,
        memory_per_node=None,
        memory_per_cpu=0,
        cpus=4,
    )
    assert metric.total_memory_mb == 0


def test_total_memory_mb_none_cpus_returns_zero():
    """memory_per_cpu set but cpus is None returns 0."""
    metric = jobs.JobMetric(
        job_id=1,
        memory_per_node=None,
        memory_per_cpu=2000,
        cpus=None,
    )
    assert metric.total_memory_mb == 0


def test_total_memory_mb_zero_cpus_returns_zero():
    """memory_per_cpu set but cpus is 0 returns 0."""
    metric = jobs.JobMetric(
        job_id=1,
        memory_per_node=None,
        memory_per_cpu=2000,
        cpus=0,
    )
    assert metric.total_memory_mb == 0
