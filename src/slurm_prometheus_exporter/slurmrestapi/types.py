"""Raw API response types for SLURM REST API.

Pydantic models representing the structure of data returned by the SLURM
REST API with minimal processing. These models provide validation and
type safety for API responses.
"""

from pydantic import BaseModel


class RawNodeData(BaseModel):
    """Raw node data from SLURM REST API.

    Represents the exact structure returned by the API with defaults for
    optional fields. Memory values are in MB, cpu_load is scaled by 100.
    """

    # Core identification
    name: str = ""
    hostname: str = ""

    # State information
    state: str = ""
    state_flags: list[str] | None = None

    # CPU information
    cpus: int = 0
    alloc_cpus: int = 0
    cpu_load: int = 0  # API returns as integer (load * 100)

    # Memory information (in MB)
    real_memory: int = 0
    free_memory: int = 0
    alloc_memory: int = 0

    # GRES (Generic Resources like GPUs)
    gres: str = ""
    gres_used: str = ""

    # Partitions
    partitions: list[str] | None = None


class RawJobData(BaseModel):
    """Raw job data from SLURM REST API.

    Represents the exact structure returned by the API with defaults for
    optional fields. Memory values are in MiB.
    """

    # Core identification
    job_id: int = 0

    # Job configuration
    partition: str = ""
    priority: int = 0
    nodes: str | None = None
    cpus: int | None = None

    # User information
    user_id: int = 0
    user_name: str = ""

    # QoS
    qos: str = ""

    # Memory configuration (in MiB)
    memory_per_node: int | None = None
    memory_per_cpu: int | None = None

    # Job state
    job_state: str = ""

    # Restart count
    restart_cnt: int = 0
