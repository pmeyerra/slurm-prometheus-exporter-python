"""SLURM REST API client package.

Provides a lightweight HTTP client for the SLURM REST API that returns
raw, validated API response types with minimal processing. Business logic
and metric transformations are handled by collector modules.

Exports:
    SlurmRestApiClient: HTTP client with authentication and error handling.
    types: Module containing Pydantic models for API responses.
    DEFAULT_API_VERSION: Default SLURM REST API version.
    DEFAULT_TIMEOUT: Default HTTP request timeout.
"""

from . import types
from .client import (
    DEFAULT_API_VERSION,
    DEFAULT_TIMEOUT,
    ExpiredTokenError,
    SlurmRestApiClient,
)

__all__ = [
    "DEFAULT_API_VERSION",
    "DEFAULT_TIMEOUT",
    "ExpiredTokenError",
    "SlurmRestApiClient",
    "types",
]
