"""SLURM REST API client.

Provides HTTP client with token-based authentication, thread safety,
and automatic response validation using Pydantic models.
"""

import base64
import json
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import structlog

from .types import RawJobData, RawNodeData

logger = structlog.get_logger(__name__)

# The exporter might not work with any other API version.
DEFAULT_API_VERSION = "v0.0.38"

DEFAULT_TIMEOUT = 30.0


class ExpiredTokenError(Exception):
    """Raised when the Slurm JWT has expired."""


def validate_jwt_not_expired(token: str) -> None:
    """Check that a JWT token has not expired.

    Decodes the JWT payload without verifying the signature and checks
    the ``exp`` claim against the current time. Raises
    :class:`ExpiredTokenError` if the token is already past its
    expiration. If the token is not a valid JWT or has no ``exp`` claim,
    a warning is logged and execution continues.

    Args:
        token: The raw JWT string (header.payload.signature).

    Raises:
        ExpiredTokenError: If the token's ``exp`` claim is in the past.
    """
    parts = token.split(".")
    if len(parts) != 3:  # noqa: PLR2004
        logger.warning("Token does not appear to be a JWT, skipping expiry check")
        return

    try:
        # JWT base64url encoding omits padding; restore it
        payload_b64 = parts[1]
        padding = 4 - len(payload_b64) % 4
        if padding != 4:  # noqa: PLR2004
            payload_b64 += "=" * padding

        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
    except (ValueError, json.JSONDecodeError):
        logger.warning("Failed to decode JWT payload, skipping expiry check")
        return

    exp = payload.get("exp")
    if exp is None:
        logger.warning("JWT has no 'exp' claim, skipping expiry check")
        return

    now = time.time()
    if now >= exp:
        msg = f"Slurm JWT has expired (exp={exp}, now={int(now)})"
        raise ExpiredTokenError(msg)

    logger.info("JWT expiry validated", expires_in_seconds=int(exp - now))


class SlurmRestApiClient:
    """HTTP client for the SLURM REST API.

    Lightweight client that handles authentication, makes HTTP requests,
    validates responses, and returns Pydantic-validated data objects.
    Business logic and transformations are delegated to collectors.

    Thread-safe through thread-local storage of httpx.Client instances.
    Can be used as a context manager for automatic cleanup.
    """

    def __init__(
        self,
        base_url: str,
        token_file: str | Path | None = None,
        api_version: str = DEFAULT_API_VERSION,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        """Initialize the REST API client.

        Args:
            base_url: Base URL for the SLURM REST API (e.g., "http://localhost:6820").
            token_file: Path to file containing authentication token.
            api_version: SLURM REST API version (default: v0.0.38).
            timeout: Request timeout in seconds (default: 30.0).

        Raises:
            ValueError: If base_url is empty or timeout is not positive.
            FileNotFoundError: If token_file is specified but doesn't exist.
        """
        if not base_url:
            msg = "base_url cannot be empty"
            raise ValueError(msg)
        if timeout <= 0:
            msg = "timeout must be positive"
            raise ValueError(msg)

        self.base_url = base_url.rstrip("/")
        self.api_version = api_version
        self._timeout = timeout

        # Build headers
        self._headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        # Read token from file if provided
        if token_file:
            token_path = Path(token_file)
            if not token_path.exists():
                msg = f"Token file not found: {token_file}"
                raise FileNotFoundError(msg)
            token = token_path.read_text().strip()
            validate_jwt_not_expired(token)
            self._headers["X-SLURM-USER-TOKEN"] = token

        # Use thread-local storage for httpx.Client (thread safety)
        self._local = threading.local()

    @property
    def client(self) -> httpx.Client:
        """Get or create thread-local httpx client.

        Each thread gets its own httpx.Client instance for thread safety.
        Clients are created lazily and reused within the same thread.

        Returns:
            Thread-local httpx.Client instance.
        """
        if not hasattr(self._local, "client") or self._local.client.is_closed:
            self._local.client = httpx.Client(
                base_url=self.base_url,
                headers=self._headers,
                timeout=self._timeout,
            )
        return self._local.client

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and cleanup resources."""
        self.close()

    def close(self):
        """Close the thread-local HTTP client if open."""
        if hasattr(self._local, "client") and not self._local.client.is_closed:
            self._local.client.close()

    def _make_request(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make HTTP request to SLURM REST API.

        Handles request execution, error checking, and JSON parsing.
        Logs request details and duration.

        Args:
            endpoint: API endpoint path (e.g., "/slurm/v0.0.38/nodes").
            params: Optional query parameters.

        Returns:
            Raw JSON response as dictionary.

        Raises:
            httpx.HTTPError: If HTTP request fails.
            RuntimeError: If API returns errors in response.
        """
        start_time = time.time()
        params = params or {}

        try:
            logger.debug(
                "Making API request",
                method="GET",
                endpoint=endpoint,
                params=params,
            )
            response = self.client.get(endpoint, params=params)
            response.raise_for_status()
            duration = time.time() - start_time
            logger.debug("API request completed", duration_seconds=round(duration, 3))

            data = response.json()

            # Check for errors in response
            if errors := data.get("errors", []):
                error_messages = []
                for error in errors:
                    error_msg = error.get("error", str(error))
                    logger.error("API error response", error_message=error_msg)
                    error_messages.append(error_msg)
                msg = f"API returned errors: {'; '.join(error_messages)}"
                raise RuntimeError(msg)
            return data  # noqa: TRY300

        except httpx.HTTPError:
            duration = time.time() - start_time
            logger.exception(
                "API request failed",
                duration_seconds=round(duration, 3),
            )
            raise

    def get_nodes(self, update_time: int | None = None) -> list[RawNodeData]:
        """Fetch all node information from SLURM REST API.

        Args:
            update_time: Optional Unix timestamp to filter nodes changed
                since this time.

        Returns:
            List of validated RawNodeData objects.

        Raises:
            httpx.HTTPError: If HTTP request fails.
            RuntimeError: If API returns errors.
        """
        endpoint = f"/slurm/{self.api_version}/nodes"
        params = {}
        if update_time is not None:
            params["update_time"] = update_time

        data = self._make_request(endpoint=endpoint, params=params)

        # Parse nodes from JSON response using Pydantic validation
        nodes = []
        for node_data in data.get("nodes", []):
            node = RawNodeData.model_validate(node_data)
            nodes.append(node)
        return nodes

    def get_jobs(self, update_time: int | None = None) -> list[RawJobData]:
        """Fetch all job information from SLURM REST API.

        Args:
            update_time: Optional Unix timestamp to filter jobs changed since this time.

        Returns:
            List of validated RawJobData objects.

        Raises:
            httpx.HTTPError: If HTTP request fails.
            RuntimeError: If API returns errors.
        """
        endpoint = f"/slurm/{self.api_version}/jobs"
        params = {}
        if update_time is not None:
            params["update_time"] = update_time

        data = self._make_request(endpoint=endpoint, params=params)

        # Parse jobs from JSON response using Pydantic validation
        jobs = []
        for job_data in data.get("jobs", []):
            job = RawJobData.model_validate(job_data)
            jobs.append(job)
        return jobs
