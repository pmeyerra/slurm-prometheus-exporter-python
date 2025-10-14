"""HTTP server for the Slurm Prometheus Exporter."""

import json
import logging
import os
import pathlib

import prometheus_client
import prometheus_client.core
import pydantic
import starlette.applications
import starlette.requests
import starlette.responses
import starlette.routing
import structlog

from . import collector, slurmrestapi
from .collectors import jobs, nodes

CONFIG_ENV_VAR = "SLURM_EXPORTER_CONFIG_PATH"
logger = structlog.get_logger(__name__)


class ExporterConfig(pydantic.BaseModel):
    """Configuration for the Slurm Prometheus Exporter."""

    rest_api_url: str = pydantic.Field(description="Base URL for SLURM REST API")
    rest_api_token_file: str = pydantic.Field(
        description="Path to file containing API auth token",
    )
    rest_api_version: str = pydantic.Field(
        slurmrestapi.DEFAULT_API_VERSION,
        description="SLURM REST API version",
    )
    rest_api_timeout: float = pydantic.Field(
        slurmrestapi.DEFAULT_TIMEOUT,
        description="Request timeout in seconds",
    )
    port: int = pydantic.Field(9092, description="HTTP server port", gt=0, lt=65536)
    metrics_path: str = pydantic.Field(
        "/metrics",
        description="URL path for metrics endpoint",
    )
    poll_limit: float = pydantic.Field(
        120.0,
        description="Minimum seconds between cache refreshes",
        gt=0,
    )
    log_level: str = pydantic.Field("INFO", description="Logging level")


def configure_logging(log_level_name: str) -> None:
    """Configure structlog for logfmt output."""
    log_level = getattr(logging, log_level_name.upper(), logging.INFO)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.EventRenamer("msg"),
            structlog.processors.format_exc_info,
            structlog.processors.LogfmtRenderer(
                key_order=("timestamp", "level", "msg"),
            ),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def load_config(config_path: str) -> ExporterConfig:
    """Load configuration from JSON file."""
    path = pathlib.Path(config_path)
    if not path.exists():
        msg = f"Configuration file not found: {config_path}"
        raise FileNotFoundError(msg)

    with path.open("r") as f:
        data = json.load(f)

    return ExporterConfig(**data)


def create_registry_with_collectors(
    rest_client: slurmrestapi.SlurmRestApiClient,
    poll_limit: float,
) -> prometheus_client.core.CollectorRegistry:
    """Create a Prometheus registry with SLURM collectors.

    Creates a custom registry (not the global one) and registers node and job
    collectors. Dependencies are injected into fetcher functions at build time.

    Args:
        rest_client: Shared REST API client for all collectors.
        poll_limit: Minimum seconds between cache refreshes.

    Returns:
        Configured Prometheus registry with injected dependencies.
    """
    # Create a custom registry instead of using the global REGISTRY
    registry = prometheus_client.core.CollectorRegistry()

    # Create nodes collector with dependency injection
    # Lambda captures rest_client in closure, creating a zero-argument fetcher
    nodes_collector = collector.SlurmCollector(
        fetcher=lambda: nodes.fetch(rest_client),
        generator=nodes.generate_metrics,
        metric_prefix="node",
        poll_limit=poll_limit,
        scraper_description=f"REST API {rest_client.base_url}",
    )
    registry.register(nodes_collector)
    logger.info("Registered collector", collector="nodes", metric_prefix="node")

    # Create jobs collector with dependency injection
    # Lambda captures rest_client in closure, creating a zero-argument fetcher
    jobs_collector = collector.SlurmCollector(
        fetcher=lambda: jobs.fetch(rest_client),
        generator=jobs.generate_metrics,
        metric_prefix="job",
        poll_limit=poll_limit,
        scraper_description=f"REST API {rest_client.base_url}",
    )
    registry.register(jobs_collector)
    logger.info("Registered collector", collector="jobs", metric_prefix="job")

    return registry


def create_starlette_app(
    metrics_path: str,
    registry: prometheus_client.core.CollectorRegistry,
) -> starlette.applications.Starlette:
    """Create a Starlette application for serving Prometheus metrics.

    Args:
        metrics_path: URL path for metrics endpoint (e.g., "/metrics").
        registry: Prometheus collector registry.

    Returns:
        Configured Starlette application.
    """

    def metrics_endpoint(
        request: starlette.requests.Request,
    ) -> starlette.responses.Response:
        """Generate and serve Prometheus metrics.

        Args:
            request: The incoming HTTP request.

        Returns:
            PlainTextResponse with metrics in Prometheus exposition format.
        """
        metrics_output = prometheus_client.generate_latest(registry)
        logger.info(
            "HTTP request",
            client_ip=request.client.host if request.client else "unknown",
            method=request.method,
            path=request.url.path,
        )
        return starlette.responses.PlainTextResponse(
            content=metrics_output,
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    routes = [
        starlette.routing.Route(metrics_path, metrics_endpoint, methods=["GET"]),
    ]

    return starlette.applications.Starlette(routes=routes)


def create_exporter(config: ExporterConfig) -> starlette.applications.Starlette:
    """Construct the exporter ASGI app from validated config."""
    rest_client = slurmrestapi.SlurmRestApiClient(
        base_url=config.rest_api_url,
        token_file=config.rest_api_token_file,
        api_version=config.rest_api_version,
        timeout=config.rest_api_timeout,
    )
    logger.info("Created shared REST client", base_url=config.rest_api_url)

    registry = create_registry_with_collectors(
        rest_client=rest_client,
        poll_limit=config.poll_limit,
    )

    return create_starlette_app(
        metrics_path=config.metrics_path,
        registry=registry,
    )


def create_app(config_path: str | None = None) -> starlette.applications.Starlette:
    """Create the exporter ASGI app using a config path or environment default."""
    resolved_path = config_path or os.environ.get(CONFIG_ENV_VAR, "/config.json")
    config = load_config(resolved_path)
    configure_logging(config.log_level)
    return create_exporter(config)
