# Multi-stage build for minimal final image
# Stage 1: Builder - Build wheel and install dependencies
FROM python:3.13-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN mkdir /app
COPY . /app

# Build the wheel
WORKDIR /app
RUN uv build --wheel --out-dir /dist

# Create venv and install the package with dependencies
ENV UV_PROJECT_ENVIRONMENT=/venv
ENV VIRTUAL_ENV=/venv
RUN uv venv
RUN uv pip install /dist/*.whl
RUN uv pip install uvicorn

# Stage 2: Runtime - Minimal image with installed package
FROM python:3.13-slim AS runtime

# Copy the virtual environment with the installed package
COPY --from=builder /venv /venv

ENV PATH="/venv/bin:$PATH"

# Expose default port
EXPOSE 9092

WORKDIR /
ENTRYPOINT ["uvicorn"]
CMD ["slurm_prometheus_exporter.server:create_app", "--factory", "--host", "0.0.0.0", "--port", "9092", "--log-level", "info", "--no-access-log"]
