# syntax=docker/dockerfile:1.6
# ---------------------------------------------------------------------------
# Stage 1: builder — install deps + the openlithohub wheel into a venv.
# ---------------------------------------------------------------------------
# Digest pinned (audit P1.11); Dependabot (docker ecosystem) keeps it current.
FROM python:3.12-slim@sha256:2f17fc044b579bab302c2e8054d3a686e2cb9a83de48e70534b94cd8ebbe06a9 AS builder

WORKDIR /build

ARG VERSION=0.0.0

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY src/ src/

# Build into a relocatable venv so the runtime stage can copy /opt/venv
# wholesale; also build a wheel so the `server` stage can add the [server]
# extra on top of EXACTLY the same package bytes (never a PyPI re-fetch).
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip \
 && SETUPTOOLS_SCM_PRETEND_VERSION=${VERSION} \
    /opt/venv/bin/pip install --no-cache-dir ".[data,models,workflow,jupyter]" \
 && SETUPTOOLS_SCM_PRETEND_VERSION=${VERSION} \
    /opt/venv/bin/pip wheel --no-deps -w /wheels .

# ---------------------------------------------------------------------------
# Stage 2: runtime — slim image with KLayout's Qt deps + the venv.
# ---------------------------------------------------------------------------
FROM python:3.12-slim@sha256:2f17fc044b579bab302c2e8054d3a686e2cb9a83de48e70534b94cd8ebbe06a9 AS runtime

WORKDIR /app

# KLayout's Python wheel ships its own .so files but links against the
# system libstdc++/libgomp/libGL/Qt — install the minimum runtime set.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libqt5core5a \
        libqt5gui5 \
        libqt5widgets5 \
        libxrender1 \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Non-root runtime user (audit P1.10) with writable HOME for caches and
# a dedicated scratch directory for request temp files.
RUN useradd --create-home --shell /usr/sbin/nologin openlithohub \
    && mkdir -p /app/scratch \
    && chown -R openlithohub:openlithohub /app
ENV HOME=/home/openlithohub \
    OLH_SCRATCH_DIR=/app/scratch

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /wheels /wheels
ENV PATH="/opt/venv/bin:${PATH}"

USER openlithohub
WORKDIR /app

ENTRYPOINT ["openlithohub"]
CMD ["--help"]

# ---------------------------------------------------------------------------
# Stage 3: server — the runtime image plus the HTTP service extra.
# Build/publish with:  docker build --target server -t openlithohub:server .
# The default (last) target remains the CLI image; the server adds
# fastapi/uvicorn and an HTTP healthcheck.
# ---------------------------------------------------------------------------
FROM runtime AS server

USER root
RUN /opt/venv/bin/pip install --no-cache-dir '/wheels/openlithohub'*.whl'[server]' \
    && chown -R openlithohub:openlithohub /opt/venv
USER openlithohub

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/v1/health', timeout=3).status == 200 else 1)"]

ENTRYPOINT ["openlithohub"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
