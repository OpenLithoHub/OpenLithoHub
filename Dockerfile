# syntax=docker/dockerfile:1.6
# ---------------------------------------------------------------------------
# Image tiering (repair-plan §9 / PR-B):
#
#   runtime     (default)  CLI/core image — fat dev-convenient venv
#                          (data + models + workflow + jupyter)
#   server                  the fat runtime venv + HTTP [server] extra
#   server-cpu              MINIMAL production server: core + workflow +
#                          required models + server; no jupyter, no dataset
#                          clients, no developer tooling
#
# The wheel is built EXACTLY ONCE per variant branch and installed into
# every target from /wheels — never re-fetched or rebuilt per target.
# ---------------------------------------------------------------------------
# Digest pinned (audit P1.11); Dependabot (docker ecosystem) keeps it current.
FROM python:3.12-slim@sha256:2f17fc044b579bab302c2e8054d3a686e2cb9a83de48e70534b94cd8ebbe06a9 AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY scripts/write_build_info.py /tmp/write_build_info.py
COPY src/ src/

ARG VERSION=0.0.0
ARG BUILD_COMMIT=unknown
# P0.9/P0.11: bake the build identity into the wheel bytes so the running
# container reports its true provenance without a .git checkout.
RUN SETUPTOOLS_SCM_PRETEND_VERSION=${VERSION} \
    python /tmp/write_build_info.py \
      --commit "${BUILD_COMMIT}" --version "${VERSION}" \
      --out src/openlithohub/_build.py

# Build into a relocatable venv so the runtime stage can copy /opt/venv
# wholesale; also build a wheel so the server stages can add the [server]
# extra on top of EXACTLY the same package bytes (never a PyPI re-fetch).
# Fat variant: data + models + workflow + jupyter (dev-convenient CLI).
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip \
 && SETUPTOOLS_SCM_PRETEND_VERSION=${VERSION} \
    /opt/venv/bin/pip install --no-cache-dir ".[data,models,workflow,jupyter]" \
 && SETUPTOOLS_SCM_PRETEND_VERSION=${VERSION} \
    /opt/venv/bin/pip wheel --no-deps -w /wheels .

# ---------------------------------------------------------------------------
# builder-slim: minimal dependency set for the server-cpu target — core +
# workflow (klayout/scipy) + model registry deps; NO jupyter, NO dataset
# clients. Same wheel bytes as the fat builder.
# ---------------------------------------------------------------------------
FROM python:3.12-slim@sha256:2f17fc044b579bab302c2e8054d3a686e2cb9a83de48e70534b94cd8ebbe06a9 AS builder-slim

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY scripts/write_build_info.py /tmp/write_build_info.py
COPY src/ src/

ARG VERSION=0.0.0
ARG BUILD_COMMIT=unknown
RUN SETUPTOOLS_SCM_PRETEND_VERSION=${VERSION} \
    python /tmp/write_build_info.py \
      --commit "${BUILD_COMMIT}" --version "${VERSION}" \
      --out src/openlithohub/_build.py

RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip \
 && SETUPTOOLS_SCM_PRETEND_VERSION=${VERSION} \
    /opt/venv/bin/pip install --no-cache-dir ".[models,workflow]" \
 && SETUPTOOLS_SCM_PRETEND_VERSION=${VERSION} \
    /opt/venv/bin/pip wheel --no-deps -w /wheels .

# ---------------------------------------------------------------------------
# runtime-base: the shared non-root runtime skeleton — KLayout's Qt deps,
# non-root user, scratch dir. No venv; targets add their own.
# ---------------------------------------------------------------------------
FROM python:3.12-slim@sha256:2f17fc044b579bab302c2e8054d3a686e2cb9a83de48e70534b94cd8ebbe06a9 AS runtime-base

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

WORKDIR /app
USER openlithohub

# ---------------------------------------------------------------------------
# Stage: runtime (default target) — fat CLI image.
# ---------------------------------------------------------------------------
FROM runtime-base AS runtime

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

ENTRYPOINT ["openlithohub"]
CMD ["--help"]

# ---------------------------------------------------------------------------
# Stage: server — the fat runtime image plus the HTTP service extra.
# Build/publish with:  docker build --target server -t openlithohub:server .
# The server adds fastapi/uvicorn and an HTTP healthcheck.
# ---------------------------------------------------------------------------
FROM runtime AS server

# P4.1: the CLI image does not carry build wheels; the server stage
# copies them straight from the builder and deletes them after install.
COPY --from=builder /wheels /wheels

USER root
RUN set -eux; \
    whl="$(ls /wheels/openlithohub-*.whl)"; \
    # --no-deps + explicit extra pins: the venv already satisfies every
    # wheel dependency, so resolution is skipped on purpose — install stays
    # deterministic and offline. (Since PR-B, the wheel itself carries no
    # VCS requirement either: core/server metadata is git-free.)
    /opt/venv/bin/pip install --no-deps --no-cache-dir "${whl}[server]"; \
    /opt/venv/bin/pip install --no-cache-dir \
        "fastapi>=0.110" "uvicorn[standard]>=0.27" "python-multipart>=0.0.9"; \
    rm -rf /wheels; \
    chown -R openlithohub:openlithohub /opt/venv
USER openlithohub

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/v1/health', timeout=3).status == 200 else 1)"]

ENTRYPOINT ["openlithohub"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]

# ---------------------------------------------------------------------------
# Stage: server-cpu — MINIMAL production server image (repair-plan §9):
# core + workflow + required models + server. Excludes jupyter, dataset
# clients, docs tooling and developer tools. Build with:
#   docker build --target server-cpu -t openlithohub:server-cpu .
# ---------------------------------------------------------------------------
FROM runtime-base AS server-cpu

COPY --from=builder-slim /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

COPY --from=builder-slim /wheels /wheels

USER root
RUN set -eux; \
    whl="$(ls /wheels/openlithohub-*.whl)"; \
    # --no-deps + explicit extra pins: same determinism contract as the
    # fat server stage; the slim venv already satisfies the wheel deps.
    /opt/venv/bin/pip install --no-deps --no-cache-dir "${whl}[server]"; \
    /opt/venv/bin/pip install --no-cache-dir \
        "fastapi>=0.110" "uvicorn[standard]>=0.27" "python-multipart>=0.0.9"; \
    rm -rf /wheels; \
    chown -R openlithohub:openlithohub /opt/venv
USER openlithohub

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/v1/health', timeout=3).status == 200 else 1)"]

ENTRYPOINT ["openlithohub"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
