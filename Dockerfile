# syntax=docker/dockerfile:1.6
# ---------------------------------------------------------------------------
# Image tiering (repair-plan §9 / PR-B, revised per PR-B review):
#
#   wheel-builder            builds THE ONE canonical openlithohub wheel.
#                            Every other stage consumes that wheel — no
#                            stage source-installs or re-builds the package.
#
#   wheel-builder ──┬─> fat-env-builder        wheel[data,models,workflow,jupyter]
#                   │     └─> runtime  (default CLI image)
#                   │         └─> server (fat CLI + [server])
#                   ├─> server-env-builder    wheel[data,models,workflow,jupyter,server]
#                   └─> server-cpu-env-builder
#                         CPU torch FIRST (download.pytorch.org/whl/cpu),
#                         then wheel[models,workflow,server]
#                         └─> server-cpu (minimal CPU production server)
#
# ALL published targets are genuine CPU images: torch is installed from the
# official CPU wheel index BEFORE the openlithohub wheel, so the
# "torch>=2.12" requirement is satisfied by the +cpu build and pip never
# pulls the CUDA stack. Gates in docker.yml assert
# torch.version.cuda is None and the absence of nvidia-*/jupyter.
# ---------------------------------------------------------------------------
# Digest pinned (audit P1.11); Dependabot (docker ecosystem) keeps it current.
FROM python:3.14-slim@sha256:caaf356f40667c496d405780745b9ac25771c189a51dfcc42430d531ea09f8a2 AS wheel-builder

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

# THE canonical package build. Exactly one `pip wheel` invocation exists in
# this Dockerfile; every image tier installs the artifact produced here.
RUN SETUPTOOLS_SCM_PRETEND_VERSION=${VERSION} \
    pip wheel --no-deps --no-cache-dir -w /wheels .

# ---------------------------------------------------------------------------
# fat-env-builder: dev-convenient CLI dependency set (data + models +
# workflow + jupyter) on genuine CPU torch, from the canonical wheel.
# ---------------------------------------------------------------------------
FROM python:3.14-slim@sha256:caaf356f40667c496d405780745b9ac25771c189a51dfcc42430d531ea09f8a2 AS fat-env-builder

COPY --from=wheel-builder /wheels /wheels

RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip \
 # Genuine CPU torch FIRST: the +cpu wheel satisfies "torch>=2.12", so the
 # openlithohub install below can never replace it with the CUDA stack.
 && /opt/venv/bin/pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
      "torch>=2.12" \
 && whl="$(ls /wheels/openlithohub-*.whl)" \
 && /opt/venv/bin/pip install --no-cache-dir "${whl}[data,models,workflow,jupyter]" \
 && rm -rf /wheels

# ---------------------------------------------------------------------------
# server-env-builder: the fat server set (adds the [server] extra).
# ---------------------------------------------------------------------------
FROM python:3.14-slim@sha256:caaf356f40667c496d405780745b9ac25771c189a51dfcc42430d531ea09f8a2 AS server-env-builder

COPY --from=wheel-builder /wheels /wheels

RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip \
 && /opt/venv/bin/pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
      "torch>=2.12" \
 && whl="$(ls /wheels/openlithohub-*.whl)" \
 && /opt/venv/bin/pip install --no-cache-dir "${whl}[data,models,workflow,jupyter,server]" \
 && rm -rf /wheels

# ---------------------------------------------------------------------------
# server-cpu-env-builder: MINIMAL production dependency set (models +
# workflow + server; no jupyter, no dataset clients) on genuine CPU torch.
# ---------------------------------------------------------------------------
FROM python:3.14-slim@sha256:caaf356f40667c496d405780745b9ac25771c189a51dfcc42430d531ea09f8a2 AS server-cpu-env-builder

COPY --from=wheel-builder /wheels /wheels

RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip \
 && /opt/venv/bin/pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
      "torch>=2.12" \
 && whl="$(ls /wheels/openlithohub-*.whl)" \
 && /opt/venv/bin/pip install --no-cache-dir "${whl}[models,workflow,server]" \
 && rm -rf /wheels

# ---------------------------------------------------------------------------
# runtime-base: the shared non-root runtime skeleton — KLayout's Qt deps,
# non-root user, scratch dir. No venv; targets add their own.
# ---------------------------------------------------------------------------
FROM python:3.14-slim@sha256:caaf356f40667c496d405780745b9ac25771c189a51dfcc42430d531ea09f8a2 AS runtime-base

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
# Stage: runtime (default target) — fat CLI image, genuine CPU torch.
# ---------------------------------------------------------------------------
FROM runtime-base AS runtime

COPY --from=fat-env-builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

ENTRYPOINT ["openlithohub"]
CMD ["--help"]

# ---------------------------------------------------------------------------
# Stage: server — fat CLI + HTTP service extra, pre-built as one venv.
# Build/publish with:  docker build --target server -t openlithohub:server .
# ---------------------------------------------------------------------------
FROM runtime-base AS server

COPY --from=server-env-builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/v1/health', timeout=3).status == 200 else 1)"]

ENTRYPOINT ["openlithohub"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]

# ---------------------------------------------------------------------------
# Stage: server-cpu — MINIMAL production server image (repair-plan §9):
# core + workflow + required models + server, genuine CPU torch, no CUDA/
# NVIDIA runtime distributions, no jupyter. Build with:
#   docker build --target server-cpu -t openlithohub:server-cpu .
# ---------------------------------------------------------------------------
FROM runtime-base AS server-cpu

COPY --from=server-cpu-env-builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/v1/health', timeout=3).status == 200 else 1)"]

ENTRYPOINT ["openlithohub"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
