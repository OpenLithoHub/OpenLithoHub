# syntax=docker/dockerfile:1.6
# ---------------------------------------------------------------------------
# Stage 1: builder — install deps + the openlithohub wheel into a venv.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /build

ARG VERSION=0.0.0

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY src/ src/

# Build into a relocatable venv so the runtime stage can copy /opt/venv wholesale.
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip \
 && SETUPTOOLS_SCM_PRETEND_VERSION=${VERSION} \
    /opt/venv/bin/pip install --no-cache-dir ".[data,models,workflow,jupyter]"

# ---------------------------------------------------------------------------
# Stage 2: runtime — slim image with KLayout's Qt deps + the venv.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

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

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

ENTRYPOINT ["openlithohub"]
CMD ["--help"]
