FROM python:3.12-slim

WORKDIR /app

ARG VERSION=0.0.0

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY src/ src/

RUN SETUPTOOLS_SCM_PRETEND_VERSION=${VERSION} pip install --no-cache-dir -e ".[data,models]"

ENTRYPOINT ["openlithohub"]
CMD ["--help"]
