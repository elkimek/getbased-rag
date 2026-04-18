FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LENS_HOST=0.0.0.0 \
    LENS_PORT=8322 \
    LENS_DATA_DIR=/data

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      curl \
      ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src ./src

RUN pip install --upgrade pip && pip install ".[full]"

RUN useradd --create-home --uid 1000 lens \
    && mkdir -p /data \
    && chown -R lens:lens /data /app
USER lens

VOLUME ["/data"]
EXPOSE 8322

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8322/health || exit 1

ENTRYPOINT ["lens"]
CMD ["serve"]
