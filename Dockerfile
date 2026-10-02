# syntax=docker/dockerfile:1
# =============================================================================
# AI Content Summarizer - inference image (FastAPI + PEGASUS, CPU)
#
#   docker build -t ai-content-summarizer .
#   docker run -p 8000:8000 ai-content-summarizer
#   -> http://localhost:8000/ui/   (first start downloads the model, ~2.3 GB)
#
# Multi-stage build:
#   builder  - installs CPU-only PyTorch + runtime deps into a virtualenv
#   runtime  - slim image with just the venv and the app code, non-root user
#
# The model is NOT baked into the image (it would add ~2.3 GB to every push
# and pull). It is downloaded from the Hugging Face Hub on first start into
# HF_HOME; mount a volume there (docker-compose does) so it survives restarts.
# =============================================================================

ARG PYTHON_VERSION=3.12

# ----------------------------------------------------------------- builder
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# CPU-only PyTorch: ~200 MB instead of the ~2 GB CUDA build.
RUN pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu

# Dependencies first (cached layer), then the package itself.
WORKDIR /build
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps .

# ----------------------------------------------------------------- runtime
FROM python:${PYTHON_VERSION}-slim AS runtime

LABEL org.opencontainers.image.title="ai-content-summarizer" \
      org.opencontainers.image.description="Abstractive dialogue summarization API (PEGASUS + FastAPI)" \
      org.opencontainers.image.licenses="MIT"

# PROJECT_ROOT: the package is installed into site-packages, so tell it where
#               config/ and frontend/ live.
# LOG_FORMAT=json: one JSON object per line, easy to query in CloudWatch.
# LOG_TO_FILE=false: containers log to stdout; the platform collects it.
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PROJECT_ROOT=/app \
    HF_HOME=/app/.cache/huggingface \
    LOG_FORMAT=json \
    LOG_TO_FILE=false \
    API_PORT=8000

RUN groupadd --system --gid 1001 app \
    && useradd --system --uid 1001 --gid app --home-dir /app --shell /usr/sbin/nologin app

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --chown=1001:1001 config ./config
COPY --chown=1001:1001 api ./api
COPY --chown=1001:1001 frontend ./frontend
RUN mkdir -p /app/.cache/huggingface /app/artifacts/models \
    && chown -R 1001:1001 /app/.cache /app/artifacts

USER 1001:1001
EXPOSE 8000

# /health returns 200 only once the model is loaded. start-period allows for
# the first-start model download.
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"API_PORT\", \"8000\")}/health', timeout=4)"]

# One worker: each worker loads its own copy of the model (~2.5 GB RAM).
# Scale out with more containers/instances instead of more workers.
CMD ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port ${API_PORT} --workers 1 --proxy-headers --forwarded-allow-ips='*'"]
