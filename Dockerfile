# syntax=docker/dockerfile:1
# Multi-stage production image for the FastAPI proposal-generation service.

# --------------------------------------------------------------------------- #
# Builder stage: compile dependency wheels in an isolated layer.
# --------------------------------------------------------------------------- #
FROM python:3.11-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# --------------------------------------------------------------------------- #
# Runtime stage: minimal, non-root-friendly image that serves the API.
# --------------------------------------------------------------------------- #
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    EXECUTION_MODE=cpu_demo

WORKDIR /app

COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*

COPY . .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen('http://localhost:' + os.environ.get('PORT', '8000') + '/health')" || exit 1

# Bind to $PORT when the platform supplies one (Render / Railway / Cloud Run),
# defaulting to 8000 for local `docker run`.
CMD uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8000}