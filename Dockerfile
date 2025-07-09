FROM python:3.11.7-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

FROM base AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock ./
COPY mettagrid/ ./mettagrid/
COPY app_backend/ ./app_backend/
COPY common/ ./common/
COPY agent/ ./agent/

RUN uv sync --frozen --no-dev --no-editable

FROM base AS development

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock ./
COPY mettagrid/ ./mettagrid/
COPY app_backend/ ./app_backend/
COPY common/ ./common/
COPY agent/ ./agent/
COPY . .

RUN uv sync --frozen

FROM base AS production

COPY --from=builder /app/.venv /app/.venv
COPY . .

RUN useradd --create-home --shell /bin/bash app
RUN chown -R app:app /app
USER app

ENV PATH="/app/.venv/bin:$PATH"

COPY docker/ ./docker/
RUN chmod +x ./docker/docker-entrypoint.py

EXPOSE 8000

ENTRYPOINT ["./docker/docker-entrypoint.py"]