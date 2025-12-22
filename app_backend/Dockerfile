FROM ubuntu:22.04

WORKDIR /workspace

COPY . /workspace/metta

WORKDIR /workspace/metta

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    rm -rf /var/lib/apt/lists/* && \
    apt-get update && apt-get install -y --no-install-recommends \
    curl git build-essential ca-certificates

COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /uvx /bin/

COPY .python-version pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv python install

RUN --mount=type=cache,target=/root/.cache/uv \
    bash install.sh --profile softmax-docker --non-interactive

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --package metta-app-backend

WORKDIR /workspace/metta/app_backend
EXPOSE 8000
ENV HOST=0.0.0.0
ENV PORT=8000

CMD ["uv", "run", "python", "src/metta/app_backend/server.py"]
