FROM ubuntu:22.04

WORKDIR /workspace

COPY . /workspace/metta

WORKDIR /workspace/metta

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /uvx /bin/
RUN uv --version

RUN bash install.sh --profile softmax-docker --non-interactive
RUN uv sync --frozen --package metta-app-backend

WORKDIR /workspace/metta/app_backend
EXPOSE 8000
ENV HOST=0.0.0.0
ENV PORT=8000

CMD ["uv", "run", "python", "src/metta/app_backend/server.py"]
