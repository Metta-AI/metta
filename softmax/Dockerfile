FROM ubuntu:24.04

WORKDIR /workspace

COPY . /workspace/metta

WORKDIR /workspace/metta

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /uvx /bin/
RUN uv --version

RUN bash install.sh --profile softmax-docker --non-interactive

WORKDIR /workspace/metta/softmax
RUN uv sync --frozen --no-dev
RUN uv run python -c "import softmax.dashboard.report"

CMD ["uv", "run", "python", "-m", "softmax.dashboard.report", "--push"]
