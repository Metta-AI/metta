FROM ubuntu:22.04

WORKDIR /workspace

COPY . /workspace/metta

WORKDIR /workspace/metta

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /uvx /bin/

COPY .python-version pyproject.toml uv.lock ./
RUN uv python install

RUN bash install.sh --profile softmax-docker --non-interactive

RUN uv sync --all-extras --frozen

CMD ["/bin/bash"]
