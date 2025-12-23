FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST=7.5;8.0;8.6;8.9;9.0
ENV READTHEDOCS=True
ENV UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    ca-certificates \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && curl -L https://github.com/bazelbuild/bazelisk/releases/download/v1.27.0/bazelisk-linux-amd64 -o /usr/local/bin/bazel \
    && chmod +x /usr/local/bin/bazel

WORKDIR /workspace/metta

COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /uvx /bin/

COPY .python-version pyproject.toml uv.lock ./

RUN --mount=type=cache,id=metta-uv-cache,target=/root/.cache/uv \
  uv python install

# Install bootstrap dependencies (bazel, nimby, nim, git, g++) BEFORE uv sync
# This ensures uv sync can build packages like mettagrid that require these tools
# Copy bootstrap script to /tmp to avoid importing from metta package structure
COPY metta/setup/components/system_packages/bootstrap.py /tmp/bootstrap.py
RUN --mount=type=cache,id=metta-uv-cache,target=/root/.cache/uv \
  cd /tmp && uv run --no-sync python bootstrap.py --non-interactive

COPY packages/mettagrid/pyproject.toml packages/mettagrid/
COPY app_backend/pyproject.toml app_backend/
COPY common/pyproject.toml common/
COPY agent/pyproject.toml agent/

RUN --mount=type=cache,target=/root/.cache/uv \
  uv sync --no-install-workspace --frozen

COPY . .

ENV PATH="/workspace/metta/.venv/bin:$PATH"
ENV VIRTUAL_ENV="/workspace/metta/.venv"

RUN git config --global checkout.defaultRemote origin \
    && git config --global branch.autoSetupMerge simple

RUN echo 'cd /workspace/metta' >> /root/.bashrc && \
    echo 'source /workspace/metta/.venv/bin/activate' >> /root/.bashrc

RUN --mount=type=cache,id=metta-uv-cache,target=/root/.cache/uv \
  --mount=type=cache,id=metta-bazel-cache,target=/workspace/metta/.bazel_output \
  uv sync --locked

ENTRYPOINT ["/bin/bash", "-c"]
