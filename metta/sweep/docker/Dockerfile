# Multi-stage build for aggressive caching
FROM ubuntu:22.04 AS base

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies in layers for better caching
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    build-essential \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Install Python and basic tools
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Create metta user (mimics SkyPilot setup)
RUN useradd -m -s /bin/bash metta && \
    echo "metta ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER metta
WORKDIR /home/metta

# Install uv for fast Python package management
COPY --from=ghcr.io/astral-sh/uv:0.7.2 /uv /uvx /bin/

# Install OpenMPI for distributed training (mimics SkyPilot)
USER root
RUN apt-get update && apt-get install -y \
    openmpi-bin \
    openmpi-common \
    libopenmpi-dev \
    openssh-server \
    openssh-client \
    && rm -rf /var/lib/apt/lists/*

# Configure SSH for multi-node simulation
RUN mkdir /var/run/sshd && \
    echo 'root:metta123' | chpasswd && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config

USER metta

# Set up SSH keys for passwordless access
RUN ssh-keygen -t rsa -N "" -f ~/.ssh/id_rsa && \
    cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys && \
    chmod 600 ~/.ssh/authorized_keys

# Stage 2: Development environment
FROM base AS dev

# Copy dependencies and workspace files for better caching
COPY --chown=metta:metta .python-version pyproject.toml uv.lock /home/metta/metta/
COPY --chown=metta:metta common/pyproject.toml /home/metta/metta/common/
COPY --chown=metta:metta agent/pyproject.toml /home/metta/metta/agent/
COPY --chown=metta:metta app_backend/pyproject.toml /home/metta/metta/app_backend/
COPY --chown=metta:metta mettagrid/pyproject.toml /home/metta/metta/mettagrid/
COPY --chown=metta:metta experiments/pyproject.toml /home/metta/metta/experiments/
WORKDIR /home/metta/metta

# Install dependencies without workspace
ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv,id=uv-cache \
    uv sync --no-install-workspace

# Copy the full codebase
COPY --chown=metta:metta . /home/metta/metta

# Install the workspace
RUN --mount=type=cache,target=/root/.cache/uv,id=uv-cache \
    uv sync --locked

# Set up environment variables that mimic SkyPilot
ENV CUDA_VISIBLE_DEVICES=0
ENV NCCL_DEBUG=INFO
ENV PYTHONPATH=/home/metta/metta:$PYTHONPATH
ENV PATH="/home/metta/metta/.venv/bin:$PATH"

# Create directories that mimic SkyPilot structure (as root)
USER root
RUN mkdir -p /mnt/s3/train_dir && \
    mkdir -p /tmp/ray && \
    chown -R metta:metta /mnt/s3
USER metta
RUN ln -s /mnt/s3/train_dir ./train_dir

# Stage 3: Testing environment
FROM dev AS test

# Install additional testing dependencies
RUN uv add pytest pytest-xdist pytest-timeout pytest-cov

# Copy test configurations and scripts
COPY --chown=metta:metta metta/sweep/docker/configs/ /home/metta/metta/test-configs/
COPY --chown=metta:metta metta/sweep/docker/scripts/ /home/metta/metta/test-scripts/

# Set up test data directory structure
RUN mkdir -p /home/metta/test-results && \
    mkdir -p /home/metta/test-logs && \
    mkdir -p /home/metta/test-artifacts

# Configure git for test commits
RUN git config --global user.email "test@metta.ai" && \
    git config --global user.name "Metta Test Runner"

# Health check to ensure environment is ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" || exit 1

# Default command starts SSH and drops to shell
CMD ["/home/metta/metta/test-scripts/start-test-env.sh"]
