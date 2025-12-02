FROM ubuntu:24.04

WORKDIR /workspace

# Copy the entire metta repo
COPY . /workspace/metta

WORKDIR /workspace/metta

# Install bootstrap dependencies (bazel, nimby, nim, git, g++) before uv sync
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv if not present
RUN curl -LsSf https://astral.sh/uv/install.sh | sh || true
ENV PATH="/root/.local/bin:$PATH"

# Install packaging (bootstrap.py dependency) and run bootstrap
# Copy bootstrap.py to /tmp to avoid importing from metta package structure
RUN cp metta/setup/components/system_packages/bootstrap.py /tmp/bootstrap.py && \
    python3 -m pip install --break-system-packages packaging && \
    cd /tmp && \
    python3 bootstrap.py --non-interactive

# Install dependencies for softmax package only
WORKDIR /workspace/metta/softmax
RUN uv sync --frozen --no-dev

# Verify imports work
RUN uv run python -c "import softmax.dashboard.report"

# Run the dashboard
CMD ["uv", "run", "python", "-m", "softmax.dashboard.report", "--push"]
