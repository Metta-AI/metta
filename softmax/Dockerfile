FROM ubuntu:24.04

WORKDIR /workspace

# Copy the entire metta repo
COPY . /workspace/metta

WORKDIR /workspace/metta

# Install system dependencies
RUN bash ./devops/tools/install-system.sh

# Install metta with softmax profile
RUN bash install.sh --profile softmax-docker --non-interactive

# Pre-build packages to avoid import delays
# Verify the dashboard can be imported
RUN uv sync --frozen && \
    .venv/bin/python -c "import softmax.dashboard.report"

# Set PATH to use the pre-built venv
ENV PATH="/workspace/metta/.venv/bin:$PATH"
ENV VIRTUAL_ENV="/workspace/metta/.venv"

# Run directly from venv (no uv run to avoid re-syncing at startup)
CMD ["python", "-m", "softmax.dashboard.report", "report", "--push"]
