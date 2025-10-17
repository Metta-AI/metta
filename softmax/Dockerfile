FROM ubuntu:24.04

WORKDIR /workspace

# Copy the entire metta repo (dashboard depends on metta-common)
COPY . /workspace/metta

WORKDIR /workspace/metta

# Install system dependencies
RUN bash ./devops/tools/install-system.sh

# Install metta with softmax profile
RUN bash install.sh --profile softmax-docker --non-interactive

# Pre-build packages to avoid import delays
RUN uv sync --frozen && \
    uv run python -c "import softmax.dashboard.report"

# Run the dashboard report command
CMD ["uv", "run", "python", "-m", "softmax.dashboard.report", "report", "--push"]
