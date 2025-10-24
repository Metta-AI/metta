FROM ubuntu:24.04

WORKDIR /workspace

# Copy the entire metta repo
COPY . /workspace/metta

WORKDIR /workspace/metta

# Install system dependencies
RUN bash ./devops/tools/install-system.sh

# Install all dependencies (includes asana, skypilot, github libraries)
RUN uv sync --frozen --no-dev

# Verify all collector imports work
RUN uv run python -c "from devops.datadog.collectors import GitHubCollector, SkypilotCollector, AsanaCollector, EC2Collector, WandbCollector, KubernetesCollector; print('✓ All 6 collectors imported successfully')"

# Default command runs all collectors sequentially
# Helm chart can override this if needed
# Use absolute path because uv run resolves paths relative to package directories
CMD ["uv", "run", "python", "/workspace/metta/devops/datadog/run_all_collectors.py"]
