FROM ubuntu:24.04

WORKDIR /workspace

# Copy the entire metta repo
COPY . /workspace/metta

WORKDIR /workspace/metta

# Install system dependencies
RUN bash ./devops/tools/install-system.sh

# Install all dependencies (includes asana, skypilot, github libraries)
RUN uv sync --frozen --no-dev

# Verify collector imports work
RUN uv run python -c "from devops.datadog.collectors import GitHubCollector, SkypilotCollector, AsanaCollector; print('✓ All collectors imported successfully')"

# Default command runs GitHub collector
# Helm chart will override this command to run specific collectors
CMD ["uv", "run", "python", "devops/datadog/run_collector.py", "github", "--push"]
