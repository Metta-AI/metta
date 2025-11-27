FROM ubuntu:24.04

WORKDIR /workspace

# Copy the entire metta repo
COPY . /workspace/metta

WORKDIR /workspace/metta

# Install system dependencies
RUN bash ./devops/tools/install-system.sh

# Install dependencies for softmax package only
WORKDIR /workspace/metta/softmax

# Force pufferlib-core to build CPU-only in this image (no CUDA here)
ENV PUFFERLIB_CPU_ONLY=1

RUN uv sync --frozen --no-dev

# Verify imports work
RUN uv run python -c "import softmax.dashboard.report" 

# Run the dashboard
CMD ["uv", "run", "python", "-m", "softmax.dashboard.report", "--push"]
