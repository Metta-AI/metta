FROM ubuntu:22.04

WORKDIR /workspace

COPY . /workspace/metta

WORKDIR /workspace/metta

# Install bootstrap dependencies (bazel, nimby, nim, git, g++) before uv sync
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
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
    python3 -m pip install packaging && \
    cd /tmp && \
    python3 bootstrap.py --non-interactive

# Pre-build app_backend package to avoid issues during container startup
RUN uv sync --frozen --package metta-app-backend

# Set working directory to app_backend
WORKDIR /workspace/metta/app_backend

# Expose port
EXPOSE 8000

# Set environment variables
ENV HOST=0.0.0.0
ENV PORT=8000

# Run the application
CMD ["uv", "run", "python", "src/metta/app_backend/server.py"]
