#!/bin/bash
# test_pufferlib_docker.sh
# Docker-based test for PufferLib integration
# This provides a clean, reproducible environment for testing

set -euo pipefail

# Parse command line arguments
PUFFERLIB_VERSION=${1:-"stable"}
PYTHON_VERSION=${2:-"3.11.7"}

echo "=== Docker-based PufferLib Integration Test ==="
echo "PufferLib Version: $PUFFERLIB_VERSION"
echo "Python Version: $PYTHON_VERSION"

# Create temporary Dockerfile
cat > Dockerfile.pufferlib-test << EOF
FROM python:${PYTHON_VERSION}-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    git \\
    curl \\
    build-essential \\
    cmake \\
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:\$PATH"

# Set working directory
WORKDIR /workspace

# Copy project files
COPY . .

# Make test script executable
RUN chmod +x tests/integration/test_pufferlib_fresh_install.sh

# Run the test
ENTRYPOINT ["tests/integration/test_pufferlib_fresh_install.sh"]
CMD ["${PUFFERLIB_VERSION}", "${PYTHON_VERSION}"]
EOF

# Build the Docker image
echo "Building Docker image..."
docker build -f Dockerfile.pufferlib-test -t metta-pufferlib-test:${PUFFERLIB_VERSION} .

# Run the test
echo "Running test in Docker container..."
docker run --rm \
    --name metta-pufferlib-test-${PUFFERLIB_VERSION} \
    metta-pufferlib-test:${PUFFERLIB_VERSION}

# Clean up
rm -f Dockerfile.pufferlib-test

echo "✅ Docker test completed successfully!"