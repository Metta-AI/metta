# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Bazel for mettagrid build
RUN curl -L https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64 -o /usr/local/bin/bazel \
    && chmod +x /usr/local/bin/bazel

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.7.2 /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy entire project structure
COPY . .

# Install dependencies using uv
RUN cd packages/softmax-orchestrator && uv sync --frozen

# Set working directory to orchestrator package
WORKDIR /app/packages/softmax-orchestrator

# Expose port
EXPOSE 8000

# Set environment variables
ENV HOST=0.0.0.0
ENV PORT=8000

# Run the application
CMD ["uv", "run", "python", "src/softmax/orchestrator/server.py"]
