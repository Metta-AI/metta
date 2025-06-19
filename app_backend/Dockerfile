# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.7.2 /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy uv configuration files from top-level directory
COPY uv.lock ./
COPY app_backend/pyproject.toml ./

COPY app_backend ./app_backend

# Install dependencies using uv
RUN uv sync --frozen

# Expose port
EXPOSE 8000

# Set environment variables
ENV HOST=0.0.0.0
ENV PORT=8000

# Run the application
CMD ["uv", "run", "python", "app_backend/server.py"]
