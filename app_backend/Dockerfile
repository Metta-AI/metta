# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

RUN mkdir -p /app/app_backend

# Set working directory
WORKDIR /app

# Copy uv configuration files
COPY pyproject.toml uv.lock* ./

# Install dependencies using uv
RUN uv sync --frozen

WORKDIR /app/app_backend

# Copy the entire app_backend directory
COPY . .

# Create a non-root user
RUN useradd --create-home --shell /bin/bash app && chown -R app:app /app
USER app

# Expose port
EXPOSE 8000

# Set environment variables
ENV HOST=0.0.0.0
ENV PORT=8000

WORKDIR /app
ENV PYTHONPATH=/app

# Run the application
CMD ["uv", "run", "python", "app_backend/src/server.py"]
