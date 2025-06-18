#!/bin/bash
set -e

echo "Building app_backend Docker image..."

# Generate uv.lock file
echo "Generating uv.lock file..."
uv lock

# Build Docker image
echo "Building Docker image..."
docker build -t metta-app-backend:latest .

echo "Build complete! You can run the container with:"
echo "docker run -p 8000:8000 -e STATS_DB_URI=your_db_uri metta-app-backend:latest"
