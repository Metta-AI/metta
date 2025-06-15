#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Determine the repository root (assuming script is in devops/docker/ or root)
if [[ "$SCRIPT_DIR" == *"/devops/docker"* ]]; then
  # Script is in devops/docker/, go up two levels
  REPO_ROOT="$SCRIPT_DIR/../.."
elif [[ -d "$SCRIPT_DIR/devops" ]]; then
  # Script is in root
  REPO_ROOT="$SCRIPT_DIR"
else
  echo "Error: Cannot determine repository root. Please run from the repository root."
  exit 1
fi

# Change to repository root
cd "$REPO_ROOT" || exit 1
echo "Working from repository root: $(pwd)"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
  echo "Error: Docker is not installed!"
  echo ""
  echo "To install Docker on macOS:"
  echo "  1. Using Homebrew (recommended):"
  echo "     brew install --cask docker"
  echo ""
  echo "  2. Or download Docker Desktop from:"
  echo "     https://www.docker.com/products/docker-desktop/"
  echo ""
  echo "After installation:"
  echo "  1. Open Docker Desktop from your Applications folder"
  echo "  2. Wait for Docker to start (you'll see a whale icon in the menu bar)"
  echo "  3. Run this script again"
  echo ""
  exit 1
fi

# Check if Docker daemon is running
if ! docker info &> /dev/null; then
  echo "Error: Docker is installed but not running!"
  echo ""
  echo "Please start Docker Desktop:"
  echo "  1. Open Docker Desktop from your Applications folder"
  echo "  2. Wait for the whale icon to appear in your menu bar"
  echo "  3. Make sure the whale icon shows 'Docker Desktop is running'"
  echo "  4. Run this script again"
  echo ""
  exit 1
fi

echo "Docker is installed and running ✓"
echo ""
echo "Building Docker images..."

# Build base image
echo "Building metta-base image for linux/amd64 (AWS compatible)..."
docker build --platform linux/amd64 \
  -f devops/docker/Dockerfile.base \
  -t mettaai/metta-base:latest .

if [ $? -ne 0 ]; then
  echo "Error: Failed to build metta-base image"
  exit 1
fi

echo "✓ Successfully built metta-base image"
echo ""

# Build main image
echo "Building metta image for linux/amd64 (AWS compatible)..."
docker build --platform linux/amd64 \
  -f devops/docker/Dockerfile \
  -t mettaai/metta:latest .

if [ $? -ne 0 ]; then
  echo "Error: Failed to build metta image"
  exit 1
fi

echo "✓ Successfully built metta image"
echo ""
echo "Build complete! Don't forget to push the image:"
echo "  ${SCRIPT_DIR}/push_image.sh"
echo ""
