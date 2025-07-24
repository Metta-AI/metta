# Start with Playwright base image that includes browsers and Node.js
FROM mcr.microsoft.com/playwright:v1.53.0-jammy

# Install Python 3.11 and build dependencies
RUN apt-get update && apt-get install -y \
    # Python 3.11
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    # Build tools for native extensions
    build-essential \
    cmake \
    # Additional useful tools
    curl \
    git \
    jq \
    # Cleanup
    && rm -rf /var/lib/apt/lists/*

# Make Python 3.11 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.7.2 /uv /uvx /bin/

# Verify installations
RUN echo "Node version:" && node --version \
    && echo "NPM version:" && npm --version \
    && echo "Python version:" && python --version \
    && echo "Playwright version:" && npx playwright --version \
    && echo "uv version:" && uv --version

# Set working directory
WORKDIR /app

# This image will be used as a base for CI/CD
