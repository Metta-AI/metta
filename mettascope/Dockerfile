# Start with Playwright base image that includes browsers and Node.js
FROM mcr.microsoft.com/playwright:v1.54.1-jammy

# Set non-interactive to avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

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
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Make Python 3.11 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Set working directory
WORKDIR /app
