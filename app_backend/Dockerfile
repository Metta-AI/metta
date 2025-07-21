# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Install system dependencies and add Kubernetes apt repository
RUN apt-get update && apt-get install -y \
    curl \
    git \
    apt-transport-https \
    ca-certificates \
    gnupg \
    && curl -fsSL https://pkgs.k8s.io/core:/stable:/v1.31/deb/Release.key | gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg \
    && echo 'deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v1.31/deb/ /' > /etc/apt/sources.list.d/kubernetes.list \
    && apt-get update \
    && apt-get install -y kubectl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.7.2 /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy entire project structure
COPY . .

# Install dependencies using uv
RUN cd app_backend && uv sync --frozen

# Set working directory to app_backend
WORKDIR /app/app_backend

# Expose port
EXPOSE 8000

# Set environment variables
ENV HOST=0.0.0.0
ENV PORT=8000

# Run the application
CMD ["uv", "run", "python", "src/metta/app_backend/server.py"]
