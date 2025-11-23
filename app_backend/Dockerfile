FROM ubuntu:22.04

WORKDIR /workspace

COPY . /workspace/metta

WORKDIR /workspace/metta

RUN bash ./devops/tools/install-system.sh

# Force pufferlib-core to build CPU-only in this image (no CUDA here)
ENV PUFFERLIB_CPU_ONLY=1

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
