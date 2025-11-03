FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (minimal)
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"
ENV PYTHONPATH="/app"

# Copy only PR similarity code and dependencies
COPY mcp_servers/pr_similarity /app/mcp_servers/pr_similarity
COPY metta/tools/pr_similarity.py /app/metta/tools/pr_similarity.py
COPY packages/gitta /app/packages/gitta

# Create minimal pyproject.toml with only PR similarity dependencies
RUN cat > /app/pyproject.toml <<'EOF'
[project]
name = "pr-similarity-cache"
version = "1.0.0"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.24.0",
    "google-generativeai>=0.3.0",
    "boto3>=1.34.0",
]

[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
py-modules = []
EOF

# Install gitta as editable package
RUN uv pip install --system -e ./packages/gitta

# Install PR similarity dependencies (including boto3 for S3 cache management)
RUN uv pip install --system \
    "numpy>=1.24.0" \
    "google-generativeai>=0.3.0" \
    "boto3>=1.34.0"

# Default command runs PR similarity cache build with S3 download/upload
CMD ["python", "mcp_servers/pr_similarity/build_cache.py", "--download-from-s3"]
