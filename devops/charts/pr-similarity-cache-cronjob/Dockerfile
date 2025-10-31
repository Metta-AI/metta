FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"
ENV PYTHONPATH="/app"

# Copy project files
COPY .python-version pyproject.toml uv.lock ./
COPY softmax ./softmax
COPY packages/gitta ./packages/gitta

# Install dependencies
RUN uv sync --frozen

# Default command (can be overridden in helm chart)
CMD ["uv", "run", "python", "-m", "softmax.pr_similarity.cache_refresh"]
