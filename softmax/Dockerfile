FROM python:3.11-slim

WORKDIR /workspace/metta

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.8.13 /uv /uvx /bin/

# Copy workspace metadata
COPY .python-version pyproject.toml uv.lock ./

# Install Python
RUN uv python install

# Copy only the packages we need
COPY common/pyproject.toml common/
COPY softmax/pyproject.toml softmax/
COPY common/src common/src
COPY softmax/src softmax/src

# Install only common and softmax packages (uv will skip others not in the directory)
RUN uv pip install --system -e ./common -e ./softmax

# Verify imports work
RUN python -c "import softmax.dashboard.report"

# Run the dashboard
CMD ["python", "-m", "softmax.dashboard.report", "report", "--push"]
