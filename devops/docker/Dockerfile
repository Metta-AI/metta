FROM mettaai/metta-base:latest

ARG CACHE_DATE
RUN echo "Cache breaker: ${CACHE_DATE}"

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.7.2 /uv /uvx /bin/

# Use ARG to create a build-time variable
ARG BRANCH=main

# Use the ARG in a RUN command to break cache

# Checkout metta and submodules
WORKDIR /workspace/
RUN git clone --recursive https://github.com/Metta-AI/metta.git metta
WORKDIR /workspace/metta

RUN git checkout ${BRANCH}

RUN uv sync

ENV PATH="/workspace/metta/.venv/bin:$PATH"

# Set up terminal configuration in .bashrc
RUN echo '# Terminal configuration for proper log display' >> ~/.bashrc && \
    echo 'export COLUMNS=200' >> ~/.bashrc && \
    echo 'export TERM=xterm-256color' >> ~/.bashrc && \
    echo 'export PYTHONUNBUFFERED=1' >> ~/.bashrc && \
    echo '# Try to set terminal size if possible' >> ~/.bashrc && \
    echo 'stty cols $COLUMNS rows $LINES 2>/dev/null || true' >> ~/.bashrc

WORKDIR /workspace/metta/
ENTRYPOINT ["/bin/bash", "-c"]
