FROM mettaai/metta-base:latest

ARG CACHE_DATE
RUN echo "Cache breaker: ${CACHE_DATE}"

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.7.2 /uv /uvx /bin/

# Use ARG to create a build-time variable
ARG BRANCH=main

# Checkout metta and submodules
WORKDIR /workspace/
RUN git clone --recursive https://github.com/Metta-AI/metta.git metta
WORKDIR /workspace/metta

RUN git checkout ${BRANCH}

RUN uv sync

ENV PATH="/workspace/metta/.venv/bin:$PATH"

# The terminal configuration is already in the base image
# Just ensure we're using a login shell to source bashrc
WORKDIR /workspace/metta/
ENTRYPOINT ["/bin/bash", "-l", "-c"]
