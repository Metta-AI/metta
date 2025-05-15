FROM mettaai/metta-base:latest

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Use ARG to create a build-time variable
ARG CACHE_DATE

# Use the ARG in a RUN command to break cache
RUN echo "Cache breaker: ${CACHE_DATE}"

# Checkout metta and submodules
WORKDIR /workspace/
RUN git clone --recursive https://github.com/Metta-AI/metta.git metta
WORKDIR /workspace/metta

RUN ./devops/setup_build.sh

WORKDIR /workspace/metta/
ENTRYPOINT ["/bin/bash", "-c"]
