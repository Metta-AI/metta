FROM mettaai/metta-base:latest

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Install dependencies with uv
RUN uv pip install boto3 chex hydra-core matplotlib pettingzoo rich scipy tabulate

# Use ARG to create a build-time variable
ARG CACHE_DATE

# Use the ARG in a RUN command to break cache
RUN echo "Cache breaker: ${CACHE_DATE}"

# Checkout metta and submodules
WORKDIR /workspace/
RUN git clone --recursive https://github.com/Metta-AI/metta.git metta
WORKDIR /workspace/metta

# Install requirements with uv
RUN uv pip install -r requirements.txt
RUN make install

WORKDIR /workspace/metta/
ENTRYPOINT ["/bin/bash", "-c"]
