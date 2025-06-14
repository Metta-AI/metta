FROM mettaai/metta-base:3.0

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

RUN sed -i '\#cd /puffertank/pufferlib#d' /root/.bashrc \
 && echo 'cd /workspace/metta' >> /root/.bashrc

WORKDIR /workspace/metta/
ENTRYPOINT ["/bin/bash", "-c"]
