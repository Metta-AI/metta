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

<<<<<<< HEAD
RUN pip install -r requirements.txt
RUN make install
=======
RUN git checkout ${BRANCH}
RUN ./devops/setup_build.sh
ENV PATH="/workspace/metta/.venv/bin:$PATH"
>>>>>>> 13c12a2fdf120e435aa056c95de09aa7ccaa5a87

WORKDIR /workspace/metta/
ENTRYPOINT ["/bin/bash", "-c"]
