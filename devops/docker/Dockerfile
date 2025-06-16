FROM pufferai/puffertank:3.0

# Set the timezone and install packages in one layer, clean up apt cache
RUN ln -snf /usr/share/zoneinfo/America/Los_Angeles /etc/localtime \
    && echo America/Los_Angeles > /etc/timezone \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        ninja-build \
        git \
        sudo \
        wget \
        vim \
        screen \
        spirv-tools \
        xdg-utils \
        python3-opencv \
        curl \
        ffmpeg \
        libx264-dev \
        git-lfs \
        tmux \
        iproute2 \
        netcat-openbsd \
        iputils-ping \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# Configure screen
RUN echo "defscrollback 10000" >> /root/.screenrc && \
    echo "termcapinfo xterm* ti@:te@" >> /root/.screenrc

RUN touch /root/.no_auto_tmux
COPY devops/mettabox/tmux.conf /root/.tmux.conf

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.7.2 /uv /uvx /bin/

# Clone metta repository with full .git and submodules
WORKDIR /workspace/
ARG BRANCH=main
RUN git clone --recursive https://github.com/Metta-AI/metta.git metta \
 && cd metta \
 && git checkout ${BRANCH}

# Set working dir for rest of image
WORKDIR /workspace/metta

# Install dependencies
ENV UV_LINK_MODE=copy
ARG TARGETPLATFORM
RUN --mount=type=cache,target=/root/.cache/uv,id=uv-cache/${TARGETPLATFORM} \
    uv sync --no-install-workspace

RUN --mount=type=cache,target=/root/.cache/uv,id=uv-cache/${TARGETPLATFORM} \
    uv sync --locked

ENV PATH="/workspace/metta/.venv/bin:$PATH"

RUN sed -i '\#cd /puffertank/pufferlib#d' /root/.bashrc \
 && echo 'cd /workspace/metta' >> /root/.bashrc

ENTRYPOINT ["/bin/bash", "-c"]

