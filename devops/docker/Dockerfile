FROM pufferai/puffertank:3.0
# FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel

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

WORKDIR /workspace/metta

# Install dependencies
ADD .python-version pyproject.toml uv.lock ./
COPY mettagrid/pyproject.toml mettagrid/
COPY app_backend/pyproject.toml app_backend/
COPY common/pyproject.toml common/
COPY agent/pyproject.toml agent/
COPY experiments/pyproject.toml experiments/

ENV UV_LINK_MODE=copy
ENV UV_FREEZE=1
ARG TARGETPLATFORM
RUN --mount=type=cache,target=/root/.cache/uv,id=uv-cache/${TARGETPLATFORM} \
    uv sync --no-install-workspace

# Install sources
ADD . .

# When building in CI, actions/checkout stores an auth token in .git/config that we need to remove
RUN git config --remove-section http."https://github.com/" || true

# Make sure the remote is configured for anonymous access
RUN git remote remove origin && git remote add origin https://github.com/Metta-AI/metta.git

# Doesn't do anything but guarantees that the remote is configured for anonymous access
RUN git fetch

RUN --mount=type=cache,target=/root/.cache/uv,id=uv-cache/${TARGETPLATFORM} \
    uv sync --locked

ENV PATH="/workspace/metta/.venv/bin:$PATH"

RUN sed -i '\#cd /puffertank/pufferlib#d' /root/.bashrc \
 && echo 'cd /workspace/metta' >> /root/.bashrc

ENTRYPOINT ["/bin/bash", "-c"]
