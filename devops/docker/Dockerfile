FROM mettaai/metta-base:latest

RUN pip install boto3 chex hydra-core matplotlib pettingzoo
RUN pip install rich scipy tabulate

# Use ARG to create a build-time variable
ARG CACHE_DATE

# Use the ARG in a RUN command to break cache
RUN echo "Cache breaker: ${CACHE_DATE}"

# Checkout metta and submodules
WORKDIR /workspace/
RUN git clone --recursive https://github.com/Metta-AI/metta.git metta
WORKDIR /workspace/metta

RUN pip install -r requirements.txt
RUN python setup.py build_ext --inplace

WORKDIR /workspace/metta/deps/
RUN git clone --depth=1 --branch=metta https://github.com/Metta-AI/pufferlib.git
RUN git clone https://github.com/kywch/carbs.git
RUN git clone https://github.com/Metta-AI/wandb_carbs.git

WORKDIR /workspace/metta/deps/pufferlib
RUN pip install -e .

WORKDIR /workspace/metta/mettagrid
RUN python setup.py build_ext --inplace
RUN pip install -e .

WORKDIR /workspace/metta/deps/carbs
RUN python setup.py build_ext --inplace
RUN pip install -e .

WORKDIR /workspace/metta/deps/wandb_carbs
RUN pip install -e .

WORKDIR /workspace/metta/
ENTRYPOINT ["/bin/bash", "-c"]
