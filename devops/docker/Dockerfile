FROM daveey/metta-base:latest

RUN pip install boto3 chex hydra-core matplotlib pettingzoo
RUN pip install rich scipy tabulate torchrl

RUN echo "Cache breaker: $(date)" > cache-breaker.txt

# Checkout metta and submodules
WORKDIR /workspace/
RUN date
RUN git clone --recursive https://github.com/Metta-AI/metta.git metta
WORKDIR /workspace/metta

RUN pip install -r requirements.txt
RUN bash ./devops/setup_build.sh

# Download the latest baseline model
#RUN python -m sample_factory.huggingface.load_from_hub -r metta-ai/baseline -d /workspace/metta-baseline
#RUN ln -s /workspace/metta-baseline/baseline/checkpoint_p0/$(ls /workspace/metta-baseline/baseline/checkpoint_p0 | tail -n1) /workspace/metta-baseline/latest

