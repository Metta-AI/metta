To run pufferlib training for metta (including CUDA 13.0 kernels on sm_120
systems), first deactivate your uv environment then do the following:

```
git clone https://github.com/PufferAI/PufferLib
cd PufferLib
uv venv
source .venv/bin/activate
pip install setuptools scikit_build_core numpy pybind11 torch einops
python -m pip install --pre torch torchvision \
    --index-url https://download.pytorch.org/whl/nightly/cu130
FORCE_CUDA=1 CUDA_HOME=/usr/local/cuda-13.0 \
python -m pip install -e .[metta] --no-build-isolation
puffer train metta
```

Within the metta repository install the cu130 torch wheels first and then run:

```
uv pip install --pre torch torchvision \
    --index-url https://download.pytorch.org/whl/nightly/cu130
export CUDA_HOME=/usr/local/cuda-13.0
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH
FORCE_CUDA=1 uv sync
```

Ensure the CUDA 13.0 toolkit and cuDNN are installed under
`/usr/local/cuda-13.0` so the shared libraries are available to PyTorch.
