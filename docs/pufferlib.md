To run pufferlib training for metta (including the CUDA 13.0 kernels used on
our sm_120 systems), first deactivate your uv environment then do the following:

```
git clone https://github.com/PufferAI/PufferLib
cd PufferLib
uv venv
source .venv/bin/activate
pip install setuptools scikit_build_core numpy pybind11 torch einops
CUDA_HOME=/path/to/metta/build/cuda13-wrapper \
TORCH_CUDA_ARCH_LIST=12.0 \
FORCE_CUDA=1 \
pip install -e .[metta] --no-build-isolation
puffer train metta

See the upstream fork documentation at
https://github.com/relh/PufferLib/blob/cu130-sm120-wheel/docs/cu130-wheel.md for
details on creating the CUDA 13.0 nvcc wrapper required by PyTorch 2.8. You can
also run `scripts/setup_cuda13_wrapper.sh` in this repository to generate the
wrapper locally (override `CUDA_TOOLKIT_HOME` if your CUDA install lives
elsewhere).
```
