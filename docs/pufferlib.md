To run pufferlib training for metta, first deactivate your uv environment then do the following:

```
```
git clone https://github.com/PufferAI/PufferLib
cd PufferLib
uv venv
source .venv/bin/activate
pip install setuptools scikit_build_core numpy pybind11 torch einops
pip install -e .[metta] --no-build-isolation
puffer train metta
```
```
