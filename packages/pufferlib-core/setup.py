import os
import platform
import shutil
import sys
from pathlib import Path

from setuptools import setup

# Always build extensions
BUILD_EXTENSIONS = True


def _sync_cuda_home_env() -> None:
    """Point CUDA_HOME at the directory containing nvcc if it's unset or stale."""
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home:
        nvcc_candidate = Path(cuda_home) / "bin" / "nvcc"
        if nvcc_candidate.exists():
            return

    nvcc_path = shutil.which("nvcc")
    if nvcc_path:
        os.environ["CUDA_HOME"] = str(Path(nvcc_path).parent.parent)


_sync_cuda_home_env()

# Import torch for extensions
try:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

    print("Building pufferlib-core with C++/CUDA extensions")
except ImportError:
    print("Error: torch not available. Please install torch first.")
    sys.exit(1)

# Build with DEBUG=1 to enable debug symbols
DEBUG = os.getenv("DEBUG", "0") == "1"

# Compile args
cxx_args = ["-fdiagnostics-color=always"]
nvcc_args = []

if DEBUG:
    cxx_args += ["-O0", "-g"]
    nvcc_args += ["-O0", "-g"]
else:
    cxx_args += ["-O3"]
    nvcc_args += ["-O3"]

# Extensions setup
torch_sources = ["src/pufferlib/extensions/pufferlib.cpp"]

# Get torch library path for rpath
torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "lib")

# Check if CUDA compiler is available
if shutil.which("nvcc"):
    extension_class = CUDAExtension
    torch_sources.append("src/pufferlib/extensions/cuda/pufferlib.cu")
    print("Building with CUDA support")
else:
    extension_class = CppExtension
    print("Building with CPU-only support")

# Add rpath for torch libraries
extra_link_args = []
if platform.system() == "Darwin":  # macOS
    extra_link_args.extend([f"-Wl,-rpath,{torch_lib_path}", "-Wl,-headerpad_max_install_names"])
elif platform.system() == "Linux":  # Linux
    extra_link_args.extend([f"-Wl,-rpath,{torch_lib_path}", "-Wl,-rpath,$ORIGIN"])

ext_modules = [
    extension_class(
        "pufferlib._C",
        torch_sources,
        extra_compile_args={
            "cxx": cxx_args,
            "nvcc": nvcc_args,
        },
        extra_link_args=extra_link_args,
    ),
]
cmdclass = {"build_ext": BuildExtension}

setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
