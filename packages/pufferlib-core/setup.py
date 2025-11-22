import os
import platform
import shutil
import sys
from typing import Optional

from setuptools import setup

# Always build extensions
BUILD_EXTENSIONS = True

# Import torch for extensions
try:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME

    print("Building pufferlib-core with C++/CUDA extensions")
except ImportError:
    print("Error: torch not available. Please install torch first.")
    sys.exit(1)

# Build with DEBUG=1 to enable debug symbols
DEBUG = os.getenv("DEBUG", "0") == "1"


def _detect_cuda_home() -> Optional[str]:
    """Use the CUDA toolkit path baked into our base image."""
    if CUDA_HOME and os.path.exists(CUDA_HOME):
        return CUDA_HOME
    return None


def _nvcc_available() -> bool:
    """Detect nvcc, preferring the toolkit path shipped in our sandbox image."""
    nvcc_path = shutil.which("nvcc")
    if nvcc_path:
        return True

    cuda_home = _detect_cuda_home()
    if not cuda_home:
        return False

    candidate = os.path.join(cuda_home, "bin", "nvcc")
    if os.path.exists(candidate):
        os.environ.setdefault("CUDA_HOME", cuda_home)
        os.environ["PATH"] = os.pathsep.join([os.path.join(cuda_home, "bin"), os.environ.get("PATH", "")])
        return True

    return False


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
if _nvcc_available():
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
