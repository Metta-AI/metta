import os
import platform
import shutil
import sys

from setuptools import setup

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

force_cuda = os.getenv("PUFFERLIB_BUILD_CUDA", "0") == "1"
disable_cuda = os.getenv("PUFFERLIB_DISABLE_CUDA", "0") == "1"
has_nvcc = shutil.which("nvcc") is not None

try:
    cuda_runtime_available = bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
except Exception:
    cuda_runtime_available = False

build_with_cuda = has_nvcc and (force_cuda or (cuda_runtime_available and not disable_cuda))

if build_with_cuda:
    extension_class = CUDAExtension
    torch_sources.append("src/pufferlib/extensions/cuda/pufferlib.cu")
    if force_cuda and not cuda_runtime_available:
        print("Building with CUDA support (PUFFERLIB_BUILD_CUDA=1; runtime CUDA unavailable)")
    else:
        print("Building with CUDA support")
else:
    extension_class = CppExtension
    if disable_cuda and has_nvcc:
        print("Building with CPU-only support (PUFFERLIB_DISABLE_CUDA=1)")
    else:
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
