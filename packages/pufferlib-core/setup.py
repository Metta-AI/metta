import os
import platform
import shutil
import sys
from pathlib import Path

import re
import subprocess

from setuptools import setup

# Always build extensions
BUILD_EXTENSIONS = True

# Detect nvcc before importing torch so we can force a CPU-only build when CUDA tools are absent.
NVCC_PATH = shutil.which("nvcc")
if not NVCC_PATH:
    # Prevent torch.utils.cpp_extension from attempting CUDA checks when nvcc isn't available.
    os.environ.setdefault("FORCE_CUDA", "0")
else:
    # Point CUDA_HOME at the discovered nvcc path so torch finds the right toolkit (and not /usr/lib/cuda).
    os.environ["CUDA_HOME"] = str(Path(NVCC_PATH).parent.parent)


def _sync_cuda_home_env() -> None:
    """If CUDA_HOME is stale or missing, sync it to the nvcc location (once torch is imported)."""
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home:
        nvcc_candidate = Path(cuda_home) / "bin" / "nvcc"
        if nvcc_candidate.exists():
            return

    if NVCC_PATH:
        os.environ["CUDA_HOME"] = str(Path(NVCC_PATH).parent.parent)


_sync_cuda_home_env()

# Import torch for extensions
try:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

    print("Building pufferlib-core with C++/CUDA extensions")
except ImportError:
    print("Error: torch not available. Please install torch first.")
    sys.exit(1)

# Decide whether to build CUDA extension based on nvcc availability and version match with the torch wheel
def _detect_nvcc_version(nvcc_path: str | None) -> str | None:
    if not nvcc_path:
        return None
    try:
        out = subprocess.check_output([nvcc_path, "--version"], text=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    match = re.search(r"release (\d+\.\d+)", out)
    return match.group(1) if match else None


torch_cuda = getattr(torch.version, "cuda", None)
nvcc_ver = _detect_nvcc_version(NVCC_PATH)

use_cuda = False
if NVCC_PATH and torch_cuda and nvcc_ver:
    use_cuda = torch_cuda.startswith(nvcc_ver)
    if not use_cuda:
        print(
            f"CUDA mismatch: torch was built against {torch_cuda}, but nvcc reports {nvcc_ver}. "
            "Building CPU-only extension."
        )
else:
    if NVCC_PATH:
        print("nvcc found but unable to determine version; building CPU-only extension.")
    else:
        print("nvcc not found; building CPU-only extension.")

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

# Check if CUDA compiler is available and versions align with torch
if use_cuda:
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
