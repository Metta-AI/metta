import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

from setuptools import setup

# Always build extensions
BUILD_EXTENSIONS = True

def _nvcc_version(nvcc_path: Optional[str]) -> Optional[str]:
    """Return nvcc major.minor version string (e.g., '12.4')."""
    if not nvcc_path:
        return None
    try:
        output = subprocess.check_output([nvcc_path, "--version"], text=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    match = re.search(r"release (\d+\.\d+)", output)
    return match.group(1) if match else None


def _cuda_choice(torch_cuda: Optional[str], nvcc_path: Optional[str]) -> Tuple[bool, str]:
    """Decide CUDA vs CPU build and a short reason."""
    nvcc_version = _nvcc_version(nvcc_path)
    if not nvcc_path:
        return False, "nvcc not found; building CPU-only extension."
    if not nvcc_version:
        return False, "nvcc version unavailable; building CPU-only extension."
    if not torch_cuda:
        return False, "torch CUDA version unavailable; building CPU-only extension."
    if torch_cuda.startswith(nvcc_version):
        return True, f"CUDA versions align (torch {torch_cuda}, nvcc {nvcc_version}); building with CUDA."
    return False, (
        f"CUDA mismatch: torch built against {torch_cuda}, nvcc reports {nvcc_version}; "
        "building CPU-only extension."
    )


# Detect nvcc and prime CUDA-related env vars for torch's build helpers.
NVCC_PATH = shutil.which("nvcc")
if NVCC_PATH:
    os.environ["CUDA_HOME"] = str(Path(NVCC_PATH).parent.parent)
else:
    os.environ.setdefault("FORCE_CUDA", "0")

# Import torch for extensions
try:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

    print("Building pufferlib-core with C++/CUDA extensions")
except ImportError:
    print("Error: torch not available. Please install torch first.")
    sys.exit(1)

use_cuda, cuda_reason = _cuda_choice(getattr(torch.version, "cuda", None), NVCC_PATH)
print(cuda_reason)

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
