import os
import platform
import shutil
import sys

from setuptools import setup

# Always build extensions
BUILD_EXTENSIONS = True

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

# Determine whether to build CUDA extensions
cpu_only_env = os.getenv("PUFFERLIB_CPU_ONLY")
has_nvcc = shutil.which("nvcc") is not None

# CPU-only is an explicit opt-in. Only the values "1", "true", or "on"
# (case-insensitive) are honored.
cpu_only_values = {"1", "true", "on"}

cpu_only = cpu_only_env is not None and cpu_only_env.lower() in cpu_only_values

if cpu_only:
    extension_class = CppExtension
    print(
        "Building with CPU-only support "
        f"(PUFFERLIB_CPU_ONLY={cpu_only_env!r})",
    )
elif has_nvcc:
    extension_class = CUDAExtension
    torch_sources.append("src/pufferlib/extensions/cuda/pufferlib.cu")
    if cpu_only_env is None:
        print(
            "Building with CUDA support "
            "(nvcc detected; PUFFERLIB_CPU_ONLY not set)",
        )
    else:
        print(
            "Ignoring PUFFERLIB_CPU_ONLY value "
            f"{cpu_only_env!r}; expected one of {sorted(cpu_only_values)}; "
            "building with CUDA support (nvcc detected)",
        )
else:
    # No explicit CPU-only opt-in and nvcc is missing. For tooling and
    # CPU-only environments (e.g. Docker build orchestration) we default
    # to a CPU-only build instead of failing hard.
    extension_class = CppExtension
    print(
        "nvcc not found and PUFFERLIB_CPU_ONLY is not set; "
        "falling back to CPU-only build of pufferlib-core.",
        file=sys.stderr,
    )

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
