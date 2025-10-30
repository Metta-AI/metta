import os

import torch
from torch.utils.cpp_extension import load

_mod_path = os.path.dirname(__file__)
_ext = None


def _load_ext():
    global _ext
    if _ext is not None:
        return _ext
    if torch.cuda.is_available():
        # Determine local GPU archs (e.g., ['8.9']).
        caps = sorted(
            {
                f"{maj}.{min}"
                for i in range(torch.cuda.device_count())
                for (maj, min) in [torch.cuda.get_device_capability(i)]
            }
        )
        if caps:
            desired = ";".join(caps)
            env_arch = os.environ.get("TORCH_CUDA_ARCH_LIST", "")
            # If env arch list misses the local archs, override to avoid PTX JIT.
            if not all(c in env_arch for c in caps):
                os.environ["TORCH_CUDA_ARCH_LIST"] = desired
    sources = [
        os.path.join(_mod_path, "rtu_seq_allin_binding.cpp"),
        os.path.join(_mod_path, "rtu_seq_allin_kernels.cu"),
    ]
    _ext = load(
        name="rtu_seq_allin",
        sources=sources,
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-Xptxas", "-O3"],
        # Use torch's default build cache under ~/.cache/torch_extensions
        # (avoid creating per-arch folders inside the repo).
        build_directory=None,
        verbose=False,
    )
    return _ext


def forward_allin(*args, **kwargs):
    return _load_ext().forward_allin(*args, **kwargs)


def backward_allin(*args, **kwargs):
    return _load_ext().backward_allin(*args, **kwargs)
