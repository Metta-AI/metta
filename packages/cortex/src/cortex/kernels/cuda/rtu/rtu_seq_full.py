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
            if not all(c in env_arch for c in caps):
                os.environ["TORCH_CUDA_ARCH_LIST"] = desired
    sources = [
        os.path.join(_mod_path, "rtu_seq_full_binding.cpp"),
        os.path.join(_mod_path, "rtu_seq_full_kernels.cu"),
    ]
    _ext = load(
        name="rtu_seq_full",
        sources=sources,
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-Xptxas", "-O3"],
        build_directory=None,
        verbose=False,
    )
    return _ext


def forward_full(*args, **kwargs):
    return _load_ext().forward_full(*args, **kwargs)


def backward_full(*args, **kwargs):
    return _load_ext().backward_full(*args, **kwargs)
