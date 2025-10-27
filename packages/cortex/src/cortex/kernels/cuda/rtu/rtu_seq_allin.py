import os

from torch.utils.cpp_extension import load

_mod_path = os.path.dirname(__file__)
_ext = None


def _load_ext():
    global _ext
    if _ext is not None:
        return _ext
    sources = [
        os.path.join(_mod_path, "rtu_seq_allin_binding.cpp"),
        os.path.join(_mod_path, "rtu_seq_allin_kernels.cu"),
    ]
    _ext = load(
        name="rtu_seq_allin",
        sources=sources,
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-Xptxas", "-O3"],
        verbose=False,
    )
    return _ext


def forward_allin(*args, **kwargs):
    return _load_ext().forward_allin(*args, **kwargs)


def backward_allin(*args, **kwargs):
    return _load_ext().backward_allin(*args, **kwargs)
