"""RTU (Real-Time Unit) PyTorch kernels subpackage.

Exports:
- rtu_stream_diag_pytorch: streaming diagonal-input RTU kernel (D == H)
- rtu_stream_full_pytorch: streaming RTU with full‑rank input maps (D -> H)
"""

from .rtu_stream_diag import rtu_stream_diag_pytorch
from .rtu_stream_fullrank import rtu_stream_full_pytorch

__all__ = [
    "rtu_stream_diag_pytorch",
    "rtu_stream_full_pytorch",
]
