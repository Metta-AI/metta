"""RTU (Real-Time Unit) PyTorch kernels subpackage.

Exports:
- rtu_stream_diag_pytorch: streaming diagonal-input RTU kernel (D == H)
"""

from .rtu_stream_diag import rtu_stream_diag_pytorch

__all__ = ["rtu_stream_diag_pytorch"]

