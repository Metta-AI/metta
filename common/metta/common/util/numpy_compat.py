"""
NumPy 2.0 compatibility shim for WandB.

WandB 0.20.1 expects certain deprecated NumPy attributes that were removed in NumPy 2.0.
This module adds them back to maintain compatibility.
"""

import numpy as np

# Add back deprecated NumPy attributes that WandB expects
if not hasattr(np, "byte"):
    np.byte = np.int8

if not hasattr(np, "short"):
    np.short = np.int16

if not hasattr(np, "intc"):
    np.intc = np.int32

if not hasattr(np, "int_"):
    np.int_ = np.int64

if not hasattr(np, "longlong"):
    np.longlong = np.int64

if not hasattr(np, "ubyte"):
    np.ubyte = np.uint8

if not hasattr(np, "ushort"):
    np.ushort = np.uint16

if not hasattr(np, "uintc"):
    np.uintc = np.uint32

if not hasattr(np, "uint"):
    np.uint = np.uint64

if not hasattr(np, "ulonglong"):
    np.ulonglong = np.uint64

if not hasattr(np, "float_"):
    np.float_ = np.float64

if not hasattr(np, "double"):
    np.double = np.float64

if not hasattr(np, "longdouble"):
    np.longdouble = np.float128 if hasattr(np, "float128") else np.float64

if not hasattr(np, "csingle"):
    np.csingle = np.complex64

if not hasattr(np, "cdouble"):
    np.cdouble = np.complex128

if not hasattr(np, "clongdouble"):
    np.clongdouble = np.complex256 if hasattr(np, "complex256") else np.complex128
