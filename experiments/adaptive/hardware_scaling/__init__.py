"""Hardware scaling experiment utilities.

This package provides a minimal interface to run one Bayesian sweep per
hardware configuration (GPU/node pair) using the simplified sweep API. See
`fom_sweep` for the sweep function and `launch.py` for a simple multi-pair
launcher.
"""

from .fom_sweep import sweep

__all__ = ["sweep"]
