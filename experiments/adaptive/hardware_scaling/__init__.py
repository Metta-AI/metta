"""Hardware scaling experiment utilities.

This package provides a minimal interface to run one Bayesian sweep per
hardware configuration (GPU/node pair) using the existing SweepTool. See
`hw_sweep` for the Tool factory and `launch.py` for a simple multi-pair
launcher.
"""

from .hw_sweep import hw_sweep

__all__ = ["hw_sweep"]
