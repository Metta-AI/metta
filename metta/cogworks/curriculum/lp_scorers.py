"""Backward compatibility shim for lp_scorers.

DEPRECATED: Use 'from agora.algorithms import LPScorer, BasicLPScorer, BidirectionalLPScorer' instead.
"""

import warnings

warnings.warn(
    "metta.cogworks.curriculum.lp_scorers is deprecated. "
    "Use 'from agora.algorithms import LPScorer, BasicLPScorer, BidirectionalLPScorer' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from agora.algorithms import BasicLPScorer, BidirectionalLPScorer, LPScorer  # noqa: E402

__all__ = [
    "LPScorer",
    "BasicLPScorer",
    "BidirectionalLPScorer",
]
