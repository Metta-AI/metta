"""Backward compatibility shim for learning_progress_algorithm.

DEPRECATED: Use 'from agora import LearningProgressAlgorithm, LearningProgressConfig' instead.
"""

import warnings

warnings.warn(
    "metta.cogworks.curriculum.learning_progress_algorithm is deprecated. "
    "Use 'from agora import LearningProgressAlgorithm, LearningProgressConfig' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from agora import LearningProgressAlgorithm, LearningProgressConfig  # noqa: E402

__all__ = ["LearningProgressAlgorithm", "LearningProgressConfig"]
