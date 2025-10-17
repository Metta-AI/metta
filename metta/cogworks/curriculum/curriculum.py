"""Backward compatibility shim for curriculum.

DEPRECATED: Use 'from agora import Curriculum, CurriculumConfig, CurriculumTask' instead.
"""

import warnings

warnings.warn(
    "metta.cogworks.curriculum.curriculum is deprecated. "
    "Use 'from agora import Curriculum, CurriculumConfig, CurriculumTask' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from agora import (  # noqa: E402
    Curriculum,
    CurriculumAlgorithm,
    CurriculumAlgorithmConfig,
    CurriculumConfig,
    CurriculumTask,
    DiscreteRandomConfig,
    DiscreteRandomCurriculum,
)

__all__ = [
    "Curriculum",
    "CurriculumConfig",
    "CurriculumTask",
    "CurriculumAlgorithm",
    "CurriculumAlgorithmConfig",
    "DiscreteRandomConfig",
    "DiscreteRandomCurriculum",
]
