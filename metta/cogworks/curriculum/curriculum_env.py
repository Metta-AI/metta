"""Backward compatibility shim for curriculum_env.

DEPRECATED: Use 'from agora.wrappers import CurriculumEnv' instead.
"""

import warnings

warnings.warn(
    "metta.cogworks.curriculum.curriculum_env is deprecated. Use 'from agora.wrappers import CurriculumEnv' instead.",
    DeprecationWarning,
    stacklevel=2,
)

try:
    from agora.wrappers import CurriculumEnv

    __all__ = ["CurriculumEnv"]
except ImportError:
    # pufferlib not available
    CurriculumEnv = None  # type: ignore[misc, assignment]
    __all__ = []
