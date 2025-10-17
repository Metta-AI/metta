"""Backward compatibility shim for task_generator.

DEPRECATED: Use 'from agora import TaskGenerator, SingleTaskGenerator, etc.' instead.
"""

import warnings

warnings.warn(
    "metta.cogworks.curriculum.task_generator is deprecated. "
    "Use 'from agora import TaskGenerator, SingleTaskGenerator, BucketedTaskGenerator' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from agora import (  # noqa: E402
    BucketedTaskGenerator,
    SingleTaskGenerator,
    Span,
    TaskGenerator,
    TaskGeneratorConfig,
    TaskGeneratorSet,
)

# Backward compatibility alias
AnyTaskGeneratorConfig = TaskGeneratorConfig

__all__ = [
    "TaskGenerator",
    "TaskGeneratorConfig",
    "AnyTaskGeneratorConfig",
    "SingleTaskGenerator",
    "BucketedTaskGenerator",
    "TaskGeneratorSet",
    "Span",
]
