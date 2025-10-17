"""Task generation for curriculum learning."""

from agora.generators.base import Span, TaskGenerator, TaskGeneratorConfig, TTaskGenerator
from agora.generators.bucketed import BucketedTaskGenerator
from agora.generators.set import TaskGeneratorSet
from agora.generators.single import SingleTaskGenerator

__all__ = [
    # Base classes
    "TaskGenerator",
    "TaskGeneratorConfig",
    "TTaskGenerator",
    "Span",
    # Concrete generators
    "SingleTaskGenerator",
    "BucketedTaskGenerator",
    "TaskGeneratorSet",
]
