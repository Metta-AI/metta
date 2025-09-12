"""Learning progress modules for curriculum algorithms.

This module provides the modular components that comprise the high-performance
learning progress curriculum system:

- TaskTracker: Task memory and performance history management
- LearningProgressScorer: Standard EMA-based learning progress scoring
- BidirectionalLearningProgressScorer: Fast/slow EMA difference scoring
- BucketAnalyzer: Parameter space completion density analysis

These components are designed to work together to provide efficient,
intelligent task selection for curriculum-based RL training.
"""

from .bucket_analyzer import BucketAnalyzer
from .learning_progress_scorer import LearningProgressScorer
from .task_tracker import TaskTracker

__all__ = ["BucketAnalyzer", "LearningProgressScorer", "TaskTracker"]
