"""Benchmark architecture recipes with progressive difficulty levels.

This package contains 5 difficulty levels for benchmarking agent architectures:

- Level 1 (Basic): Maximum reward shaping, small map, no combat
- Level 2 (Easy): Moderate reward shaping, standard map, no combat
- Level 3 (Medium): Low reward shaping, combat enabled, standard map
- Level 4 (Hard): Sparse rewards, combat enabled, more agents
- Level 5 (Expert): No intermediate rewards, curriculum learning, full complexity

Each level is designed to test different aspects of architecture performance,
from basic learning capabilities to complex strategic reasoning.

Example usage:
    uv run ./tools/run.py experiments.recipes.benchmark_architectures.level_1_basic.train
    uv run ./tools/run.py experiments.recipes.benchmark_architectures.level_3_medium.train
    uv run ./tools/run.py experiments.recipes.benchmark_architectures.level_5_expert.train
"""
