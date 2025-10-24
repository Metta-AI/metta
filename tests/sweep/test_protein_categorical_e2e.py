"""End-to-end categorical test using real ProteinOptimizer and Protein.

This test exercises the adapter path without monkeypatching:
- Categorical values are encoded to indices for Protein
- Suggestions are decoded back to string categories
- Numeric parameter bounds are respected
"""

from typing import Any

from metta.sweep.core import CategoricalParameterConfig, ParameterConfig
from metta.sweep.optimizer.protein import ProteinOptimizer
from metta.sweep.protein_config import ProteinConfig, ProteinSettings


def test_e2e_categorical_single_and_multi_suggestions() -> None:
    """Verify end-to-end categorical handling with real optimizer."""
    # Build a config containing one categorical and one numeric parameter
    config = ProteinConfig(
        metric="score",
        goal="maximize",
        parameters={
            "model": {
                "color": CategoricalParameterConfig(choices=["red", "blue", "green"]),
            },
            "trainer": {
                "optimizer": {
                    "learning_rate": ParameterConfig(
                        min=1e-5, max=1e-3, distribution="log_normal", mean=1e-4, scale="auto"
                    ),
                }
            },
        },
        # Use defaults but ensure we seed with search center for deterministic first suggestion
        settings=ProteinSettings(num_random_samples=0, seed_with_search_center=True),
    )

    optimizer = ProteinOptimizer(config)

    # 1) Single suggestion with no observations should return the search center.
    # For a 3-choice categorical, the center index is 1 -> "blue".
    suggestions = optimizer.suggest(observations=[], n_suggestions=1)

    assert isinstance(suggestions, list) and len(suggestions) == 1
    s0 = suggestions[0]
    # Real Protein returns flat keys (e.g., "model.color")
    assert s0["model.color"] == "blue"
    lr0: float = s0["trainer.optimizer.learning_rate"]
    assert 1e-5 <= lr0 <= 1e-3

    # 2) Provide an observation that includes a categorical value and request multiple suggestions.
    # Ensure all returned categorical values are valid strings from the choices.
    observations: list[dict[str, Any]] = [
        {
            # Use flat keys to align with Protein's expectations
            "suggestion": {"model.color": "green", "trainer.optimizer.learning_rate": 2e-4},
            "score": 0.5,
            "cost": 10.0,
        }
    ]

    multi = optimizer.suggest(observations=observations, n_suggestions=2)
    assert isinstance(multi, list) and len(multi) == 2

    choices = {"red", "blue", "green"}
    for s in multi:
        assert s["model.color"] in choices
        lr: float = s["trainer.optimizer.learning_rate"]
        assert 1e-5 <= lr <= 1e-3


def test_categorical_sampling_reaches_all_choices() -> None:
    """Ensure repeated suggestions cover every categorical option."""
    # Configure optimizer to skip search-center seeding so we rely on random sampling.
    settings = ProteinSettings(
        num_random_samples=10,
        random_suggestions=32,
        seed_with_search_center=False,
        randomize_acquisition=False,
    )
    config = ProteinConfig(
        metric="score",
        goal="maximize",
        parameters={
            "model": {
                "color": CategoricalParameterConfig(choices=["red", "blue", "green"]),
            }
        },
        settings=settings,
    )

    optimizer = ProteinOptimizer(config)

    # Deterministic sampling for the test.
    import numpy as np
    import random

    np.random.seed(12345)
    random.seed(12345)

    seen: set[str] = set()
    # Pull several batches; with the wider categorical scale we should cover all choices.
    for _ in range(5):
        suggestions = optimizer.suggest(observations=[], n_suggestions=8)
        for suggestion in suggestions:
            seen.add(suggestion["model.color"])
        if seen == {"red", "blue", "green"}:
            break

    assert seen == {"red", "blue", "green"}
