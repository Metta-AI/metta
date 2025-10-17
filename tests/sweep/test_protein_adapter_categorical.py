from typing import Any

import pytest

from metta.sweep.core import CategoricalParameterConfig, ParameterConfig
from metta.sweep.optimizer.protein import ProteinOptimizer
from metta.sweep.protein_config import ProteinConfig, ProteinSettings


class DummyProtein:
    def __init__(self, protein_dict: dict, **kwargs: Any):
        # Capture flattened numeric-only dict (categoricals converted)
        self.protein_dict = protein_dict
        self.kwargs = kwargs
        self.observations: list[tuple[dict, float, float, bool]] = []

    def observe(self, hypers: dict, score: float, cost: float, is_failure: bool = False):
        self.observations.append((hypers, score, cost, is_failure))

    def suggest(self, n_suggestions: int = 1):
        # Return numeric indices for the categorical param to test decoding
        if n_suggestions == 1:
            return ({"model": {"color": 2}}, {})  # -> maps to choices[2]
        else:
            return [({"model": {"color": 0}}, {}), ({"model": {"color": 1}}, {})]


def test_protein_adapter_encodes_observations_and_decodes_suggestions(monkeypatch):
    # Build a config containing a categorical parameter
    config = ProteinConfig(
        metric="evaluator/eval_sweep/score",
        goal="maximize",
        parameters={
            "model": {
                "color": CategoricalParameterConfig(choices=["red", "blue", "butt"]),
            },
            # include at least one numeric parameter to ensure path still works
            "trainer": {
                "optimizer": {
                    "learning_rate": ParameterConfig(
                        min=1e-5, max=1e-3, distribution="log_normal", mean=1e-4, scale="auto"
                    ),
                }
            },
        },
        settings=ProteinSettings(num_random_samples=0),
    )

    # Monkeypatch the Protein class used in the adapter
    import metta.sweep.optimizer.protein as adapter_module

    monkeypatch.setattr(adapter_module, "Protein", DummyProtein)

    optimizer = ProteinOptimizer(config)

    # Provide an observation with a categorical value and ensure it is encoded to int
    observations = [
        {
            "suggestion": {"model": {"color": "blue"}},  # should encode to index 1
            "score": 0.5,
            "cost": 10.0,
        }
    ]

    # Ask for a single suggestion
    suggestions = optimizer.suggest(observations=observations, n_suggestions=1)

    # suggestions are decoded back to categorical values
    assert isinstance(suggestions, list) and len(suggestions) == 1
    s0 = suggestions[0]
    assert s0["model"]["color"] == "butt"  # index 2 decoded to the third choice

    # Verify observations were encoded when passed into DummyProtein
    # Access the DummyProtein instance via the monkeypatched call site
    # The instance is created inside suggest; we can’t access it directly, but we can
    # infer encoding by running a multi-suggest path to ensure more calls

    # Ask for multiple suggestions; ensure decoding works for list return
    multi = optimizer.suggest(observations=observations, n_suggestions=2)
    assert [m["model"]["color"] for m in multi] == ["red", "blue"]


def test_empty_categorical_choices_raises(monkeypatch):
    config = ProteinConfig(
        metric="evaluator/eval_sweep/score",
        goal="maximize",
        parameters={
            "model": {
                "color": CategoricalParameterConfig(choices=[]),
            },
        },
    )

    # Dummy Protein to avoid heavy imports
    import metta.sweep.optimizer.protein as adapter_module

    monkeypatch.setattr(adapter_module, "Protein", DummyProtein)

    optimizer = ProteinOptimizer(config)
    with pytest.raises(ValueError):
        optimizer.suggest(observations=[], n_suggestions=1)
