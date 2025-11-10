from __future__ import annotations

import pytest

from experiments.recipes.arena_basic_easy_shaped import (
    sweep_full,
    sweep_minimal,
    sweep_muon,
)
from metta.tools.ray_sweep import RaySweepTool


@pytest.mark.parametrize(
    ("factory", "expect"),
    [
        (
            sweep_muon,
            {
                "timesteps": 2_000_000_000,
                "gpus": 4,
                "optimizer_type": "muon",
            },
        ),
        (
            sweep_minimal,
            {
                "timesteps": 500_000_000,
                "gpus": 2,
            },
        ),
        (
            sweep_full,
            {
                "timesteps": 2_000_000_000,
                "gpus": 4,
            },
        ),
    ],
)
def test_recipe_factories_return_configured_ray_sweep_tools(factory, expect):
    sweep_name = f"{factory.__name__}-smoke"
    tool = factory(sweep_name)

    assert isinstance(tool, RaySweepTool)
    assert tool.sweep_config.sweep_id == sweep_name
    assert tool.search_space, "search space must not be empty"

    assert tool.search_space["trainer.total_timesteps"] == expect["timesteps"]
    assert tool.sweep_config.gpus_per_trial == expect["gpus"]

    if "optimizer_type" in expect:
        assert tool.search_space["trainer.optimizer.type"] == expect["optimizer_type"]
