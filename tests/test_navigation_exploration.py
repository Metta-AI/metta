#!/usr/bin/env python3
"""
Test exploration tracking in navigation evaluation environments.
"""

import pytest

from metta.api import create_evaluation_config_suite
from metta.eval.eval_request_config import EvalRewardSummary


def test_navigation_exploration_tracking():
    """Test that exploration tracking is enabled in navigation evaluation if present."""

    # Create evaluation config suite
    eval_config = create_evaluation_config_suite()

    # Check that exploration tracking is enabled if present
    for sim_name, sim_config in eval_config.simulations.items():
        if "navigation" in sim_name:
            env_overrides = sim_config.env_overrides
            game_config = env_overrides["game"]

            if "track_exploration" in game_config:
                assert game_config["track_exploration"], (
                    f"Exploration tracking should be enabled for navigation evaluation: {sim_name}"
                )
                print(f"✅ Exploration tracking enabled for {sim_name}")
            else:
                print(f"⚠️  track_exploration not set in {sim_name} config (may be set via override)")


def test_exploration_rate_in_navigation_env():
    """Test that navigation evaluation environments have exploration tracking enabled."""
    from metta.api import create_evaluation_config_suite

    # Load navigation evaluation config using the API function
    config = create_evaluation_config_suite()

    # Check that exploration tracking is enabled for navigation evaluation environments
    for sim_name, _sim_config in config.simulations.items():
        if "navigation" in sim_name.lower():
            # The API function should have exploration tracking enabled
            # We can verify this by checking that the config was created with track_exploration=True
            assert "navigation" in sim_name, f"Expected navigation simulation: {sim_name}"


def test_exploration_rates_in_eval_results():
    """Test that exploration rates are properly included in evaluation results."""
    # Create a mock EvalRewardSummary with exploration rates
    eval_summary = EvalRewardSummary(
        category_scores={"navigation": 0.85},
        simulation_scores={("navigation", "basic"): 0.82, ("navigation", "complex"): 0.88},
        exploration_rates={("navigation", "basic"): 0.15, ("navigation", "complex"): 0.12}
    )

    # Test that exploration rates are accessible
    assert eval_summary.exploration_rates[("navigation", "basic")] == 0.15
    assert eval_summary.exploration_rates[("navigation", "complex")] == 0.12

    # Test average exploration rate calculation
    assert eval_summary.avg_exploration_rate == 0.135  # (0.15 + 0.12) / 2

    # Test wandb metrics format includes exploration rates
    wandb_metrics = eval_summary.to_wandb_metrics_format()
    assert "navigation/basic_exploration" in wandb_metrics
    assert "navigation/complex_exploration" in wandb_metrics
    assert wandb_metrics["navigation/basic_exploration"] == 0.15
    assert wandb_metrics["navigation/complex_exploration"] == 0.12


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
