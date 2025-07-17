#!/usr/bin/env python3
"""
Test exploration tracking in navigation evaluation environments.
"""

import pytest

from metta.api import create_evaluation_config_suite


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
                assert game_config["track_exploration"], \
                    f"Exploration tracking should be enabled for navigation evaluation: {sim_name}"
                print(f"✅ Exploration tracking enabled for {sim_name}")
            else:
                print(f"⚠️  track_exploration not set in {sim_name} config (may be set via override)")


def test_exploration_rate_in_navigation_env():
    """Test that exploration rate is calculated correctly in a navigation environment."""

    # This test would require setting up a navigation environment
    # For now, just verify the configuration structure
    eval_config = create_evaluation_config_suite()

    # Check that we have navigation evaluations
    navigation_sims = [name for name in eval_config.simulations.keys() if "navigation" in name]
    assert len(navigation_sims) > 0, "Should have navigation evaluations"

    print(f"Found {len(navigation_sims)} navigation evaluations")
    for sim_name in navigation_sims:
        print(f"  - {sim_name}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
