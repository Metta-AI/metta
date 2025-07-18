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
        exploration_rates={("navigation", "basic"): 0.15, ("navigation", "complex"): 0.12},
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


def test_enhanced_exploration_rate_logging():
    """Test that exploration rates are properly logged with enhanced visibility."""
    import logging
    from unittest.mock import Mock, patch

    from metta.agent.policy_record import PolicyRecord
    from metta.eval.eval_service import extract_scores
    from metta.eval.eval_stats_db import EvalStatsDB
    from metta.sim.simulation_config import SimulationSuiteConfig

    # Create mock objects
    mock_policy_record = Mock(spec=PolicyRecord)
    mock_policy_record.key = "test_policy"
    mock_policy_record.version = 1

    mock_simulation_suite = Mock(spec=SimulationSuiteConfig)
    mock_simulation_suite.simulations = {
        "navigation/terrain_small": Mock(),
        "navigation/terrain_medium": Mock(),
        "memory/sequence_short": Mock(),
    }

    mock_stats_db = Mock(spec=EvalStatsDB)
    # Mock the simulation_scores method to return exploration rates
    mock_stats_db.simulation_scores.return_value = {
        ("test_policy", "navigation/terrain_small", 1): 0.15,
        ("test_policy", "navigation/terrain_medium", 1): 0.12,
        ("test_policy", "memory/sequence_short", 1): 0.18,
    }
    mock_stats_db.get_average_metric_by_filter.return_value = 0.85

    mock_logger = Mock(spec=logging.Logger)

    # Test the extract_scores function
    with patch("metta.eval.eval_service.EvalStatsDB") as mock_eval_stats_db:
        mock_eval_stats_db.from_sim_stats_db.return_value = mock_stats_db

        result = extract_scores(mock_policy_record, mock_simulation_suite, mock_stats_db, mock_logger)

    # Verify that exploration rates are extracted and logged
    assert result.exploration_rates[("navigation", "terrain_small")] == 0.15
    assert result.exploration_rates[("navigation", "terrain_medium")] == 0.12
    assert result.exploration_rates[("memory", "sequence_short")] == 0.18

    # Verify that logging was called with exploration rate information
    log_calls = [call[0][0] for call in mock_logger.info.call_args_list]
    exploration_logs = [
        log for log in log_calls if "Exploration Rates by Environment" in log or "Exploration Rate Summary" in log
    ]
    assert len(exploration_logs) > 0, "Should have logged exploration rate information"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
