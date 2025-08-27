"""
Pytest configuration and fixtures for MCP server enhanced replay analysis tests.

This module provides fixtures for testing the enhanced replay analysis system
without graceful degradation - tests fail hard if required data is missing.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

from metta.mcp_server.stats_analysis import AgentStats, BuildingStats
from metta.mcp_server.wandb_integration import LearningProgression, WandbTrainingContext

# Set test API key for modules that require it during import
os.environ["ANTHROPIC_API_KEY"] = "test"


@pytest.fixture
def sample_replay_data() -> Dict[str, Any]:
    """Create sample replay data for testing"""
    return {
        "version": "1.0",
        "max_steps": 1000,
        "num_agents": 4,
        "map_size": [20, 20],
        "action_names": ["noop", "move", "rotate", "attack", "get_output", "put_recipe_items"],
        "inventory_items": [
            "ore_red",
            "ore_blue",
            "ore_green",
            "battery_red",
            "battery_blue",
            "battery_green",
            "heart",
        ],
        "item_names": ["ore_red", "ore_blue", "ore_green", "battery_red", "battery_blue", "battery_green", "heart"],
        "type_names": [
            "agent",
            "wall",
            "mine_red",
            "mine_blue",
            "mine_green",
            "generator_red",
            "generator_blue",
            "generator_green",
            "altar",
        ],
        "object_types": ["agent", "wall", "mine", "generator", "altar"],
        "objects": [
            {
                "agent_id": 0,
                "type_id": 0,
                "location": [[0, [5, 5, 0]], [100, [6, 5, 0]], [200, [7, 6, 0]]],
                "inventory": [[0, []], [50, [[0, 1]]], [100, [[0, 2], [1, 1]]], [150, [[1, 3]]]],
                "total_reward": [[0, 0.0], [50, 0.2], [100, 0.5], [150, 1.2]],
                "action_id": [[0, 0], [10, 1], [20, 4], [30, 5]],
                "action_param": [[0, 0], [10, 1], [20, 0], [30, 2]],
                "action_success": [[0, 1], [10, 1], [20, 1], [30, 0]],
            },
            {
                "agent_id": 1,
                "type_id": 0,
                "location": [[0, [8, 8, 0]], [100, [9, 8, 0]], [200, [9, 9, 0]]],
                "inventory": [[0, []], [75, [[0, 2]]], [125, [[1, 1]]]],
                "total_reward": [[0, 0.0], [75, 0.3], [125, 0.8]],
                "action_id": [[0, 1], [15, 4], [25, 5]],
                "action_param": [[0, 2], [15, 0], [25, 1]],
                "action_success": [[0, 1], [15, 1], [25, 1]],
            },
            {
                "type_id": 2,  # mine_red
                "id": 10,
                "location": [3, 3, 1],
            },
            {
                "type_id": 5,  # generator_red
                "id": 11,
                "location": [10, 10, 1],
            },
        ],
    }


@pytest.fixture
def sample_grid_objects_replay_data() -> Dict[str, Any]:
    """Create sample grid_objects format replay data for testing"""
    return {
        "version": "1.0",
        "max_steps": 800,
        "num_agents": 2,
        "map_size": [15, 15],
        "action_names": ["noop", "move", "rotate", "attack", "get_output", "put_recipe_items"],
        "inventory_items": ["ore_red", "battery_red", "heart"],
        "object_types": ["agent", "wall", "mine", "generator"],
        "grid_objects": [
            {
                "type": 0,  # agent
                "r": [[0, 5], [100, 6], [200, 7]],
                "c": [[0, 5], [100, 5], [200, 6]],
                "action": [[0, [0, 0]], [50, [1, 1]], [100, [4, 0]]],
                "action_success": [[0, 1], [50, 1], [100, 1]],
                "inv:ore_red": [[0, 0], [60, 1], [120, 0]],
                "inv:battery_red": [[0, 0], [150, 1]],
            },
            {
                "type": 0,  # agent
                "r": [[0, 8], [100, 9]],
                "c": [[0, 8], [100, 8]],
                "action": [[0, [1, 2]], [75, [4, 0]]],
                "action_success": [[0, 1], [75, 1]],
                "inv:ore_red": [[0, 0], [80, 2]],
                "inv:heart": [[0, 0], [200, 1]],
            },
        ],
    }


@pytest.fixture
def sample_episode_stats() -> Dict[str, Any]:
    """Create sample episode statistics for testing"""
    return {
        "game": {
            "episode_length": 1000,
            "total_agents": 4,
            "resource_scarcity_index": 0.3,
            "cooperation_index": 0.7,
            "competition_intensity": 0.4,
        },
        "agent": [
            {
                "agent_id": 0,
                "action.get_output.success": 15,
                "action.get_output.failed": 3,
                "action.put_recipe_items.success": 12,
                "action.put_recipe_items.failed": 1,
                "action.move.success": 45,
                "action.attack.success": 2,
                "ore_red.gained": 8,
                "ore_red.lost": 5,
                "battery_red.gained": 3,
                "movement.direction.up": 15,
                "movement.direction.right": 20,
                "red.hit.blue": 2,
                "friendly_fire": 0,
                "total_reward": 1.2,
            },
            {
                "agent_id": 1,
                "action.get_output.success": 10,
                "action.move.success": 38,
                "action.attack.success": 5,
                "ore_blue.gained": 6,
                "battery_blue.gained": 2,
                "heart.gained": 1,
                "movement.direction.left": 12,
                "movement.direction.down": 18,
                "blue.steals.ore_red.from.red": 1,
                "friendly_fire": 1,
                "total_reward": 2.1,
            },
        ],
        "converter": [
            {
                "type_id": 2,
                "location.r": 5,
                "location.c": 5,
                "conversions.started": 10,
                "conversions.completed": 8,
                "conversions.blocked": 2,
                "ore_red.produced": 12,
                "blocked.output_full": 1,
                "blocked.insufficient_input": 1,
            },
            {
                "type_id": 5,
                "location.r": 10,
                "location.c": 10,
                "conversions.started": 6,
                "conversions.completed": 5,
                "battery_red.produced": 5,
                "ore_red.consumed": 10,
                "cooldown.started": 3,
                "cooldown.completed": 3,
            },
        ],
    }


@pytest.fixture
def sample_agent_stats() -> list[AgentStats]:
    """Create sample AgentStats objects for testing"""
    return [
        AgentStats(
            agent_id=0,
            total_actions={"get_output": 18, "put_recipe_items": 13, "move": 45, "attack": 2},
            action_success_rates={"get_output": 0.83, "put_recipe_items": 0.92, "move": 1.0, "attack": 1.0},
            resource_flows={"ore_red": {"gained": 8, "lost": 5}, "battery_red": {"gained": 3, "lost": 0}},
            movement_patterns={"movement.direction.up": 15, "movement.direction.right": 20},
            combat_stats={"red.hit.blue": 2, "friendly_fire": 0},
            building_interactions={"get": 15, "put": 12},
            efficiency_metrics={"overall_efficiency": 1.6, "cooperation_score": 1.0, "competition_score": 0.03},
        ),
        AgentStats(
            agent_id=1,
            total_actions={"get_output": 10, "move": 38, "attack": 5},
            action_success_rates={"get_output": 1.0, "move": 1.0, "attack": 1.0},
            resource_flows={
                "ore_blue": {"gained": 6, "lost": 0},
                "battery_blue": {"gained": 2, "lost": 0},
                "heart": {"gained": 1, "lost": 0},
            },
            movement_patterns={"movement.direction.left": 12, "movement.direction.down": 18},
            combat_stats={"blue.steals.ore_red.from.red": 1, "friendly_fire": 1},
            building_interactions={"get": 10, "steals": 1},
            efficiency_metrics={"overall_efficiency": 9.0, "cooperation_score": 0.83, "competition_score": 0.09},
        ),
    ]


@pytest.fixture
def sample_building_stats() -> list[BuildingStats]:
    """Create sample BuildingStats objects for testing"""
    return [
        BuildingStats(
            building_id=0,
            type_id=2,
            type_name="mine_red",
            location=(5, 5),
            production_efficiency={"conversions.started": 10, "conversions.completed": 8},
            resource_flows={"ore_red": {"produced": 12, "consumed": 0}},
            operational_stats={"conversions.blocked": 2},
            bottleneck_analysis={"blocked.output_full": 1, "blocked.insufficient_input": 1},
        ),
        BuildingStats(
            building_id=1,
            type_id=5,
            type_name="generator_red",
            location=(10, 10),
            production_efficiency={"conversions.started": 6, "conversions.completed": 5},
            resource_flows={"battery_red": {"produced": 5, "consumed": 0}, "ore_red": {"produced": 0, "consumed": 10}},
            operational_stats={"cooldown.started": 3, "cooldown.completed": 3},
            bottleneck_analysis={},
        ),
    ]


@pytest.fixture
def sample_wandb_context() -> WandbTrainingContext:
    """Create sample Wandb training context for testing"""
    reward_progression = LearningProgression(
        metric_name="env_agent/reward",
        trend="improving",
        trend_strength=0.7,
        current_value=1.5,
        baseline_value=0.3,
        progression_rate=0.12,
    )

    return WandbTrainingContext(
        run_name="test_run_123",
        run_url="https://wandb.ai/test/project/runs/test_run_123",
        replay_timestamp_step=8000,
        context_window_steps=2000,
        reward_progression=reward_progression,
        action_mastery_progression={
            "get_output": LearningProgression(
                metric_name="env_agent/action.get_output.success",
                trend="improving",
                trend_strength=0.6,
                current_value=0.85,
                baseline_value=0.4,
                progression_rate=0.045,
            )
        },
        resource_efficiency_progression={
            "ore_red": LearningProgression(
                metric_name="env_agent/ore_red.gained",
                trend="stable",
                trend_strength=0.2,
                current_value=5.2,
                baseline_value=4.8,
                progression_rate=0.004,
            )
        },
        movement_learning_progression={
            "up": LearningProgression(
                metric_name="env_agent/movement.direction.up",
                trend="volatile",
                trend_strength=0.4,
                current_value=0.25,
                baseline_value=0.25,
                progression_rate=0.0,
            )
        },
        training_stage="mid",
        learning_velocity=0.08,
        performance_stability=0.75,
        behavioral_adaptation_rate=0.3,
        behavior_metric_correlations={"reward_vs_resource_gain": 0.82, "reward_vs_action_success": 0.67},
        critical_learning_moments=[
            {
                "type": "performance_breakthrough",
                "step": 7500,
                "description": "Significant reward improvement: 1.20 → 1.58",
                "magnitude": 0.32,
            }
        ],
    )


@pytest.fixture
def temp_replay_file(sample_replay_data) -> Path:
    """Create temporary replay file for testing"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sample_replay_data, f)
        return Path(f.name)


@pytest.fixture
def mock_mcp_client():
    """Create mock MCP client for testing Wandb integration"""

    class MockMCPClient:
        def __init__(self):
            self.calls = []

        def __call__(self, function_name: str, params: dict):
            self.calls.append((function_name, params))

            if function_name == "mcp__metta__get_wandb_run":
                return {
                    "name": params["run_name"],
                    "state": "finished",
                    "summary": {"best_reward": 2.1},
                    "config": {"learning_rate": 0.001},
                }
            elif function_name == "mcp__metta__get_wandb_run_url":
                return f"https://wandb.ai/test/project/runs/{params['run_name']}"
            else:
                raise ValueError(f"Unsupported function: {function_name}")

    return MockMCPClient()


@pytest.fixture(scope="session", autouse=True)
def ensure_no_graceful_degradation():
    """
    Ensure tests fail hard if required data is missing.

    This fixture enforces the 'no graceful degradation' requirement by
    setting up the test environment to raise errors when data is missing.
    """
    import os

    os.environ["MCP_TEST_MODE"] = "strict"
    yield
    if "MCP_TEST_MODE" in os.environ:
        del os.environ["MCP_TEST_MODE"]


# Test data validation utilities
def assert_required_stats_present(stats_data: dict, required_fields: list[str]):
    """Assert that all required fields are present in stats data"""
    for field in required_fields:
        assert field in stats_data, f"Required field '{field}' missing from stats data"
        assert stats_data[field] is not None, f"Required field '{field}' is None"


def assert_wandb_data_complete(wandb_context: WandbTrainingContext):
    """Assert that Wandb context contains all required data"""
    assert wandb_context.run_name, "Wandb run name is required"
    assert wandb_context.replay_timestamp_step > 0, "Replay timestamp step must be positive"
    assert wandb_context.reward_progression is not None, "Reward progression is required"
    assert wandb_context.training_stage in ["early", "mid", "late"], (
        f"Invalid training stage: {wandb_context.training_stage}"
    )
