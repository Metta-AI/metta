"""Test track_invalid_arg configuration."""

import numpy as np

from mettagrid.config.mettagrid_c_config import from_mettagrid_config
from mettagrid.config.mettagrid_config import (
    ActionConfig,
    ActionsConfig,
    AgentConfig,
    GameConfig,
    WallConfig,
)
from mettagrid.mettagrid_c import MettaGrid, dtype_actions


def create_config(track_invalid_arg: bool = False) -> GameConfig:
    """Create a minimal valid game configuration."""
    return GameConfig(
        resource_names=["ore", "wood"],
        num_agents=1,
        max_steps=100,
        obs_width=7,
        obs_height=7,
        num_observation_tokens=50,
        agent=AgentConfig(
            freeze_duration=0,
            resource_limits={"ore": 10, "wood": 10},
        ),
        actions=ActionsConfig(
            move=ActionConfig(track_invalid_arg=track_invalid_arg),
            noop=ActionConfig(),
            rotate=ActionConfig(),
        ),
        objects={"wall": WallConfig(type_id=1, swappable=False)},
        allow_diagonals=True,
    )


def create_simple_map():
    """Create a simple 5x5 map with walls around edges."""
    return [
        ["wall", "wall", "wall", "wall", "wall"],
        ["wall", "empty", "empty", "empty", "wall"],
        ["wall", "empty", "agent.default", "empty", "wall"],
        ["wall", "empty", "empty", "empty", "wall"],
        ["wall", "wall", "wall", "wall", "wall"],
    ]


def test_track_invalid_arg_disabled():
    """Test that invalid action argument stats are NOT tracked when track_invalid_arg=False."""
    config = create_config(track_invalid_arg=False)
    simple_map = create_simple_map()
    env = MettaGrid(from_mettagrid_config(config), simple_map, 42)
    env.reset()

    move_idx = env.action_names().index("move")
    max_args = env.max_action_args()

    # Try argument exceeding max_arg
    invalid_arg = max_args[move_idx] + 10
    invalid_action = np.array([[move_idx, invalid_arg]], dtype=dtype_actions)
    env.step(invalid_action)

    # Action should fail
    assert not env.action_success()[0], "Invalid action argument should fail"

    # Get stats for the agent
    stats = env.agent_stats(0)

    # Check that the general invalid_arg stat is incremented
    assert stats.get("action.invalid_arg", 0) > 0, "action.invalid_arg should be tracked"

    # Check that the specific stat is NOT tracked
    specific_stat = f"action.invalid_arg.{move_idx}.{invalid_arg}"
    assert stats.get(specific_stat, 0) == 0, f"{specific_stat} should NOT be tracked when track_invalid_arg=False"

    print("✓ Test passed: track_invalid_arg=False works correctly")


def test_track_invalid_arg_enabled():
    """Test that invalid action argument stats ARE tracked when track_invalid_arg=True."""
    config = create_config(track_invalid_arg=True)
    simple_map = create_simple_map()
    env = MettaGrid(from_mettagrid_config(config), simple_map, 42)
    env.reset()

    move_idx = env.action_names().index("move")
    max_args = env.max_action_args()

    # Try argument exceeding max_arg
    invalid_arg = max_args[move_idx] + 10
    invalid_action = np.array([[move_idx, invalid_arg]], dtype=dtype_actions)
    env.step(invalid_action)

    # Action should fail
    assert not env.action_success()[0], "Invalid action argument should fail"

    # Get stats for the agent
    stats = env.agent_stats(0)

    # Check that the general invalid_arg stat is incremented
    assert stats.get("action.invalid_arg", 0) > 0, "action.invalid_arg should be tracked"

    # Check that the specific stat IS tracked
    specific_stat = f"action.invalid_arg.{move_idx}.{invalid_arg}"
    assert stats.get(specific_stat, 0) > 0, f"{specific_stat} should be tracked when track_invalid_arg=True"

    print("✓ Test passed: track_invalid_arg=True works correctly")


if __name__ == "__main__":
    test_track_invalid_arg_disabled()
    test_track_invalid_arg_enabled()
    print("\n✅ All tests passed!")
