"""Test that exceptions are raised when attack resources are not in inventory."""

import pytest

from mettagrid.config.mettagrid_c_config import from_mettagrid_config
from mettagrid.config.mettagrid_config import (
    ActionConfig,
    ActionsConfig,
    AgentConfig,
    AgentRewards,
    AttackActionConfig,
    ChangeGlyphActionConfig,
    GameConfig,
    WallConfig,
)
from mettagrid.mettagrid_c import MettaGrid


def test_exception_when_laser_not_in_inventory():
    """Test that an exception is raised when attack requires laser but it's not in resource_names."""
    game_map = [
        ["wall", "wall", "wall", "wall", "wall"],
        ["wall", "agent.red", ".", "agent.blue", "wall"],
        ["wall", "wall", "wall", "wall", "wall"],
    ]

    game_config = GameConfig(
        max_steps=50,
        num_agents=2,
        obs_width=11,
        obs_height=11,
        num_observation_tokens=200,
        # Note: laser is NOT in resource_names
        resource_names=["armor", "heart"],
        actions=ActionsConfig(
            noop=ActionConfig(enabled=True),
            move=ActionConfig(enabled=True),
            attack=AttackActionConfig(
                enabled=True,
                consumed_resources={"laser": 1},  # This should trigger an exception!
                defense_resources={"armor": 1},
            ),
            put_items=ActionConfig(enabled=True),
            get_items=ActionConfig(enabled=True),
            swap=ActionConfig(enabled=True),
            change_glyph=ChangeGlyphActionConfig(enabled=False, number_of_glyphs=4),
        ),
        objects={"wall": WallConfig(type_id=1)},
        agent=AgentConfig(default_resource_limit=10, freeze_duration=5, rewards=AgentRewards()),
        agents=[
            AgentConfig(team_id=0, default_resource_limit=10, freeze_duration=5),  # red
            AgentConfig(team_id=1, default_resource_limit=10, freeze_duration=5),  # blue
        ],
    )

    # Check that creating the environment raises an exception
    with pytest.raises(ValueError) as exc_info:
        # This should trigger an exception about missing laser resource
        MettaGrid(from_mettagrid_config(game_config), game_map, 42)

    # Check the exception message
    assert "attack" in str(exc_info.value)
    assert "laser" in str(exc_info.value)
    assert "not in resource_names" in str(exc_info.value)
    print(f"✓ Got expected exception: {exc_info.value}")


def test_no_exception_when_resources_in_inventory():
    """Test that no exception is raised when all consumed resources are in inventory."""
    game_map = [
        ["wall", "wall", "wall", "wall", "wall"],
        ["wall", "agent.red", ".", "agent.blue", "wall"],
        ["wall", "wall", "wall", "wall", "wall"],
    ]

    game_config = GameConfig(
        max_steps=50,
        num_agents=2,
        obs_width=11,
        obs_height=11,
        num_observation_tokens=200,
        # Laser IS in resource_names
        resource_names=["laser", "armor", "heart"],
        actions=ActionsConfig(
            noop=ActionConfig(enabled=True),
            move=ActionConfig(enabled=True),
            attack=AttackActionConfig(
                enabled=True,
                consumed_resources={"laser": 1},
                defense_resources={"armor": 1},
            ),
            put_items=ActionConfig(enabled=True),
            get_items=ActionConfig(enabled=True),
            swap=ActionConfig(enabled=True),
            change_glyph=ChangeGlyphActionConfig(enabled=False, number_of_glyphs=4),
        ),
        objects={"wall": WallConfig(type_id=1)},
        agent=AgentConfig(default_resource_limit=10, freeze_duration=5, rewards=AgentRewards()),
        agents=[
            AgentConfig(team_id=0, default_resource_limit=10, freeze_duration=5),  # red
            AgentConfig(team_id=1, default_resource_limit=10, freeze_duration=5),  # blue
        ],
    )

    # This should not raise an exception
    try:
        MettaGrid(from_mettagrid_config(game_config), game_map, 42)
        print("✓ No exception raised when all resources are in inventory")
    except ValueError as e:
        pytest.fail(f"Unexpected ValueError: {e}")


if __name__ == "__main__":
    test_exception_when_laser_not_in_inventory()
    test_no_exception_when_resources_in_inventory()
    print("\nAll tests passed!")
