"""Test that exceptions are raised when attack resources are not in inventory."""

import pytest

import mettagrid.config.mettagrid_c_config
import mettagrid.config.mettagrid_config
import mettagrid.mettagrid_c


def test_exception_when_laser_not_in_inventory():
    """Test that an exception is raised when attack requires laser but it's not in resource_names."""
    game_map = [
        ["wall", "wall", "wall", "wall", "wall"],
        ["wall", "agent.red", ".", "agent.blue", "wall"],
        ["wall", "wall", "wall", "wall", "wall"],
    ]

    game_config = mettagrid.config.mettagrid_config.GameConfig(
        max_steps=50,
        num_agents=2,
        obs=mettagrid.config.mettagrid_config.ObsConfig(width=11, height=11, num_tokens=200),
        # Note: laser is NOT in resource_names
        resource_names=["armor", "heart"],
        actions=mettagrid.config.mettagrid_config.ActionsConfig(
            noop=mettagrid.config.mettagrid_config.NoopActionConfig(enabled=True),
            move=mettagrid.config.mettagrid_config.MoveActionConfig(enabled=True),
            attack=mettagrid.config.mettagrid_config.AttackActionConfig(
                enabled=True,
                consumed_resources={"laser": 1},  # This should trigger an exception!
                defense_resources={"armor": 1},
            ),
            change_vibe=mettagrid.config.mettagrid_config.ChangeVibeActionConfig(enabled=False, number_of_vibes=4),
        ),
        objects={"wall": mettagrid.config.mettagrid_config.WallConfig()},
        agent=mettagrid.config.mettagrid_config.AgentConfig(
            default_resource_limit=10, freeze_duration=5, rewards=mettagrid.config.mettagrid_config.AgentRewards()
        ),
        agents=[
            mettagrid.config.mettagrid_config.AgentConfig(
                team_id=0, default_resource_limit=10, freeze_duration=5
            ),  # red
            mettagrid.config.mettagrid_config.AgentConfig(
                team_id=1, default_resource_limit=10, freeze_duration=5
            ),  # blue
        ],
    )

    # Check that creating the environment raises an exception
    with pytest.raises(ValueError) as exc_info:
        # This should trigger an exception about missing laser resource
        mettagrid.mettagrid_c.MettaGrid(
            mettagrid.config.mettagrid_c_config.convert_to_cpp_game_config(game_config), game_map, 42
        )

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

    game_config = mettagrid.config.mettagrid_config.GameConfig(
        max_steps=50,
        num_agents=2,
        obs=mettagrid.config.mettagrid_config.ObsConfig(width=11, height=11, num_tokens=200),
        # Laser IS in resource_names
        resource_names=["laser", "armor", "heart"],
        actions=mettagrid.config.mettagrid_config.ActionsConfig(
            noop=mettagrid.config.mettagrid_config.NoopActionConfig(enabled=True),
            move=mettagrid.config.mettagrid_config.MoveActionConfig(enabled=True),
            attack=mettagrid.config.mettagrid_config.AttackActionConfig(
                enabled=True, consumed_resources={"laser": 1}, defense_resources={"armor": 1}
            ),
            change_vibe=mettagrid.config.mettagrid_config.ChangeVibeActionConfig(enabled=False, number_of_vibes=4),
        ),
        objects={"wall": mettagrid.config.mettagrid_config.WallConfig()},
        agent=mettagrid.config.mettagrid_config.AgentConfig(
            default_resource_limit=10, freeze_duration=5, rewards=mettagrid.config.mettagrid_config.AgentRewards()
        ),
        agents=[
            mettagrid.config.mettagrid_config.AgentConfig(
                team_id=0, default_resource_limit=10, freeze_duration=5
            ),  # red
            mettagrid.config.mettagrid_config.AgentConfig(
                team_id=1, default_resource_limit=10, freeze_duration=5
            ),  # blue
        ],
    )

    # This should not raise an exception
    try:
        mettagrid.mettagrid_c.MettaGrid(
            mettagrid.config.mettagrid_c_config.convert_to_cpp_game_config(game_config), game_map, 42
        )
        print("✓ No exception raised when all resources are in inventory")
    except ValueError as e:
        pytest.fail(f"Unexpected ValueError: {e}")


if __name__ == "__main__":
    test_exception_when_laser_not_in_inventory()
    test_no_exception_when_resources_in_inventory()
    print("\nAll tests passed!")
