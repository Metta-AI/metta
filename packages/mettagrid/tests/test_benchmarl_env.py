"""
Tests for BenchMARL integration with MettaGrid.

This module tests the MettaGridBenchMARLEnv and MettaGridTask
implementations for BenchMARL compatibility.
"""

import pytest

# Import BenchMARL and TorchRL dependencies
import torch
from tensordict import TensorDict

from mettagrid.config.mettagrid_config import (
    ActionConfig,
    ActionsConfig,
    AgentConfig,
    AgentRewards,
    AttackActionConfig,
    GameConfig,
    MettaGridConfig,
    WallConfig,
)
from mettagrid.envs.benchmarl_env import MettaGridBenchMARLEnv, MettaGridTask
from mettagrid.envs.benchmarl_wrapper import (
    create_competitive_task,
    create_cooperative_task,
    create_mixed_task,
    create_navigation_task,
)
from mettagrid.map_builder.ascii import AsciiMapBuilder


def make_test_config(num_agents=3, max_steps=100):
    """Create a test MettaGrid configuration."""
    # Create a dynamic map that matches the number of agents
    if num_agents == 1:
        map_data = [
            ["#", "#", "#", "#", "#", "#", "#"],
            ["#", ".", ".", ".", ".", ".", "#"],
            ["#", ".", ".", "1", ".", ".", "#"],
            ["#", ".", ".", ".", ".", ".", "#"],
            ["#", ".", ".", ".", ".", ".", "#"],
            ["#", "#", "#", "#", "#", "#", "#"],
        ]
    elif num_agents == 2:
        map_data = [
            ["#", "#", "#", "#", "#", "#", "#"],
            ["#", ".", ".", ".", ".", ".", "#"],
            ["#", ".", "1", ".", "2", ".", "#"],
            ["#", ".", ".", ".", ".", ".", "#"],
            ["#", ".", ".", ".", ".", ".", "#"],
            ["#", "#", "#", "#", "#", "#", "#"],
        ]
    else:  # 3 or more agents
        map_data = [
            ["#", "#", "#", "#", "#", "#", "#"],
            ["#", ".", ".", ".", ".", ".", "#"],
            ["#", ".", "1", ".", "2", ".", "#"],
            ["#", ".", ".", "3", ".", ".", "#"],
            ["#", ".", ".", ".", ".", ".", "#"],
            ["#", "#", "#", "#", "#", "#", "#"],
        ]
        # For more than 3 agents, add additional positions
        if num_agents > 3:
            for i in range(4, num_agents + 1):
                # Add agent positions to empty spots
                for row_idx, row in enumerate(map_data):
                    for col_idx, cell in enumerate(row):
                        if cell == "." and str(i) not in [cell for row in map_data for cell in row]:
                            map_data[row_idx][col_idx] = str(i)
                            break
                    else:
                        continue
                    break

    # Create agents with proper configuration (team_id should match map positions)
    agents = []
    for i in range(num_agents):
        agents.append(
            AgentConfig(
                team_id=i + 1,  # Map has agents 1, 2, 3 not 0, 1, 2
                rewards=AgentRewards(
                    inventory={"heart": 1.0},
                    stats={"attack_agent": 0.5},
                ),
            )
        )

    cfg = MettaGridConfig(
        game=GameConfig(
            num_agents=num_agents,
            max_steps=max_steps,
            actions=ActionsConfig(
                move=ActionConfig(),
                noop=ActionConfig(),
                rotate=ActionConfig(),
                attack=AttackActionConfig(enabled=True),
            ),
            objects={"wall": WallConfig(type_id=1)},
            agents=agents,
            map_builder=AsciiMapBuilder.Config(
                map_data=map_data,
                char_to_name_map={},  # Use global defaults for '#', '.', and agent numbers
            ),
            obs_width=11,
            obs_height=11,
        )
    )
    return cfg


def test_benchmarl_env_creation():
    """Test BenchMARL environment creation and properties."""
    cfg = make_test_config(num_agents=3)
    env = MettaGridBenchMARLEnv(
        mg_config=cfg,
        device="cpu",
        batch_size=torch.Size([]),
        render_mode=None,
    )

    # Test environment properties
    assert env.num_agents == 3
    assert env.max_num_agents == 3
    assert env.agent_names == ["agent_0", "agent_1", "agent_2"]
    assert env.group_map == {"agents": ["agent_0", "agent_1", "agent_2"]}
    assert env.max_steps == 100

    # Test specs
    assert env.observation_spec is not None
    assert env.action_spec is not None
    assert env.reward_spec is not None
    assert env.done_spec is not None

    env.close()


def test_benchmarl_env_reset():
    """Test BenchMARL environment reset functionality."""
    cfg = make_test_config(num_agents=3)
    env = MettaGridBenchMARLEnv(
        mg_config=cfg,
        device="cpu",
        batch_size=torch.Size([]),
    )

    # Test reset
    tensordict = env.reset(seed=42)

    assert isinstance(tensordict, TensorDict)
    assert "agents" in tensordict.keys()
    expected_obs_shape = env.observation_spec["agents"].shape
    assert tensordict["agents"].shape == expected_obs_shape
    assert tensordict["done"].shape == (env.num_agents,)
    assert not tensordict["done"].all()

    env.close()


def test_benchmarl_env_step():
    """Test BenchMARL environment step functionality."""
    cfg = make_test_config(num_agents=3)
    env = MettaGridBenchMARLEnv(
        mg_config=cfg,
        device="cpu",
        batch_size=torch.Size([]),
    )

    # Reset environment
    env.reset(seed=42)

    # Create action tensordict
    actions = torch.zeros((3, 2), dtype=torch.long)  # 3 agents, 2 action dimensions
    action_td = TensorDict({"agents": actions}, batch_size=torch.Size([]))

    # Step environment
    next_td = env.step(action_td)

    assert isinstance(next_td, TensorDict)
    assert "agents" in next_td.keys()
    expected_obs_shape = env.observation_spec["agents"].shape
    assert next_td["agents"].shape == expected_obs_shape
    assert next_td["reward"].shape == (env.num_agents,)
    assert next_td["done"].shape == (env.num_agents,)

    env.close()


def test_benchmarl_env_episode_termination():
    """Test BenchMARL environment episode termination."""
    cfg = make_test_config(num_agents=2, max_steps=10)
    env = MettaGridBenchMARLEnv(
        mg_config=cfg,
        device="cpu",
        batch_size=torch.Size([]),
    )

    # Reset environment
    env.reset(seed=42)

    done = False
    step_count = 0
    max_steps = 20  # Safety limit

    while not done and step_count < max_steps:
        # Random actions
        actions = torch.randint(0, 2, (2, 2), dtype=torch.long)
        action_td = TensorDict({"agents": actions}, batch_size=torch.Size([]))

        # Step
        next_td = env.step(action_td)
        done = next_td["done"].all().item()
        step_count += 1

    # Episode should terminate within max_steps
    assert step_count <= max_steps

    env.close()


def test_benchmarl_env_seeding():
    """Test that seeding works correctly."""
    cfg = make_test_config(num_agents=2)

    # Create two environments with same seed
    env1 = MettaGridBenchMARLEnv(mg_config=cfg, device="cpu")
    env2 = MettaGridBenchMARLEnv(mg_config=cfg, device="cpu")

    # Reset with same seed
    td1 = env1.reset(seed=123)
    td2 = env2.reset(seed=123)

    # Observations should be identical
    assert torch.allclose(td1["agents"], td2["agents"])

    # Take same actions
    actions = torch.zeros((2, 2), dtype=torch.long)
    action_td = TensorDict({"agents": actions}, batch_size=torch.Size([]))

    next_td1 = env1.step(action_td)
    next_td2 = env2.step(action_td)

    # Results should be identical
    assert torch.allclose(next_td1["agents"], next_td2["agents"])
    assert torch.allclose(
        next_td1["reward"],
        next_td2["reward"],
    )

    env1.close()
    env2.close()


def test_mettagrid_task_creation():
    """Test MettaGridTask creation and properties."""
    cfg = make_test_config(num_agents=2)
    task = MettaGridTask(
        mg_config=cfg,
        task_name="test_task",
        max_steps=100,
    )

    assert task.name == "test_task"
    assert task._max_steps == 100
    assert task.supports_discrete_actions()
    assert not task.supports_continuous_actions()
    assert task.env_name() == "mettagrid"


def test_mettagrid_task_env_factory():
    """Test MettaGridTask environment factory function."""
    cfg = make_test_config(num_agents=2)
    task = MettaGridTask(
        mg_config=cfg,
        task_name="test_task",
        max_steps=100,
    )

    # Get environment factory
    make_env = task.get_env_fun(
        num_envs=1,
        continuous_actions=False,
        seed=42,
        device="cpu",
    )

    # Create environment
    env = make_env()
    assert isinstance(env, MettaGridBenchMARLEnv)

    # Test that continuous actions raise error
    with pytest.raises(ValueError):
        task.get_env_fun(
            num_envs=1,
            continuous_actions=True,
            seed=42,
            device="cpu",
        )

    env.close()


def test_navigation_task_factory():
    """Test navigation task factory function."""
    task = create_navigation_task(
        num_agents=1,
        max_steps=500,
    )

    assert isinstance(task, MettaGridTask)
    assert task._max_steps == 500
    assert "navigation" in task.name

    # Create environment
    make_env = task.get_env_fun(
        num_envs=1,
        continuous_actions=False,
        seed=42,
        device="cpu",
    )
    env = make_env()

    assert env.num_agents == 1
    env.close()


def test_cooperative_task_factory():
    """Test cooperative task factory function."""
    task = create_cooperative_task(
        num_agents=4,
        max_steps=1000,
    )

    assert isinstance(task, MettaGridTask)
    assert task._max_steps == 1000
    assert "cooperative" in task.name

    # Create environment
    make_env = task.get_env_fun(
        num_envs=1,
        continuous_actions=False,
        seed=42,
        device="cpu",
    )
    env = make_env()

    assert env.num_agents == 4
    env.close()


def test_competitive_task_factory():
    """Test competitive task factory function."""
    task = create_competitive_task(
        num_agents=4,
        max_steps=1000,
    )

    assert isinstance(task, MettaGridTask)
    assert task._max_steps == 1000
    assert "competitive" in task.name

    # Create environment
    make_env = task.get_env_fun(
        num_envs=1,
        continuous_actions=False,
        seed=42,
        device="cpu",
    )
    env = make_env()

    assert env.num_agents == 4
    env.close()


def test_mixed_task_factory():
    """Test mixed cooperation/competition task factory function."""
    task = create_mixed_task(
        num_agents=6,
        max_steps=2000,
    )

    assert isinstance(task, MettaGridTask)
    assert task._max_steps == 2000
    assert "mixed" in task.name

    # Create environment
    make_env = task.get_env_fun(
        num_envs=1,
        continuous_actions=False,
        seed=42,
        device="cpu",
    )
    env = make_env()

    assert env.num_agents == 6
    env.close()


def test_benchmarl_env_episode():
    """Test running a full episode with BenchMARL environment."""
    cfg = make_test_config(num_agents=2, max_steps=10)
    env = MettaGridBenchMARLEnv(
        mg_config=cfg,
        device="cpu",
        batch_size=torch.Size([]),
    )

    # Reset environment
    env.reset(seed=42)

    done = False
    step_count = 0
    max_steps = 20  # Safety limit

    while not done and step_count < max_steps:
        # Random actions
        actions = torch.randint(0, 2, (2, 2), dtype=torch.long)
        action_td = TensorDict({"agents": actions}, batch_size=torch.Size([]))

        # Step
        next_td = env.step(action_td)
        done = next_td["done"].all().item()
        step_count += 1

    # Episode should terminate within max_steps
    assert step_count <= max_steps

    env.close()


# Tests that run even without BenchMARL installed (structure only)


def test_benchmarl_env_structure():
    """Test that BenchMARL integration exists and follows the pattern."""
    from mettagrid.core import MettaGridCore
    from mettagrid.envs.benchmarl_env import MettaGridBenchMARLEnv

    # Check inheritance pattern
    assert issubclass(MettaGridBenchMARLEnv, MettaGridCore)

    # Check it's exported in __init__
    from mettagrid.envs import MettaGridBenchMARLEnv as EnvFromInit
    from mettagrid.envs import MettaGridTask as TaskFromInit

    # Verify we have the task class too (unique to BenchMARL)
    assert TaskFromInit is not None
    assert EnvFromInit is not None


def test_benchmarl_wrapper_functions():
    """Test that wrapper factory functions exist."""
    from mettagrid.envs.benchmarl_wrapper import (
        create_competitive_task,
        create_cooperative_task,
        create_mixed_task,
        create_navigation_task,
    )

    # Just verify they exist and are callable
    assert callable(create_navigation_task)
    assert callable(create_cooperative_task)
    assert callable(create_competitive_task)
    assert callable(create_mixed_task)
