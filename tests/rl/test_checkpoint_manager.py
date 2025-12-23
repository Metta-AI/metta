"""Tests for CheckpointManager - the core checkpoint save/load flows used during training."""

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from pydantic import Field
from tensordict import TensorDict

import mettagrid.builder.envs as eb
from metta.agent.components.component_config import ComponentConfig
from metta.agent.mocks import MockAgent
from metta.agent.policy import PolicyArchitecture
from metta.rl.checkpoint_manager import CheckpointManager
from metta.rl.system_config import SystemConfig
from mettagrid.base_config import Config
from mettagrid.policy.checkpoint_policy import CheckpointPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.util.uri_resolvers.schemes import (
    get_checkpoint_metadata,
    policy_spec_from_uri,
    resolve_uri,
)


class MockActionComponentConfig(ComponentConfig):
    name: str = "mock"

    def make_component(self, env=None) -> nn.Module:
        return nn.Identity()


class MockAgentPolicyArchitecture(PolicyArchitecture):
    class_path: str = "metta.agent.mocks.mock_agent.MockAgent"
    action_probs_config: Config = Field(default_factory=MockActionComponentConfig)

    def make_policy(self, policy_env_info):
        return MockAgent()


@pytest.fixture
def test_system_cfg():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield SystemConfig(data_dir=Path(tmpdir), local_only=True)


@pytest.fixture
def checkpoint_manager(test_system_cfg):
    return CheckpointManager(run="test_run", system_cfg=test_system_cfg)


@pytest.fixture
def mock_agent():
    return MockAgent()


@pytest.fixture
def mock_policy_architecture():
    return MockAgentPolicyArchitecture()


class TestCheckpointManagerFlows:
    """Test the actual flows that happen during training and evaluation."""

    def test_get_latest_checkpoint_returns_highest_epoch(
        self, checkpoint_manager, mock_agent, mock_policy_architecture
    ):
        """During training resume, we need to find the latest checkpoint."""
        for epoch in [1, 5, 10]:
            CheckpointPolicy.write_checkpoint_dir(
                base_dir=checkpoint_manager.checkpoint_dir,
                run_name=checkpoint_manager.run_name,
                epoch=epoch,
                architecture=mock_policy_architecture,
                state_dict=mock_agent.state_dict(),
            )

        latest = checkpoint_manager.get_latest_checkpoint()
        assert latest is not None
        assert ":v10" in latest or "%3Av10" in latest

    def test_trainer_state_save_and_restore(self, checkpoint_manager, mock_agent, mock_policy_architecture):
        """Trainer state must be saved alongside policy for proper resume."""
        CheckpointPolicy.write_checkpoint_dir(
            base_dir=checkpoint_manager.checkpoint_dir,
            run_name=checkpoint_manager.run_name,
            epoch=5,
            architecture=mock_policy_architecture,
            state_dict=mock_agent.state_dict(),
        )

        mock_optimizer = torch.optim.Adam([torch.tensor(1.0)])
        checkpoint_manager.save_trainer_state(
            mock_optimizer, epoch=5, agent_step=1000, stopwatch_state={"elapsed_time": 123.45}
        )

        loaded = checkpoint_manager.load_trainer_state()
        assert loaded is not None
        assert loaded["epoch"] == 5
        assert loaded["agent_step"] == 1000
        assert loaded["stopwatch_state"]["elapsed_time"] == 123.45
        assert "optimizer_state" in loaded

    def test_resolve_latest_uri(self, checkpoint_manager, mock_agent, mock_policy_architecture):
        """The :latest suffix is used by eval tools to find the newest checkpoint."""
        for epoch in [1, 7, 3]:
            CheckpointPolicy.write_checkpoint_dir(
                base_dir=checkpoint_manager.checkpoint_dir,
                run_name=checkpoint_manager.run_name,
                epoch=epoch,
                architecture=mock_policy_architecture,
                state_dict=mock_agent.state_dict(),
            )

        latest_uri = f"file://{checkpoint_manager.checkpoint_dir}:latest"
        metadata = get_checkpoint_metadata(resolve_uri(latest_uri).canonical)
        assert metadata.epoch == 7

    def test_mpt_policy_loads_and_runs(self, checkpoint_manager, mock_agent, mock_policy_architecture):
        """Checkpoint policy wrapper must load checkpoint and produce actions."""
        CheckpointPolicy.write_checkpoint_dir(
            base_dir=checkpoint_manager.checkpoint_dir,
            run_name=checkpoint_manager.run_name,
            epoch=1,
            architecture=mock_policy_architecture,
            state_dict=mock_agent.state_dict(),
        )
        latest = checkpoint_manager.get_latest_checkpoint()
        assert latest is not None

        env_info = PolicyEnvInterface.from_mg_cfg(eb.make_navigation(num_agents=2))
        spec = policy_spec_from_uri(latest)
        policy = CheckpointPolicy.from_policy_spec(env_info, spec).wrapped_policy

        obs_shape = env_info.observation_space.shape
        env_obs = torch.zeros((env_info.num_agents, *obs_shape), dtype=torch.uint8)
        td = TensorDict({"env_obs": env_obs}, batch_size=[env_info.num_agents])
        result = policy(td.clone())
        assert "actions" in result


class TestCheckpointManagerValidation:
    def test_empty_directory_returns_none(self, checkpoint_manager):
        assert checkpoint_manager.load_trainer_state() is None
        assert checkpoint_manager.get_latest_checkpoint() is None

    def test_invalid_run_names_rejected(self, test_system_cfg):
        invalid_names = ["", "name with spaces", "name/with/slash"]
        for name in invalid_names:
            with pytest.raises(ValueError):
                CheckpointManager(run=name, system_cfg=test_system_cfg)
