"""Test that Policy correctly implements TrainablePolicy interface."""

import torch
from tensordict import TensorDict

from metta.agent.policy import Policy
from mettagrid.config.id_map import ObservationFeatureSpec
from mettagrid.config.mettagrid_config import ActionsConfig
from mettagrid.policy.policy import AgentPolicy, TrainablePolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Action, AgentObservation, ObservationToken


class SimplePolicy(Policy):
    """Minimal concrete policy for testing."""

    def __init__(self, actions: ActionsConfig, policy_env_info: PolicyEnvInterface | None = None):
        if policy_env_info is None:
            # Create a minimal PolicyEnvInterface for testing
            from mettagrid.config.mettagrid_config import MettaGridConfig

            mg_cfg = MettaGridConfig()
            policy_env_info = PolicyEnvInterface.from_mg_cfg(mg_cfg)
        super().__init__(policy_env_info)
        self.linear = torch.nn.Linear(10, 5)
        self._device = torch.device("cpu")

    def forward(self, td: TensorDict, action: torch.Tensor | None = None) -> TensorDict:
        td["actions"] = torch.zeros(td.batch_size[0], dtype=torch.long)
        return td

    def reset_memory(self):
        pass

    @property
    def device(self) -> torch.device:
        return self._device


def test_policy_inherits_from_trainable_policy():
    """Verify Policy is a subclass of TrainablePolicy."""
    assert issubclass(Policy, TrainablePolicy)


def test_policy_implements_network_method():
    """Verify Policy implements network() method that returns self."""
    actions = ActionsConfig()
    from mettagrid.config.mettagrid_config import MettaGridConfig

    policy_env_info = PolicyEnvInterface.from_mg_cfg(MettaGridConfig())
    policy = SimplePolicy(actions, policy_env_info)

    network = policy.network()
    assert network is policy
    assert isinstance(network, torch.nn.Module)


def test_policy_implements_agent_policy_method():
    """Verify Policy implements agent_policy() method."""
    actions = ActionsConfig()
    from mettagrid.config.mettagrid_config import MettaGridConfig

    policy_env_info = PolicyEnvInterface.from_mg_cfg(MettaGridConfig())
    policy = SimplePolicy(actions, policy_env_info)

    agent_policy = policy.agent_policy(agent_id=0)
    assert isinstance(agent_policy, AgentPolicy)


def test_agent_policy_adapter_step():
    """Verify the AgentPolicy adapter can step."""
    actions = ActionsConfig()
    from mettagrid.config.mettagrid_config import MettaGridConfig

    policy_env_info = PolicyEnvInterface.from_mg_cfg(MettaGridConfig())
    policy = SimplePolicy(actions, policy_env_info)
    agent_policy = policy.agent_policy(agent_id=0)

    # Create a simple observation (single agent, token-based)
    # Create mock feature spec for padding tokens
    feature = ObservationFeatureSpec(id=0, name="padding", normalization=255.0)
    tokens = [ObservationToken(feature=feature, location=(0, 0), value=0) for _ in range(10)]
    obs = AgentObservation(agent_id=0, tokens=tokens)

    # Get action
    action = agent_policy.step(obs)
    assert isinstance(action, Action)


def test_agent_policy_adapter_reset():
    """Verify the AgentPolicy adapter can reset."""
    actions = ActionsConfig()
    from mettagrid.config.mettagrid_config import MettaGridConfig

    policy_env_info = PolicyEnvInterface.from_mg_cfg(MettaGridConfig())
    policy = SimplePolicy(actions, policy_env_info)
    agent_policy = policy.agent_policy(agent_id=0)

    # Should not raise
    agent_policy.reset()


def test_policy_has_actions_config():
    """Verify Policy stores ActionsConfig."""
    actions = ActionsConfig()
    from mettagrid.config.mettagrid_config import MettaGridConfig

    mg_cfg = MettaGridConfig()
    mg_cfg.game.actions = actions
    policy_env_info = PolicyEnvInterface.from_mg_cfg(mg_cfg)
    policy = SimplePolicy(actions, policy_env_info)

    assert policy._actions is policy_env_info.actions
    assert policy._actions is actions


def test_policy_load_save_delegates_to_network():
    """Verify load/save use torch state dict methods."""
    import tempfile
    from pathlib import Path

    actions = ActionsConfig()
    from mettagrid.config.mettagrid_config import MettaGridConfig

    policy_env_info = PolicyEnvInterface.from_mg_cfg(MettaGridConfig())
    policy = SimplePolicy(actions, policy_env_info)

    # Save
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "policy.pt"
        policy.save_policy_data(str(path))

        # Load into new policy
        new_policy = SimplePolicy(actions, policy_env_info)
        new_policy.load_policy_data(str(path))

        # Verify weights match
        for p1, p2 in zip(policy.parameters(), new_policy.parameters(), strict=True):
            assert torch.allclose(p1, p2)
