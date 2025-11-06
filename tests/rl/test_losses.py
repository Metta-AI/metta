"""Unit tests for the new loss infrastructure."""

import types

import tensordict
import torch
import torchrl.data

import metta.agent.policy
import metta.rl.loss.loss
import mettagrid.policy.policy_env_interface


class DummyPolicy(metta.agent.policy.Policy):
    """Minimal policy implementation for exercising loss utilities."""

    def __init__(self, policy_env_info: mettagrid.policy.policy_env_interface.PolicyEnvInterface | None = None) -> None:
        if policy_env_info is None:
            import mettagrid.config.mettagrid_config

            policy_env_info = mettagrid.policy.policy_env_interface.PolicyEnvInterface.from_mg_cfg(
                mettagrid.config.mettagrid_config.MettaGridConfig()
            )
        super().__init__(policy_env_info)
        self._linear = torch.nn.Linear(1, 1)

    def forward(self, td: tensordict.TensorDict, action: torch.Tensor | None = None) -> tensordict.TensorDict:  # noqa: D401
        td = td.clone(False)
        td["values"] = torch.zeros(td.batch_size.numel(), dtype=torch.float32)
        return td

    def get_agent_experience_spec(self) -> torchrl.data.Composite:  # noqa: D401
        return torchrl.data.Composite(values=torchrl.data.UnboundedDiscrete(shape=torch.Size([]), dtype=torch.float32))

    def initialize_to_environment(self, policy_env_info, device: torch.device) -> None:  # noqa: D401
        return None

    @property
    def device(self) -> torch.device:  # noqa: D401
        return torch.device("cpu")

    @property
    def total_params(self) -> int:  # noqa: D401
        return sum(param.numel() for param in self.parameters())

    def reset_memory(self) -> None:  # noqa: D401
        return None

    def clip_weights(self) -> None:
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0, error_if_nonfinite=False)


class DummyLoss(metta.rl.loss.loss.Loss):
    """Loss subclass exposing the base-class helpers for testing."""

    def __init__(self) -> None:
        policy = DummyPolicy()
        trainer_cfg = types.SimpleNamespace()
        env = types.SimpleNamespace()
        loss_cfg = types.SimpleNamespace()
        super().__init__(policy, trainer_cfg, env, torch.device("cpu"), "dummy", loss_cfg)


def test_loss_stats_average_values() -> None:
    loss = DummyLoss()
    loss.loss_tracker["policy_loss"].extend([1.0, 3.0])
    loss.loss_tracker["value_loss"].extend([2.0, 4.0, 6.0])

    stats = loss.stats()

    assert stats["policy_loss"] == 2.0
    assert stats["value_loss"] == 4.0


def test_zero_loss_tracker_clears_values() -> None:
    loss = DummyLoss()
    loss.loss_tracker["entropy"].extend([0.1, 0.2])

    loss.zero_loss_tracker()

    assert all(len(values) == 0 for values in loss.loss_tracker.values())
