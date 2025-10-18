from types import SimpleNamespace
from typing import cast

import pytest
import torch
from tensordict import TensorDict  # type: ignore[import-untyped]
from torchrl.data import Composite  # type: ignore[import-untyped]

from metta.agent.policy import Policy
from metta.rl.loss.future_latent_ema import FutureLatentEMALoss, FutureLatentEMALossConfig
from metta.rl.loss.loss_config import LossConfig
from metta.rl.training import ComponentContext, TrainingEnvironment


class _StubPolicy(Policy):
    """Minimal policy implementation for exercising the future latent EMA loss."""

    def __init__(self) -> None:
        super().__init__()
        self._device = torch.device("cpu")

    def forward(self, td: TensorDict, action: torch.Tensor | None = None) -> TensorDict:  # noqa: D401
        return td

    def get_agent_experience_spec(self) -> Composite:  # noqa: D401
        return Composite()

    def initialize_to_environment(self, env_metadata, device: torch.device) -> None:  # noqa: D401
        return None

    @property
    def device(self) -> torch.device:  # noqa: D401
        return self._device

    def reset_memory(self) -> None:  # noqa: D401
        return None


def _build_loss(decay: float = 0.5, horizon: int = 2) -> FutureLatentEMALoss:
    policy = _StubPolicy()
    trainer_cfg = SimpleNamespace()
    vec_env = cast(TrainingEnvironment, SimpleNamespace())
    cfg = FutureLatentEMALossConfig(ema_decay=decay, prediction_horizon=horizon, loss_coef=1.0)
    loss = FutureLatentEMALoss(policy, trainer_cfg, vec_env, torch.device("cpu"), "future_latent_ema", cfg)
    return loss


def test_future_latent_ema_loss_matches_manual_target() -> None:
    loss = _build_loss(decay=0.5, horizon=2)

    core = torch.tensor(
        [
            [
                [1.0, 0.0],
                [0.5, 1.0],
                [0.25, 0.5],
                [0.0, 0.25],
            ]
        ],
        dtype=torch.float32,
    )
    # future_latent_pred will be overwritten with the computed target so the loss should be zero.
    policy_td = TensorDict(
        {
            "core": core.clone(),
            "future_latent_pred": torch.zeros_like(core),
        },
        batch_size=[1, 4],
    )
    shared_loss_data = TensorDict({"policy_td": policy_td}, batch_size=[1, 4])

    # Manually compute EMA target for comparison.
    decay = 0.5
    horizon = 2
    normalisation = 1.0 - decay**horizon
    manual_target = torch.zeros(1, core.size(1) - horizon, core.size(2))
    for offset in range(core.size(1) - horizon):
        manual_target[:, offset, :] = (
            (1 - decay) * core[:, offset + 1, :] + (1 - decay) * decay * core[:, offset + 2, :]
        ) / normalisation

    policy_td["future_latent_pred"][:, : manual_target.size(1), :] = manual_target

    context = cast(ComponentContext, SimpleNamespace())
    computed_loss, _, _ = loss.run_train(shared_loss_data, context, mb_idx=0)

    assert computed_loss.item() == pytest.approx(0.0)
    assert loss.loss_tracker is not None
    assert loss.loss_tracker["future_latent_ema_mse"][-1] == pytest.approx(0.0)


def test_future_latent_ema_loss_skips_when_predictions_missing() -> None:
    loss = _build_loss()
    policy_td = TensorDict(
        {
            "core": torch.ones(1, 3, 2),
        },
        batch_size=[1, 3],
    )
    shared_loss_data = TensorDict({"policy_td": policy_td}, batch_size=[1, 3])

    context = cast(ComponentContext, SimpleNamespace())
    computed_loss, _, _ = loss.run_train(shared_loss_data, context, mb_idx=0)

    assert computed_loss.item() == pytest.approx(0.0)
    assert loss.loss_tracker is not None
    assert not loss.loss_tracker["future_latent_ema_mse"]


def test_loss_config_enables_future_latent_ema_by_flag() -> None:
    cfg = LossConfig(enable_future_latent_ema=True)
    assert "future_latent_ema" in cfg.loss_configs
