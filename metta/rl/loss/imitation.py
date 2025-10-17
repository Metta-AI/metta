"""Imitation learning loss backed by behaviour cloning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from pydantic import Field, model_validator
from tensordict import TensorDict
from torch import Tensor
from torch.utils.data import Dataset

try:
    from gymnasium import spaces as gym_spaces
except ImportError:  # pragma: no cover - gym fallback if gymnasium unavailable
    from gym import spaces as gym_spaces  # type: ignore[assignment]

from metta.agent.policy import Policy
from metta.rl.loss.loss import Loss
from metta.rl.training import TrainingEnvironment
from mettagrid.base_config import Config


class _DemonstrationDataset(Dataset[TensorDict]):
    """Simple in-memory dataset backed by a torch-saved dictionary."""

    def __init__(self, path: Path, device: torch.device) -> None:
        super().__init__()
        if not path.exists():
            raise FileNotFoundError(f"Imitation dataset not found at {path}")

        loaded = torch.load(path)
        if not isinstance(loaded, dict):
            raise ValueError("Imitation dataset must be a torch-saved dict containing 'env_obs' and 'actions'")

        missing: set[str] = {"env_obs", "actions"} - set(loaded.keys())
        if missing:
            raise KeyError(f"Imitation dataset missing required keys: {sorted(missing)}")

        self._device = device
        self._obs = loaded["env_obs"]
        self._actions = loaded["actions"]

        if not isinstance(self._obs, torch.Tensor) or not isinstance(self._actions, torch.Tensor):
            raise TypeError("Dataset tensors must be torch.Tensor instances")
        if self._obs.shape[0] != self._actions.shape[0]:
            raise ValueError("Observation and action tensors must share leading dimension")

    def __len__(self) -> int:
        return self._obs.shape[0]

    def __getitem__(self, index: int) -> TensorDict:
        obs = self._obs[index]
        act = self._actions[index]
        td = TensorDict(
            {
                "env_obs": obs,
                "actions": act,
            },
            batch_size=(),
        )
        return td

    def sample(self, batch_size: int) -> TensorDict:
        """Sample a random batch and stage onto the configured device."""
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0 for imitation sampling")

        idx = torch.randint(0, len(self), (batch_size,))
        batch_obs = self._obs[idx].to(self._device)
        batch_actions = self._actions[idx].to(self._device)
        return TensorDict({"env_obs": batch_obs, "actions": batch_actions}, batch_size=(batch_size,))


class ImitationConfig(Config):
    """Configuration for imitation learning / behaviour cloning."""

    dataset_path: str = Field(description="Path to a torch.save'd dict with 'env_obs' and 'actions'.")
    batch_size: int = Field(default=256, gt=0, description="Mini batch size sampled from the demonstration dataset.")
    loss_coef: float = Field(default=1.0, ge=0.0, description="Weight applied to the imitation loss term.")
    clamp_action_space: bool = Field(
        default=True,
        description="Whether to clamp sampled actions to the environment action space bounds.",
    )

    @model_validator(mode="after")
    def _validate_path(self) -> "ImitationConfig":
        if not self.dataset_path:
            raise ValueError("dataset_path must be provided for imitation learning.")
        return self

    def create(
        self,
        policy: Policy,
        trainer_cfg: Any,
        env: TrainingEnvironment,
        device: torch.device,
        instance_name: str,
        loss_config: "ImitationConfig",
    ) -> "ImitationLearningLoss":
        return ImitationLearningLoss(
            policy=policy,
            trainer_cfg=trainer_cfg,
            env=env,
            device=device,
            instance_name=instance_name,
            loss_config=loss_config,
        )


@dataclass(slots=True)
class _ActionSpaceConstraints:
    """Utility encapsulating action-space-specific post-processing."""

    discrete_n: Optional[int] = None
    box_low: Optional[torch.Tensor] = None
    box_high: Optional[torch.Tensor] = None

    @classmethod
    def from_space(cls, space: gym_spaces.Space, device: torch.device) -> "_ActionSpaceConstraints":
        if isinstance(space, gym_spaces.Discrete):
            return cls(discrete_n=int(space.n))
        if isinstance(space, gym_spaces.Box):
            low = torch.as_tensor(space.low, dtype=torch.float32, device=device)
            high = torch.as_tensor(space.high, dtype=torch.float32, device=device)
            return cls(box_low=low, box_high=high)
        raise NotImplementedError(f"Unsupported action space for imitation learning: {space!r}")

    def clamp(self, actions: Tensor) -> Tensor:
        if self.discrete_n is not None:
            return actions.clamp_(0, self.discrete_n - 1).to(dtype=torch.long)
        if self.box_low is not None and self.box_high is not None:
            return torch.max(torch.min(actions, self.box_high), self.box_low)
        return actions


class ImitationLearningLoss(Loss):
    """Behaviour cloning loss optimising log-prob of demonstration actions."""

    def __init__(
        self,
        policy: Policy,
        trainer_cfg: Any,
        env: TrainingEnvironment,
        device: torch.device,
        instance_name: str,
        loss_config: ImitationConfig,
    ) -> None:
        super().__init__(
            policy=policy,
            trainer_cfg=trainer_cfg,
            env=env,
            device=device,
            instance_name=instance_name,
            loss_cfg=loss_config,
        )
        dataset_path = Path(loss_config.dataset_path)
        self._dataset = _DemonstrationDataset(dataset_path, device=device)
        self._constraints = _ActionSpaceConstraints.from_space(env.single_action_space, device)
        self._obs_dtype = torch.uint8 if getattr(env.single_observation_space, "dtype", None) == torch.uint8 else None

    def run_rollout(self, td: TensorDict, context: Any) -> None:  # pragma: no cover - imitation does not use rollouts
        return

    def run_train(
        self,
        shared_loss_data: TensorDict,
        context: Any,
        mb_idx: int,
    ) -> tuple[Tensor, TensorDict, bool]:
        del mb_idx  # not required for imitation updates
        self._ensure_context(context)

        batch = self._dataset.sample(self.loss_cfg.batch_size)
        obs = batch["env_obs"]
        if self._obs_dtype is not None:
            obs = obs.to(dtype=torch.uint8)
        else:
            obs = obs.to(dtype=torch.float32)

        actions = batch["actions"]
        if self.loss_cfg.clamp_action_space:
            actions = self._constraints.clamp(actions)

        policy_td = TensorDict(
            {
                "env_obs": obs,
                "batch": torch.full((obs.shape[0],), 1, dtype=torch.long, device=self.device),
                "bptt": torch.ones(obs.shape[0], dtype=torch.long, device=self.device),
            },
            batch_size=(obs.shape[0],),
        )

        action_tensor = actions
        if action_tensor.dim() == 1:
            action_tensor = action_tensor.unsqueeze(-1)

        outputs = self.policy.forward(policy_td, action=action_tensor)
        if "act_log_prob" not in outputs.keys():
            raise RuntimeError("Policy must produce 'act_log_prob' when actions are provided for imitation learning.")

        log_prob = outputs["act_log_prob"].view(-1)
        imitation_loss = -log_prob.mean() * self.loss_cfg.loss_coef

        self._track("imitation_loss", imitation_loss.detach())
        avg_logprob = log_prob.mean().detach()
        self._track("imitation_log_prob", avg_logprob)

        return imitation_loss, shared_loss_data, False

    def _track(self, key: str, value: Tensor) -> None:
        self.loss_tracker[key].append(float(value.item()))
