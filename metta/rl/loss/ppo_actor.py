from typing import Any, Optional

import numpy as np
import torch
from pydantic import Field
from tensordict import TensorDict
from torch import Tensor
from torchrl.data import Composite, UnboundedContinuous

from metta.agent.policy import Policy
from metta.rl.advantage import normalize_advantage_distributed
from metta.rl.loss.loss import Loss, LossConfig
from metta.rl.training import ComponentContext, TrainingEnvironment


class PPOActorConfig(LossConfig):
    # PPO hyperparameters
    # Clip coefficient (0.1-0.3 typical; Schulman et al. 2017)
    clip_coef: float = Field(default=0.22017136216163635, gt=0, le=1.0)
    # Entropy term weight from sweep
    ent_coef: float = Field(default=0.05000, ge=0)

    # Normalization and clipping
    # Advantage normalization toggle
    norm_adv: bool = True
    # Target KL for early stopping (None disables)
    target_kl: float | None = None

    def create(
        self,
        policy: Policy,
        trainer_cfg: Any,
        env: TrainingEnvironment,
        device: torch.device,
        instance_name: str,
    ) -> "PPOActor":
        return PPOActor(policy, trainer_cfg, env, device, instance_name, self)


class PPOActor(Loss):
    """PPO actor loss."""

    __slots__ = ()

    def __init__(
        self,
        policy: Policy,
        trainer_cfg: Any,
        env: TrainingEnvironment,
        device: torch.device,
        instance_name: str,
        cfg: "PPOActorConfig",
    ):
        super().__init__(policy, trainer_cfg, env, device, instance_name, cfg)

    def get_experience_spec(self) -> Composite:
        return Composite(act_log_prob=UnboundedContinuous(shape=torch.Size([]), dtype=torch.float32))

    def policy_output_keys(self, policy_td: Optional[TensorDict] = None) -> set[str]:
        return {"act_log_prob", "entropy"}

    def run_train(
        self, shared_loss_data: TensorDict, context: ComponentContext, mb_idx: int
    ) -> tuple[Tensor, TensorDict, bool]:
        stop_update_epoch = False
        if mb_idx > 0 and self.cfg.target_kl is not None:
            avg_kl = np.mean(self.loss_tracker["approx_kl"]) if self.loss_tracker["approx_kl"] else 0.0
            if avg_kl > self.cfg.target_kl:
                stop_update_epoch = True

        cfg = self.cfg

        minibatch = shared_loss_data["sampled_mb"]
        if minibatch.batch_size.numel() == 0:  # early exit if minibatch is empty
            return self._zero_tensor, shared_loss_data, False

        policy_td = shared_loss_data["policy_td"]
        old_logprob = minibatch["act_log_prob"]
        new_logprob = policy_td["act_log_prob"].reshape(old_logprob.shape)
        entropy = policy_td["entropy"]

        importance_sampling_ratio = shared_loss_data.get("importance_sampling_ratio", None)
        if importance_sampling_ratio is None:
            logratio = torch.clamp(new_logprob - old_logprob, -10, 10)
            importance_sampling_ratio = logratio.exp()

        update_td = TensorDict(
            {
                "ratio": importance_sampling_ratio.detach(),
            },
            batch_size=minibatch.batch_size,
        )
        indices = shared_loss_data["indices"][:, 0]
        self.replay.update(indices, update_td)

        adv = shared_loss_data.get("advantages_pg", None)
        if adv is None:
            adv = shared_loss_data["advantages"]
        adv = adv.detach()

        # Normalize advantages with distributed support, then apply prioritized weights
        adv = normalize_advantage_distributed(adv, cfg.norm_adv)
        prio_weights = shared_loss_data["prio_weights"]
        adv = prio_weights * adv

        pg_loss1 = -adv * importance_sampling_ratio
        pg_loss2 = -adv * torch.clamp(
            importance_sampling_ratio,
            1 - self.cfg.clip_coef,
            1 + self.cfg.clip_coef,
        )
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        entropy_loss = entropy.mean()

        loss = pg_loss - cfg.ent_coef * entropy_loss

        # Compute metrics
        with torch.no_grad():
            logratio = new_logprob - minibatch["act_log_prob"]
            approx_kl = ((importance_sampling_ratio - 1) - logratio).mean()
            clipfrac = ((importance_sampling_ratio - 1.0).abs() > self.cfg.clip_coef).float().mean()

        # Update loss tracking
        self.loss_tracker["policy_loss"].append(float(pg_loss.item()))
        self.loss_tracker["entropy"].append(float(entropy_loss.item()))
        self.loss_tracker["approx_kl"].append(float(approx_kl.item()))
        self.loss_tracker["clipfrac"].append(float(clipfrac.item()))
        self.loss_tracker["importance"].append(float(importance_sampling_ratio.mean().item()))
        self.loss_tracker["current_logprobs"].append(float(new_logprob.mean().item()))

        return loss, shared_loss_data, stop_update_epoch
