from typing import Any

import torch
from pydantic import Field
from tensordict import NonTensorData, TensorDict
from torch import Tensor

from metta.agent.policy import Policy
from metta.rl.advantage import compute_advantage
from metta.rl.loss.loss import Loss, LossConfig
from metta.rl.training import ComponentContext
from metta.utils.batch import calculate_prioritized_sampling_params


class PrioLossSamplerConfig(LossConfig):
    # Alpha=0 means uniform sampling; tuned via sweep
    prio_alpha: float = Field(default=0.0, ge=0, le=1.0)
    # Beta baseline per Schaul et al. (2016)
    prio_beta0: float = Field(default=0.6, ge=0, le=1.0)

    def create(
        self,
        policy: Policy,
        trainer_cfg: Any,
        vec_env: Any,
        device: torch.device,
        instance_name: str,
        loss_config: Any,
    ) -> "PrioLossSampler":
        """Create PrioLossSampler loss instance."""
        return PrioLossSampler(
            policy,
            trainer_cfg,
            vec_env,
            device,
            instance_name=instance_name,
            loss_config=loss_config,
        )


class PrioLossSampler(Loss):
    def __init__(
        self,
        policy: Policy,
        trainer_cfg: Any,
        vec_env: Any,
        device: torch.device,
        instance_name: str,
        loss_config: Any,
    ):
        super().__init__(policy, trainer_cfg, vec_env, device, instance_name, loss_config)
        self.advantages = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self.adv_magnitude = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self.prio_weights = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self.prio_probs = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self.anneal_beta = 0.0

    def run_train(
        self,
        shared_loss_data: TensorDict,
        context: ComponentContext,
        mb_idx: int,
    ) -> tuple[Tensor, TensorDict, bool]:

        if mb_idx == 0:
            if "ratio" in self.replay.buffer.keys():
                self.replay.buffer["ratio"].fill_(1.0)

            cfg = self.cfg
            with torch.no_grad():
                self.anneal_beta = calculate_prioritized_sampling_params(
                    epoch=context.epoch,
                    total_timesteps=self.trainer_cfg.total_timesteps,
                    batch_size=self.trainer_cfg.batch_size,
                    prio_alpha=cfg.prioritized_experience_replay.prio_alpha,
                    prio_beta0=cfg.prioritized_experience_replay.prio_beta0,
                )

                advantages = torch.zeros_like(self.replay.buffer["values"], device=self.device)
                self.advantages = compute_advantage(
                    self.replay.buffer["values"],
                    self.replay.buffer["rewards"],
                    self.replay.buffer["dones"],
                    torch.ones_like(self.replay.buffer["values"]),
                    advantages,
                    cfg.gamma,
                    cfg.gae_lambda,
                    cfg.vtrace.rho_clip,
                    cfg.vtrace.c_clip,
                    self.device,
                )

            self.adv_magnitude = self.advantages.abs().sum(dim=1)
            self.prio_weights = torch.nan_to_num(self.adv_magnitude**cfg.prioritized_experience_replay.prio_alpha, 0, 0, 0)
            self.prio_probs = (self.prio_weights + 1e-6) / (self.prio_weights.sum() + 1e-6)

        # Sample segment indices
        idx = torch.multinomial(self.prio_probs, self.replay.minibatch_segments)

        minibatch = self.replay.buffer[idx]

        with torch.no_grad():
            minibatch["advantages"] = advantages[idx]
            minibatch["returns"] = advantages[idx] + minibatch["values"]
            prio_weights = (self.replay.segments * self.prio_probs[idx, None]) ** -self.anneal_beta

        shared_loss_data["sampled_mb"] = minibatch  # one loss should write the sampled mb for others to use
        shared_loss_data["indices"] = NonTensorData(idx)  # av this breaks compile
        shared_loss_data["prio_weights"] = self.prio_weights[idx]

        

        return torch.tensor(0.0, dtype=torch.float32, device=self.device), shared_loss_data, False
