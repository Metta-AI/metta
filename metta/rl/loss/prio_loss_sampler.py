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
        self.all_prio_is_weights = None
        self.anneal_beta = 0.0

        if hasattr(self.policy, "burn_in_steps"):
            self.burn_in_steps = self.policy.burn_in_steps
        else:
            self.burn_in_steps = 0
        self.burn_in_steps_iter = 0

    def run_rollout(self, td: TensorDict, context: ComponentContext) -> None:
        """Rollout step: forward policy and store experience with optional burn-in."""
        with torch.no_grad():
            self.policy.forward(td)

        if self.burn_in_steps_iter < self.burn_in_steps:
            self.burn_in_steps_iter += 1
            return

        env_slice = context.training_env_id
        if env_slice is None:
            raise RuntimeError("ComponentContext.training_env_id is missing in rollout.")
        self.replay.store(data_td=td, env_id=env_slice)

    def run_train(
        self,
        shared_loss_data: TensorDict,
        context: ComponentContext,
        mb_idx: int,
    ) -> tuple[Tensor, TensorDict, bool]:
        if mb_idx == 0:
            # precompute what we can upfront on the first mb. Then index as we need based on sampling.
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
                    1.0,  # v-trace is used in PPO actor instead. 1.0 means no v-trace
                    1.0,  # v-trace is used in PPO actor instead. 1.0 means no v-trace
                    self.device,
                )

            self.adv_magnitude = self.advantages.abs().sum(dim=1)
            self.prio_weights = torch.nan_to_num(
                self.adv_magnitude**cfg.prioritized_experience_replay.prio_alpha, 0, 0, 0
            )
            self.prio_probs = (self.prio_weights + 1e-6) / (self.prio_weights.sum() + 1e-6)
            self.all_prio_is_weights = (self.replay.segments * self.prio_probs) ** -self.anneal_beta

        # Sample segment indices
        idx = torch.multinomial(self.prio_probs, self.replay.minibatch_segments)

        minibatch = self.replay.buffer[idx].clone()

        with torch.no_grad():
            minibatch["advantages"] = self.advantages[idx]
            minibatch["returns"] = self.advantages[idx] + minibatch["values"]
            prio_weights = self.all_prio_is_weights[idx, None]

        shared_loss_data["sampled_mb"] = minibatch  # one loss should write the sampled mb for others to use
        shared_loss_data["indices"] = NonTensorData(idx)  # av this breaks compile
        shared_loss_data["prio_weights"] = prio_weights

        # Then forward the policy using the sampled minibatch
        policy_td = minibatch.select(*self.policy_experience_spec.keys(include_nested=True))
        B, TT = policy_td.batch_size
        policy_td = policy_td.reshape(B * TT)
        policy_td.set("bptt", torch.full((B * TT,), TT, device=policy_td.device, dtype=torch.long))
        policy_td.set("batch", torch.full((B * TT,), B, device=policy_td.device, dtype=torch.long))

        flat_actions = minibatch["actions"].reshape(B * TT, -1)

        policy_td = self.policy.forward(policy_td, action=flat_actions)
        shared_loss_data["policy_td"] = policy_td.reshape(B, TT)

        return torch.tensor(0.0, dtype=torch.float32, device=self.device), shared_loss_data, False
