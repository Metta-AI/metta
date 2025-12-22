from typing import Any, Optional

import numpy as np
import torch
from pydantic import Field
from tensordict import TensorDict
from torch import Tensor
from torchrl.data import Composite, UnboundedContinuous, UnboundedDiscrete

from metta.agent.policy import Policy
from metta.rl.loss.loss import Loss, LossConfig
from metta.rl.training import ComponentContext, TrainingEnvironment


class PPOCriticConfig(LossConfig):
    vf_clip_coef: float = Field(default=0.1, ge=0)
    vf_coef: float = Field(default=0.753832, ge=0)
    # Value loss clipping toggle
    clip_vloss: bool = True

    def create(
        self,
        policy: Policy,
        trainer_cfg: Any,
        env: TrainingEnvironment,
        device: torch.device,
        instance_name: str,
    ) -> "PPOCritic":
        return PPOCritic(policy, trainer_cfg, env, device, instance_name, self)


class PPOCritic(Loss):
    """PPO value loss."""

    __slots__ = (
        "burn_in_steps",
        "burn_in_steps_iter",
    )

    def __init__(
        self,
        policy: Policy,
        trainer_cfg: Any,
        env: TrainingEnvironment,
        device: torch.device,
        instance_name: str,
        cfg: "PPOCriticConfig",
    ):
        super().__init__(policy, trainer_cfg, env, device, instance_name, cfg)

        if hasattr(self.policy, "burn_in_steps"):
            self.burn_in_steps = self.policy.burn_in_steps
        else:
            self.burn_in_steps = 0
        self.burn_in_steps_iter = 0

    def get_experience_spec(self) -> Composite:
        act_space = self.env.single_action_space
        act_dtype = torch.int32 if np.issubdtype(act_space.dtype, np.integer) else torch.float32
        scalar_f32 = UnboundedContinuous(shape=torch.Size([]), dtype=torch.float32)

        return Composite(
            actions=UnboundedDiscrete(shape=torch.Size([]), dtype=act_dtype),
            values=scalar_f32,
            rewards=scalar_f32,
            dones=scalar_f32,
            truncateds=scalar_f32,
        )

    def run_rollout(self, td: TensorDict, context: ComponentContext) -> None:
        with torch.no_grad():
            # If another loss already produced actions (e.g., sliced_cloner teacher slice),
            # reuse them to avoid overwriting while still computing values/logprobs.
            if "actions" in td.keys():
                self.policy.forward(td, action=td["actions"])
            else:
                self.policy.forward(td)

        if self.burn_in_steps_iter < self.burn_in_steps:
            self.burn_in_steps_iter += 1
            return

        env_slice = context.training_env_id
        if env_slice is None:
            raise RuntimeError("ComponentContext.training_env_id is missing in rollout.")
        self.replay.store(data_td=td, env_id=env_slice)

    def policy_output_keys(self, policy_td: Optional[TensorDict] = None) -> set[str]:
        return {"values"}

    def run_train(
        self, shared_loss_data: TensorDict, context: ComponentContext, mb_idx: int
    ) -> tuple[Tensor, TensorDict, bool]:
        # Sampling happens in the core loop; use the shared minibatch and indices.
        minibatch = shared_loss_data["sampled_mb"]

        if minibatch.batch_size.numel() == 0:  # early exit if minibatch is empty
            return self._zero_tensor, shared_loss_data, False

        # Advantages are computed in the core loop and passed through shared_loss_data.
        # Keep the full advantages around for explained variance logging.
        old_values = minibatch["values"]
        advantages_mb = shared_loss_data["advantages"]
        returns = advantages_mb + minibatch["values"]
        minibatch["returns"] = returns
        # Read policy forward results from the core loop (forward_policy_for_training).
        policy_td = shared_loss_data.get("policy_td", None)
        newvalue_reshaped = None
        if policy_td is not None:
            newvalue = policy_td["values"]
            newvalue_reshaped = newvalue.view(returns.shape)

        if newvalue_reshaped is not None:
            if self.cfg.clip_vloss:
                v_loss_unclipped = (newvalue_reshaped - returns) ** 2
                vf_clip_coef = self.cfg.vf_clip_coef
                v_clipped = old_values + torch.clamp(
                    newvalue_reshaped - old_values,
                    -vf_clip_coef,
                    vf_clip_coef,
                )
                v_loss_clipped = (v_clipped - returns) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
            else:
                v_loss = 0.5 * ((newvalue_reshaped - returns) ** 2).mean()
            # Update values in experience buffer
            update_td = TensorDict(
                {
                    "values": newvalue.view(minibatch["values"].shape).detach(),
                },
                batch_size=minibatch.batch_size,
            )
            indices = shared_loss_data["indices"][:, 0]
            self.replay.update(indices, update_td)
        else:
            v_loss = 0.5 * ((old_values - returns) ** 2).mean()
        # Scale value loss by coefficient
        v_loss = v_loss * self.cfg.vf_coef
        self.loss_tracker["value_loss"].append(float(v_loss.item()))

        return v_loss, shared_loss_data, False

    def on_train_phase_end(self, context: ComponentContext) -> None:
        """Compute value-function explained variance for logging, mirroring monolithic PPO."""
        with torch.no_grad():
            y_pred = self.replay.buffer["values"].flatten()
            y_true = self.replay.buffer["advantages_full"].flatten() + self.replay.buffer["values"].flatten()
            var_y = y_true.var()
            ev = (1 - (y_true - y_pred).var() / var_y).item() if var_y > 0 else 0.0
            self.loss_tracker["explained_variance"].append(float(ev))

        super().on_train_phase_end(context)
