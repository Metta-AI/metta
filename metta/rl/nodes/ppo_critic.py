from typing import Any, Optional

import numpy as np
import torch
from pydantic import Field
from tensordict import TensorDict
from torch import Tensor
from torchrl.data import Composite, UnboundedContinuous, UnboundedDiscrete
from typing_extensions import Literal

from metta.agent.policy import Policy
from metta.rl.advantage import td_lambda_reverse_scan
from metta.rl.nodes.base import NodeBase, NodeConfig, analyze_loss_alignment
from metta.rl.nodes.registry import NodeSpec
from metta.rl.training import ComponentContext, TrainingEnvironment


class PPOCriticConfig(NodeConfig):
    vf_clip_coef: float = Field(default=0.1, ge=0)
    vf_coef: float = Field(default=0.49657103419303894, ge=0)
    # Value loss clipping toggle
    clip_vloss: bool = True
    critic_update: Literal["mse", "gtd_lambda"] = "gtd_lambda"
    aux_coef: float = Field(default=1.0, ge=0)
    beta: float = Field(default=1.0, ge=0)
    teacher_offpolicy_rho_clip: float = Field(default=1.0, gt=0)

    def create(
        self,
        policy: Policy,
        trainer_cfg: Any,
        env: TrainingEnvironment,
        device: torch.device,
        instance_name: str,
    ) -> "PPOCritic":
        return PPOCritic(policy, trainer_cfg, env, device, instance_name, self)


class PPOCritic(NodeBase):
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

        env_slice = self._training_env_id(context)
        self.replay.store(data_td=td, env_id=env_slice)

    def policy_output_keys(self, policy_td: Optional[TensorDict] = None) -> set[str]:
        if self.cfg.critic_update == "gtd_lambda":
            return {"values", "h_values"}
        return {"values"}

    def _importance_sampled_delta_lambda(
        self,
        *,
        values: Tensor,
        rewards: Tensor,
        dones: Tensor,
        rho: Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> Tensor:
        _, tt = values.shape
        delta_lambda = torch.zeros_like(values)
        if tt <= 1:
            return delta_lambda

        terminal_next = dones[:, 1:]
        mask_next = 1.0 - terminal_next

        delta = rewards[:, 1:] + gamma * mask_next * values[:, 1:] - values[:, :-1]  # [B, TT-1]
        rho = rho[:, :-1].clamp(max=float(self.cfg.teacher_offpolicy_rho_clip))

        x = rho * delta
        discounts = rho * mask_next
        delta_lambda[:, :-1] = td_lambda_reverse_scan(x, discounts, float(gamma * gae_lambda))

        return delta_lambda

    def run_train(
        self, shared_loss_data: TensorDict, context: ComponentContext, mb_idx: int
    ) -> tuple[Tensor, TensorDict, bool]:
        # Sampling happens in the core loop; use the shared minibatch and indices.
        minibatch = shared_loss_data["sampled_mb"]

        if minibatch.batch_size.numel() == 0:  # early exit if minibatch is empty
            return self._zero_tensor, shared_loss_data, False

        # Advantages are computed in the core loop and passed through shared_loss_data.
        # Keep the full advantages around for explained variance logging and prioritized sampling.
        old_values = minibatch["values"]
        if self.cfg.critic_update == "gtd_lambda":
            policy_td = shared_loss_data["policy_td"]
            if "h_values" not in policy_td.keys():
                raise RuntimeError("Policy must output 'h_values' for critic_update='gtd_lambda'")

            new_values = policy_td["values"].reshape(old_values.shape)
            h_values = policy_td["h_values"]
            if h_values.dim() == 3 and h_values.shape[-1] == 1:
                h_values = h_values.squeeze(-1)
            h_values = h_values.reshape(old_values.shape)

            delta_lambda = shared_loss_data["advantages_pg"]
            if "teacher_mask" in minibatch.keys():
                teacher_mask = minibatch["teacher_mask"][:, 0]
                if bool(teacher_mask.any()):
                    if "act_log_prob" not in policy_td.keys():
                        raise RuntimeError("Teacher-slice TD(λ) correction requires policy_td['act_log_prob']")
                    rho = policy_td["act_log_prob"].reshape(minibatch["actions"].shape).exp()
                    rho_trim = rho.detach()[teacher_mask][:, :-1]
                    rho_clip = float(self.cfg.teacher_offpolicy_rho_clip)
                    self.loss_tracker["teacher_td_lambda_rho_clipfrac"].append(
                        float((rho_trim > rho_clip).float().mean().item())
                    )
                    centered_rewards = minibatch["rewards"] - minibatch["reward_baseline"]
                    corrected = self._importance_sampled_delta_lambda(
                        values=new_values[teacher_mask],
                        rewards=centered_rewards[teacher_mask],
                        dones=minibatch["dones"][teacher_mask],
                        rho=rho.detach()[teacher_mask],
                        gamma=float(context.config.advantage.gamma),
                        gae_lambda=float(context.config.advantage.gae_lambda),
                    )
                    delta_lambda = delta_lambda.clone()
                    delta_lambda[teacher_mask] = corrected

            # Use only valid transitions (t=0..TT-2). The last step is padding.
            dl = delta_lambda[:, :-1]
            v_t = new_values[:, :-1]
            h_t = h_values[:, :-1]

            critic_loss = (h_t.detach() * dl).mean() - ((dl.detach() - h_t.detach()) * v_t).mean()

            l2_sum = torch.tensor(0.0, device=critic_loss.device, dtype=critic_loss.dtype)
            l2_count = 0
            if self.cfg.beta > 0:
                aux_module = getattr(self.policy, "gtd_aux", None)
                components = getattr(self.policy, "components", None)
                if aux_module is None and components is not None:
                    aux_module = components.get("gtd_aux", None)
                if aux_module is not None:
                    for param in aux_module.parameters():
                        l2_sum = l2_sum + (param * param).sum()
                        l2_count += int(param.numel())
            l2 = l2_sum / max(l2_count, 1)
            aux_loss = 0.5 * ((dl.detach() - h_t) ** 2).mean() + 0.5 * float(self.cfg.beta) * l2

            total = float(self.cfg.vf_coef) * critic_loss + float(self.cfg.aux_coef) * aux_loss
            self.loss_tracker["value_loss"].append(float(total.item()))
            self.loss_tracker["gtd_critic_loss"].append(float(critic_loss.item()))
            self.loss_tracker["gtd_aux_loss"].append(float(aux_loss.item()))
            self.loss_tracker["gtd_h_mse"].append(float(((dl.detach() - h_t) ** 2).mean().item()))
            self.loss_tracker["gtd_delta_lambda_abs"].append(float(dl.detach().abs().mean().item()))

            # Update values in experience buffer for advantage_full recomputation + EV logging.
            update_td = TensorDict(
                {
                    "values": new_values.reshape(minibatch["values"].shape).detach(),
                },
                batch_size=minibatch.batch_size,
            )
            indices = shared_loss_data["indices"][:, 0]
            self.replay.update(indices, update_td)

            return total, shared_loss_data, False

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
                v_loss_vec = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped)
            else:
                v_loss_vec = 0.5 * ((newvalue_reshaped - returns) ** 2)

            v_loss = v_loss_vec.mean()

            shared_loss_data["ppo_val_loss_vec"] = v_loss_vec

            # 12-21-25 av experimental code. cute but delete later (compare with Kickstarter value loss if available)
            if "ks_val_loss_vec" in shared_loss_data:
                analyze_loss_alignment(
                    shared_data=shared_loss_data,
                    name1="ks_val",
                    name2="ppo_val",
                    params=list(self.policy.parameters()),
                    tracker=self.loss_tracker,
                )

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


NODE_SPECS = [
    NodeSpec(
        key="ppo_critic",
        config_cls=PPOCriticConfig,
        default_enabled=True,
        has_rollout=True,
        has_train=True,
    )
]
