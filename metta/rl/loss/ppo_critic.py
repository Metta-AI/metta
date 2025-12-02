from typing import Any

import numpy as np
import torch
from pydantic import Field
from tensordict import NonTensorData, TensorDict
from torch import Tensor
from torchrl.data import Composite, UnboundedContinuous, UnboundedDiscrete

from metta.agent.policy import Policy
from metta.rl.advantage import compute_advantage
from metta.rl.loss.loss import Loss, LossConfig
from metta.rl.loss.replay_samplers import prio_sample
from metta.rl.training import ComponentContext, TrainingEnvironment
from metta.rl.utils import prepare_policy_forward_td


class PPOCriticConfig(LossConfig):
    vf_clip_coef: float = Field(default=0.1, ge=0)
    vf_coef: float = Field(default=0.897619, ge=0)
    # Value loss clipping toggle
    clip_vloss: bool = True
    gamma: float = Field(default=0.977, ge=0, le=1.0)
    gae_lambda: float = Field(default=0.891477, ge=0, le=1.0)
    prio_alpha: float = Field(default=0.0, ge=0, le=1.0)
    prio_beta0: float = Field(default=0.6, ge=0, le=1.0)
    profiles: list[str] | None = Field(default=None, description="Optional loss profiles this loss should run for.")

    # control flow for forwarding and sampling. clunky but needed if other losses want to drive (e.g. action supervised)
    sample_enabled: bool = Field(default=True)  # if true, this loss samples from buffer during training
    train_forward_enabled: bool = Field(default=True)  # if true, this forwards the policy under training
    rollout_forward_enabled: bool = Field(default=True)  # if true, this forwards the policy under rollout

    deferred_training_start_step: int | None = None  # if set, sample/train_forward enable after this step

    def create(
        self,
        policy: Policy,
        trainer_cfg: Any,
        env: TrainingEnvironment,
        device: torch.device,
        instance_name: str,
        loss_config: Any,
    ) -> "PPOCritic":
        """Points to the PPO class for initialization."""
        return PPOCritic(
            policy,
            trainer_cfg,
            env,
            device,
            instance_name=instance_name,
            loss_config=loss_config,
        )


class PPOCritic(Loss):
    """PPO value loss."""

    __slots__ = (
        "advantages",
        "burn_in_steps",
        "burn_in_steps_iter",
        "sample_enabled",
        "train_forward_enabled",
        "rollout_forward_enabled",
    )

    def __init__(
        self,
        policy: Policy,
        trainer_cfg: Any,
        env: TrainingEnvironment,
        device: torch.device,
        instance_name: str,
        loss_config: Any,
    ):
        super().__init__(policy, trainer_cfg, env, device, instance_name, loss_config)
        self.advantages = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self.sample_enabled = self.cfg.sample_enabled
        self.train_forward_enabled = self.cfg.train_forward_enabled
        self.rollout_forward_enabled = self.cfg.rollout_forward_enabled
        self.trainable_only = True
        self.loss_profiles: set[int] | None = None

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

    def on_rollout_start(self, context: ComponentContext | None = None) -> None:
        """Called before starting a rollout phase."""
        super().on_rollout_start(context)
        if self.cfg.deferred_training_start_step is not None:
            if self._require_context(context).agent_step >= self.cfg.deferred_training_start_step:
                self.sample_enabled = True
                self.train_forward_enabled = True

    def run_rollout(self, td: TensorDict, context: ComponentContext) -> None:
        if not self.rollout_forward_enabled:
            return

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
        self, shared_loss_data: TensorDict, context: ComponentContext, mb_idx: int
    ) -> tuple[Tensor, TensorDict, bool]:
        # compute advantages on the first mb
        if mb_idx == 0:
            # a hack because loss run gates can get updated between rollout and train
            if self.cfg.deferred_training_start_step is not None:
                if self._require_context(context).agent_step >= self.cfg.deferred_training_start_step:
                    self.sample_enabled = True
                    self.train_forward_enabled = True

            advantages = torch.zeros_like(self.replay.buffer["values"], device=self.device)
            self.advantages = compute_advantage(
                self.replay.buffer["values"],
                self.replay.buffer["rewards"],
                self.replay.buffer["dones"],
                torch.ones_like(self.replay.buffer["values"]),
                advantages,
                self.cfg.gamma,
                self.cfg.gae_lambda,
                1.0,  # v-trace is used in PPO actor instead. 1.0 means no v-trace
                1.0,  # v-trace is used in PPO actor instead. 1.0 means no v-trace
                self.device,
            )

        # sample from the buffer if called for
        if self.sample_enabled:
            minibatch, indices, prio_weights = prio_sample(
                buffer=self.replay,
                mb_idx=mb_idx,
                epoch=context.epoch,
                total_timesteps=self.trainer_cfg.total_timesteps,
                batch_size=self.trainer_cfg.batch_size,
                prio_alpha=self.cfg.prio_alpha,
                prio_beta0=self.cfg.prio_beta0,
                advantages=self.advantages,
            )
            shared_loss_data["sampled_mb"] = minibatch
            shared_loss_data["indices"] = NonTensorData(indices)
            shared_loss_data["prio_weights"] = prio_weights
        else:
            minibatch = shared_loss_data["sampled_mb"]
            indices = shared_loss_data["indices"]
            if isinstance(indices, NonTensorData):
                indices = indices.data

            if "prio_weights" not in shared_loss_data:
                shared_loss_data["prio_weights"] = torch.ones(
                    (minibatch.shape[0], minibatch.shape[1]),
                    device=self.device,
                    dtype=torch.float32,
                )

        shared_loss_data = self._filter_minibatch(shared_loss_data)
        minibatch = shared_loss_data["sampled_mb"]
        if minibatch.batch_size.numel() == 0:  # early exit if minibatch is empty
            return self._zero_tensor, shared_loss_data, False
        indices = shared_loss_data["indices"]
        if isinstance(indices, NonTensorData):
            indices = indices.data
        prio_weights = shared_loss_data["prio_weights"]

        shared_loss_data["advantages"] = self.advantages[indices]
        # Share gamma/lambda with other losses (e.g. actor) to ensure consistency
        batch_size = shared_loss_data.batch_size
        shared_loss_data["gamma"] = torch.full(batch_size, self.cfg.gamma, device=self.device)
        shared_loss_data["gae_lambda"] = torch.full(batch_size, self.cfg.gae_lambda, device=self.device)

        # forward the policy if called for
        if self.train_forward_enabled:
            policy_td, B, TT = prepare_policy_forward_td(minibatch, self.policy_experience_spec, clone=False)
            flat_actions = minibatch["actions"].reshape(B * TT, -1)
            self.policy.reset_memory()
            policy_td = self.policy.forward(policy_td, action=flat_actions)
            policy_td = policy_td.reshape(B, TT)
            shared_loss_data["policy_td"] = policy_td

        # compute value loss
        old_values = minibatch["values"]
        returns = shared_loss_data["advantages"] + minibatch["values"]
        minibatch["returns"] = returns
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
            y_true = self.advantages.flatten() + self.replay.buffer["values"].flatten()
            var_y = y_true.var()
            ev = (1 - (y_true - y_pred).var() / var_y).item() if var_y > 0 else 0.0
            self.loss_tracker["explained_variance"].append(float(ev))

        super().on_train_phase_end(context)
