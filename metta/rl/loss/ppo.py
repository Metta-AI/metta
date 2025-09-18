from typing import TYPE_CHECKING, Any, Tuple

import numpy as np
import torch
from pydantic import Field
from tensordict import NonTensorData, TensorDict
from torch import Tensor
from torchrl.data import Composite, MultiCategorical, UnboundedContinuous

from metta.mettagrid.config import Config
from metta.rl.advantage import compute_advantage, normalize_advantage_distributed
from metta.rl.loss.loss import Loss
from metta.utils.batch import calculate_prioritized_sampling_params

# from metta.rl.trainer_config import TrainerConfig

if TYPE_CHECKING:
    from metta.agent.policy_base import Policy
    from metta.rl.training.training_environment import TrainingEnvironment


class PrioritizedExperienceReplayConfig(Config):
    # Alpha=0 disables prioritization (uniform sampling), Type 2 default to be updated by sweep
    prio_alpha: float = Field(default=0.0, ge=0, le=1.0)
    # Beta0=0.6: From Schaul et al. (2016) "Prioritized Experience Replay" paper
    prio_beta0: float = Field(default=0.6, ge=0, le=1.0)


class VTraceConfig(Config):
    # V-trace rho clipping at 1.0: From IMPALA paper (Espeholt et al., 2018), standard for on-policy
    rho_clip: float = Field(default=1.0, gt=0)
    # V-trace c clipping at 1.0: From IMPALA paper (Espeholt et al., 2018), standard for on-policy
    c_clip: float = Field(default=1.0, gt=0)


class PPOConfig(Config):
    schedule: None = None  # TODO: Implement this
    # PPO hyperparameters
    # Clip coefficient: 0.1 is conservative, common range 0.1-0.3 from PPO paper (Schulman et al., 2017)
    clip_coef: float = Field(default=0.264407, gt=0, le=1.0)
    # Entropy coefficient: Type 2 default chosen from sweep
    ent_coef: float = Field(default=0.010000, ge=0)
    # GAE lambda: Type 2 default chosen from sweep, deviates from typical 0.95, bias/variance tradeoff
    gae_lambda: float = Field(default=0.891477, ge=0, le=1.0)
    # Gamma: Type 2 default chosen from sweep, deviates from typical 0.99, suggests shorter
    # effective horizon for multi-agent
    gamma: float = Field(default=0.977, ge=0, le=1.0)

    # Training parameters
    # Gradient clipping: 0.5 is standard PPO default to prevent instability
    max_grad_norm: float = Field(default=0.5, gt=0)
    # Value function clipping: Matches policy clip for consistency
    vf_clip_coef: float = Field(default=0.1, ge=0)
    # Value coefficient: Type 2 default chosen from sweep, balances policy vs value loss
    vf_coef: float = Field(default=0.897619, ge=0)
    # L2 regularization: Disabled by default, common in RL
    l2_reg_loss_coef: float = Field(default=0, ge=0)
    l2_init_loss_coef: float = Field(default=0, ge=0)

    # Normalization and clipping
    # Advantage normalization: Standard PPO practice for stability
    norm_adv: bool = True
    # Value loss clipping: PPO best practice from implementation details
    clip_vloss: bool = True
    # Target KL: None allows unlimited updates, common for stable environments
    target_kl: float | None = None

    # Steps in rollout to discard before starting to save experience (for LSTM_reset). This field needs to be moved to
    # a rollout spec that we have not created yet. Alternatively, it could be an attribute of the policy
    burn_in_steps: int = Field(default=0, ge=0)

    vtrace: VTraceConfig = Field(default_factory=VTraceConfig)

    prioritized_experience_replay: PrioritizedExperienceReplayConfig = Field(
        default_factory=PrioritizedExperienceReplayConfig
    )

    def create(
        self,
        policy: "Policy",
        trainer_cfg: Any,
        env: "TrainingEnvironment",
        device: torch.device,
        instance_name: str,
        loss_config: Any,
    ):
        """Points to the PPO class for initialization."""
        return PPO(
            policy,
            trainer_cfg,
            env,
            device,
            instance_name=instance_name,
            loss_config=loss_config,
        )


class PPO(Loss):
    """This could be slightly faster by looking for repeated access to hashed vars."""

    __slots__ = (
        "advantages",
        "anneal_beta",
        "burn_in_steps",
        "burn_in_steps_iter",
    )

    def __init__(
        self,
        policy: "Policy",
        trainer_cfg: Any,
        env: "TrainingEnvironment",
        device: torch.device,
        instance_name: str,
        loss_config: Any,
    ):
        super().__init__(policy, trainer_cfg, env, device, instance_name, loss_config)
        self.advantages = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self.anneal_beta = 0.0
        self.burn_in_steps = self.loss_cfg.burn_in_steps
        self.burn_in_steps_iter = 0

    def get_experience_spec(self) -> Composite:
        act_space = self.env.single_action_space
        nvec = act_space.nvec
        act_dtype = torch.int32 if np.issubdtype(act_space.dtype, np.integer) else torch.float32
        scalar_f32 = UnboundedContinuous(shape=torch.Size([]), dtype=torch.float32)

        return Composite(
            rewards=scalar_f32,
            dones=scalar_f32,
            truncateds=scalar_f32,
            actions=MultiCategorical(
                nvec=nvec,
                dtype=act_dtype,
            ),
            act_log_prob=scalar_f32,
            values=scalar_f32,
        )

    # BaseLoss calls this method
    def run_rollout(self, td: TensorDict, env_id: slice) -> None:
        with torch.no_grad():
            self.policy.forward(td)

        if self.burn_in_steps_iter < self.burn_in_steps:
            self.burn_in_steps_iter += 1
            return

        # Store experience
        self.replay.store(data_td=td, env_id=env_id)

        return

    # BaseLoss calls this method
    def run_train(
        self, shared_loss_data: TensorDict, env_id: slice, epoch: int, mb_idx: int
    ) -> tuple[Tensor, TensorDict, bool]:
        """This is the PPO algorithm training loop."""
        # Tell the policy that we're starting a new minibatch so it can do things like reset its memory
        stop_update_epoch = False
        self.policy.reset_memory()
        self.burn_in_steps_iter = 0
        # Check if we should early stop this update epoch (on subsequent minibatches)
        if self.loss_cfg.target_kl is not None and mb_idx > 0:
            average_approx_kl = np.mean(self.loss_tracker["approx_kl"]) if self.loss_tracker["approx_kl"] else 0.0
            if average_approx_kl > self.loss_cfg.target_kl:
                stop_update_epoch = True

        # On the first minibatch of the update epoch, compute advantages and sampling params
        if mb_idx == 0:
            self.advantages, self.anneal_beta = self._on_first_mb(
                epoch, self.trainer_cfg.total_timesteps, self.trainer_cfg.batch_size
            )

        # Then sample from the buffer (this happens at every minibatch)
        minibatch, indices, prio_weights = self._sample_minibatch(
            advantages=self.advantages,
            prio_alpha=self.loss_cfg.prioritized_experience_replay.prio_alpha,
            prio_beta=self.anneal_beta,
        )

        shared_loss_data["sampled_mb"] = minibatch  # one loss should write the sampled mb for others to use
        shared_loss_data["indices"] = NonTensorData(indices)  # av this breaks compile

        # Then forward the policy using the sampled minibatch
        policy_td = minibatch.select(*self.policy_experience_spec.keys(include_nested=True))
        B = policy_td.batch_size[0]
        TT = policy_td.batch_size[1]
        policy_td = policy_td.reshape(policy_td.batch_size.numel())  # flatten to BT
        policy_td.set("bptt", torch.full((B * TT,), TT, device=policy_td.device, dtype=torch.long))

        policy_td = self.policy.forward(policy_td, action=minibatch["actions"])
        shared_loss_data["policy_td"] = policy_td.reshape(B, TT)  # write the policy output td for others to reuse

        # Finally, calculate the loss!
        loss = self._process_minibatch_update(
            minibatch=minibatch,
            policy_td=policy_td,
            indices=indices,
            prio_weights=prio_weights,
        )

        return loss, shared_loss_data, stop_update_epoch

    def on_train_phase_end(self, epoch: int) -> None:
        with torch.no_grad():
            y_pred = self.replay.buffer["values"].flatten()
            y_true = self.advantages.flatten() + self.replay.buffer["values"].flatten()
            var_y = y_true.var()
            ev = (1 - (y_true - y_pred).var() / var_y).item() if var_y > 0 else 0.0
            self.loss_tracker["explained_variance"].append(float(ev))

    def _on_first_mb(self, epoch: int, total_timesteps: int, batch_size: int) -> tuple[Tensor, float]:
        # reset importance sampling ratio
        if "ratio" in self.replay.buffer.keys():
            self.replay.buffer["ratio"].fill_(1.0)

        with torch.no_grad():
            anneal_beta = calculate_prioritized_sampling_params(
                epoch=epoch,
                total_timesteps=self.trainer_cfg.total_timesteps,
                batch_size=self.trainer_cfg.batch_size,
                prio_alpha=self.loss_cfg.prioritized_experience_replay.prio_alpha,
                prio_beta0=self.loss_cfg.prioritized_experience_replay.prio_beta0,
            )

            # Compute initial advantages
            advantages = torch.zeros(self.replay.buffer["values"].shape, device=self.device)
            initial_importance_sampling_ratio = torch.ones_like(self.replay.buffer["values"])

            advantages = compute_advantage(
                self.replay.buffer["values"],
                self.replay.buffer["rewards"],
                self.replay.buffer["dones"],
                initial_importance_sampling_ratio,
                advantages,
                self.loss_cfg.gamma,
                self.loss_cfg.gae_lambda,
                self.loss_cfg.vtrace.rho_clip,
                self.loss_cfg.vtrace.c_clip,
                self.device,
            )

        return advantages, anneal_beta

    def _process_minibatch_update(
        self,
        minibatch: TensorDict,
        policy_td: TensorDict,
        indices: Tensor,
        prio_weights: Tensor,
    ) -> Tensor:
        old_act_log_prob = minibatch["act_log_prob"]
        new_logprob = policy_td["act_log_prob"].reshape(old_act_log_prob.shape)
        entropy = policy_td["entropy"]
        newvalue = policy_td["values"]

        logratio = new_logprob - old_act_log_prob
        # Bound the log ratio to prevent extreme importance sampling ratios
        logratio = torch.clamp(logratio, -10, 10)  # exp(-10) ≈ 0.000045, exp(10) ≈ 22026
        importance_sampling_ratio = logratio.exp()

        # Re-compute advantages with new ratios (V-trace)
        adv = compute_advantage(
            minibatch["values"],
            minibatch["rewards"],
            minibatch["dones"],
            importance_sampling_ratio,
            minibatch["advantages"],
            self.loss_cfg.gamma,
            self.loss_cfg.gae_lambda,
            self.loss_cfg.vtrace.rho_clip,
            self.loss_cfg.vtrace.c_clip,
            self.device,
        )

        # Normalize advantages with distributed support, then apply prioritized weights
        adv = normalize_advantage_distributed(adv, self.loss_cfg.norm_adv)
        adv = prio_weights * adv

        # Compute losses
        pg_loss, v_loss, entropy_loss, approx_kl, clipfrac = self.compute_ppo_losses(
            minibatch,
            new_logprob,
            entropy,
            newvalue,
            importance_sampling_ratio,
            adv,
        )

        loss = pg_loss - self.loss_cfg.ent_coef * entropy_loss + v_loss * self.loss_cfg.vf_coef

        # Update values and ratio in experience buffer
        update_td = TensorDict(
            {"values": newvalue.view(minibatch["values"].shape).detach(), "ratio": importance_sampling_ratio.detach()},
            batch_size=minibatch.batch_size,
        )
        self.replay.update(indices, update_td)

        # Update loss tracking
        self.loss_tracker["policy_loss"].append(float(pg_loss.item()))
        self.loss_tracker["value_loss"].append(float(v_loss.item()))
        self.loss_tracker["entropy"].append(float(entropy_loss.item()))
        self.loss_tracker["approx_kl"].append(float(approx_kl.item()))
        self.loss_tracker["clipfrac"].append(float(clipfrac.item()))
        # av why were these getting normalized by mb_idx???
        self.loss_tracker["importance"].append(float(importance_sampling_ratio.mean().item()))
        self.loss_tracker["current_logprobs"].append(float(new_logprob.mean().item()))

        return loss

    def compute_ppo_losses(
        self,
        minibatch: TensorDict,
        new_logprob: Tensor,
        entropy: Tensor,
        newvalue: Tensor,
        importance_sampling_ratio: Tensor,
        adv: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Compute PPO losses for policy and value functions."""
        # Policy loss
        pg_loss1 = -adv * importance_sampling_ratio
        pg_loss2 = -adv * torch.clamp(
            importance_sampling_ratio,
            1 - self.loss_cfg.clip_coef,
            1 + self.loss_cfg.clip_coef,
        )
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()
        returns = minibatch["returns"]
        old_values = minibatch["values"]

        # Value loss
        newvalue_reshaped = newvalue.view(returns.shape)
        if self.loss_cfg.clip_vloss:
            v_loss_unclipped = (newvalue_reshaped - returns) ** 2
            vf_clip_coef = self.loss_cfg.vf_clip_coef
            v_clipped = old_values + torch.clamp(
                newvalue_reshaped - old_values,
                -vf_clip_coef,
                vf_clip_coef,
            )
            v_loss_clipped = (v_clipped - returns) ** 2
            v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
        else:
            v_loss = 0.5 * ((newvalue_reshaped - returns) ** 2).mean()

        entropy_loss = entropy.mean()

        # Compute metrics
        with torch.no_grad():
            logratio = new_logprob - minibatch["act_log_prob"]
            approx_kl = ((importance_sampling_ratio - 1) - logratio).mean()
            clipfrac = ((importance_sampling_ratio - 1.0).abs() > self.loss_cfg.clip_coef).float().mean()

        return pg_loss, v_loss, entropy_loss, approx_kl, clipfrac

    def _sample_minibatch(
        self,
        advantages: Tensor,
        prio_alpha: float,
        prio_beta: float,
    ) -> tuple[TensorDict, Tensor, Tensor]:
        """Sample a prioritized minibatch."""
        # Prioritized sampling based on advantage magnitude
        adv_magnitude = advantages.abs().sum(dim=1)
        prio_weights = torch.nan_to_num(adv_magnitude**prio_alpha, 0, 0, 0)
        prio_probs = (prio_weights + 1e-6) / (prio_weights.sum() + 1e-6)

        # Sample segment indices
        idx = torch.multinomial(prio_probs, self.replay.minibatch_segments)

        minibatch = self.replay.buffer[idx]

        with torch.no_grad():
            minibatch["advantages"] = advantages[idx]
            minibatch["returns"] = advantages[idx] + minibatch["values"]
            prio_weights = (self.replay.segments * prio_probs[idx, None]) ** -prio_beta
        return minibatch.clone(), idx, prio_weights


def compute_ppo_losses(
    minibatch: TensorDict,
    new_logprob: Tensor,
    entropy: Tensor,
    newvalue: Tensor,
    importance_sampling_ratio: Tensor,
    adv: Tensor,
    trainer_cfg: Any,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Standalone function to compute PPO losses for policy and value functions."""
    # Policy loss
    pg_loss1 = -adv * importance_sampling_ratio
    pg_loss2 = -adv * torch.clamp(
        importance_sampling_ratio,
        1 - trainer_cfg.ppo.clip_coef,
        1 + trainer_cfg.ppo.clip_coef,
    )
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
    returns = minibatch["returns"]
    old_values = minibatch["values"]

    # Value loss
    newvalue_reshaped = newvalue.view(returns.shape)
    if trainer_cfg.ppo.clip_vloss:
        v_loss_unclipped = (newvalue_reshaped - returns) ** 2
        vf_clip_coef = trainer_cfg.ppo.vf_clip_coef
        v_clipped = old_values.detach() + torch.clamp(
            newvalue_reshaped - old_values.detach(),
            -vf_clip_coef,
            vf_clip_coef,
        )
        v_loss_clipped = (v_clipped - returns) ** 2
        v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
    else:
        v_loss = 0.5 * ((newvalue_reshaped - returns) ** 2).mean()

    entropy_loss = entropy.mean()

    # Compute metrics
    with torch.no_grad():
        logratio = new_logprob - minibatch["act_log_prob"]
        approx_kl = ((importance_sampling_ratio - 1) - logratio).mean()
        clipfrac = ((importance_sampling_ratio - 1.0).abs() > trainer_cfg.ppo.clip_coef).float().mean()

    return pg_loss, v_loss, entropy_loss, approx_kl, clipfrac
