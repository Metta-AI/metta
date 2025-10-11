from typing import Any, Tuple

import numpy as np
import torch
from pydantic import Field
from tensordict import NonTensorData, TensorDict
from torch import Tensor
from torchrl.data import Composite, UnboundedContinuous, UnboundedDiscrete

from metta.agent.policy import Policy
from metta.rl.advantage import compute_advantage, normalize_advantage_distributed
from metta.rl.loss import Loss
from metta.rl.training import ComponentContext, TrainingEnvironment
from metta.utils.batch import calculate_prioritized_sampling_params
from mettagrid.base_config import Config


class PrioritizedExperienceReplayConfig(Config):
    # Alpha=0 means uniform sampling; tuned via sweep
    prio_alpha: float = Field(default=0.0, ge=0, le=1.0)
    # Beta baseline per Schaul et al. (2016)
    prio_beta0: float = Field(default=0.6, ge=0, le=1.0)


class VTraceConfig(Config):
    # Defaults follow IMPALA (Espeholt et al., 2018)
    rho_clip: float = Field(default=1.0, gt=0)
    c_clip: float = Field(default=1.0, gt=0)


class PPOConfig(Config):
    schedule: None = None  # TODO: Implement this
    # PPO hyperparameters
    # Clip coefficient (0.1-0.3 typical; Schulman et al. 2017)
    clip_coef: float = Field(default=0.264407, gt=0, le=1.0)
    # Entropy term weight from sweep
    ent_coef: float = Field(default=0.010000, ge=0)
    # MC coefficients (simple sum, no normalization)
    mc_pg_coef: float = Field(default=0.3, ge=0)
    mc_ent_coef: float = Field(default=0.003, ge=0)
    # GAE lambda tuned via sweep (cf. standard 0.95)
    gae_lambda: float = Field(default=0.891477, ge=0, le=1.0)
    # Gamma tuned for shorter effective horizon
    gamma: float = Field(default=0.977, ge=0, le=1.0)

    # Training parameters
    # Gradient clipping default
    max_grad_norm: float = Field(default=0.5, gt=0)
    # Value clipping mirrors policy clip
    vf_clip_coef: float = Field(default=0.1, ge=0)
    # Value term weight from sweep
    vf_coef: float = Field(default=0.897619, ge=0)
    # L2 regularization defaults to disabled
    l2_reg_loss_coef: float = Field(default=0, ge=0)
    l2_init_loss_coef: float = Field(default=0, ge=0)

    # Normalization and clipping
    # Advantage normalization toggle
    norm_adv: bool = True
    # Value loss clipping toggle
    clip_vloss: bool = True
    # Target KL for early stopping (None disables)
    target_kl: float | None = None

    vtrace: VTraceConfig = Field(default_factory=VTraceConfig)

    prioritized_experience_replay: PrioritizedExperienceReplayConfig = Field(
        default_factory=PrioritizedExperienceReplayConfig
    )

    def create(
        self,
        policy: Policy,
        trainer_cfg: Any,
        env: TrainingEnvironment,
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
    """PPO loss with prioritized replay and V-trace tweaks."""

    __slots__ = (
        "advantages",
        "anneal_beta",
        "burn_in_steps",
        "burn_in_steps_iter",
        "last_action",
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
        self.anneal_beta = 0.0
        self.burn_in_steps = 0
        if hasattr(self.policy, "burn_in_steps"):
            self.burn_in_steps = self.policy.burn_in_steps
        self.burn_in_steps_iter = 0
        self.last_action = None
        self.register_state_attr("anneal_beta", "burn_in_steps_iter")

    def get_experience_spec(self) -> Composite:
        act_space = self.env.single_action_space
        act_dtype = torch.int32 if np.issubdtype(act_space.dtype, np.integer) else torch.float32
        scalar_f32 = UnboundedContinuous(shape=torch.Size([]), dtype=torch.float32)

        return Composite(
            rewards=scalar_f32,
            dones=scalar_f32,
            truncateds=scalar_f32,
            actions=UnboundedDiscrete(shape=torch.Size([]), dtype=act_dtype),
            act_log_prob=scalar_f32,
            values=scalar_f32,
            mc_act_log_prob=scalar_f32,
            mc_actions=UnboundedDiscrete(shape=torch.Size([]), dtype=torch.int32),
        )

    def run_rollout(self, td: TensorDict, context: ComponentContext) -> None:
        with torch.no_grad():
            self.policy.forward(td)

        if self.burn_in_steps_iter < self.burn_in_steps:
            self.burn_in_steps_iter += 1
            return

        # Store experience
        env_slice = context.training_env_id
        if env_slice is None:
            raise RuntimeError("ComponentContext.training_env_id is required for PPO rollout")
        self.replay.store(data_td=td, env_id=env_slice)

        return

    def run_train(
        self, shared_loss_data: TensorDict, context: ComponentContext, mb_idx: int
    ) -> tuple[Tensor, TensorDict, bool]:
        """This is the PPO algorithm training loop."""
        config = self.loss_cfg
        stop_update_epoch = False
        self.policy.reset_memory()
        self.burn_in_steps_iter = 0
        if config.target_kl is not None and mb_idx > 0:
            avg_kl = np.mean(self.loss_tracker["approx_kl"]) if self.loss_tracker["approx_kl"] else 0.0
            if avg_kl > config.target_kl:
                stop_update_epoch = True

        # On the first minibatch of the update epoch, compute advantages and sampling params
        if mb_idx == 0:
            # Update MC action curriculum at the start of each epoch
            self._maybe_update_mc_action_curriculum(context)
            self.advantages, self.anneal_beta = self._on_first_mb(context)

        # Then sample from the buffer (this happens at every minibatch)
        minibatch, indices, prio_weights = self._sample_minibatch(
            advantages=self.advantages,
            prio_alpha=config.prioritized_experience_replay.prio_alpha,
            prio_beta=self.anneal_beta,
        )

        shared_loss_data["sampled_mb"] = minibatch  # one loss should write the sampled mb for others to use
        shared_loss_data["indices"] = NonTensorData(indices)  # av this breaks compile

        # Then forward the policy using the sampled minibatch
        policy_td = minibatch.select(*self.policy_experience_spec.keys(include_nested=True))
        B, TT = policy_td.batch_size
        policy_td = policy_td.reshape(B * TT)
        policy_td.set("bptt", torch.full((B * TT,), TT, device=policy_td.device, dtype=torch.long))
        policy_td.set("batch", torch.full((B * TT,), B, device=policy_td.device, dtype=torch.long))

        flat_actions = minibatch["actions"].reshape(B * TT, -1)

        policy_td = self.policy.forward(policy_td, action=flat_actions)
        shared_loss_data["policy_td"] = policy_td.reshape(B, TT)

        # Finally, calculate the loss!
        loss = self._process_minibatch_update(
            minibatch=minibatch,
            policy_td=policy_td,
            indices=indices,
            prio_weights=prio_weights,
        )

        return loss, shared_loss_data, stop_update_epoch

    def on_train_phase_end(self, context: ComponentContext) -> None:
        with torch.no_grad():
            y_pred = self.replay.buffer["values"].flatten()
            y_true = self.advantages.flatten() + self.replay.buffer["values"].flatten()
            var_y = y_true.var()
            ev = (1 - (y_true - y_pred).var() / var_y).item() if var_y > 0 else 0.0
            self.loss_tracker["explained_variance"].append(float(ev))

    def _on_first_mb(self, context: ComponentContext) -> tuple[Tensor, float]:
        # reset importance sampling ratio
        if "ratio" in self.replay.buffer.keys():
            self.replay.buffer["ratio"].fill_(1.0)

        cfg = self.loss_cfg
        with torch.no_grad():
            anneal_beta = calculate_prioritized_sampling_params(
                epoch=context.epoch,
                total_timesteps=self.trainer_cfg.total_timesteps,
                batch_size=self.trainer_cfg.batch_size,
                prio_alpha=cfg.prioritized_experience_replay.prio_alpha,
                prio_beta0=cfg.prioritized_experience_replay.prio_beta0,
            )

            advantages = torch.zeros_like(self.replay.buffer["values"], device=self.device)
            advantages = compute_advantage(
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

        return advantages, anneal_beta

    def _process_minibatch_update(
        self,
        minibatch: TensorDict,
        policy_td: TensorDict,
        indices: Tensor,
        prio_weights: Tensor,
    ) -> Tensor:
        cfg = self.loss_cfg
        old_logprob = minibatch["act_log_prob"]
        new_logprob = policy_td["act_log_prob"].reshape(old_logprob.shape)
        entropy = policy_td["entropy"]
        newvalue = policy_td["values"]

        importance_sampling_ratio = self._importance_ratio(new_logprob, old_logprob)

        mc_old_logprob = minibatch["mc_act_log_prob"]
        mc_new_logprob = policy_td["mc_act_log_prob"].reshape(mc_old_logprob.shape)
        mc_entropy = policy_td["mc_entropy"]

        mc_importance_sampling_ratio = self._importance_ratio(mc_new_logprob, mc_old_logprob)

        # Re-compute advantages with new ratios (V-trace)
        adv = compute_advantage(
            minibatch["values"],
            minibatch["rewards"],
            minibatch["dones"],
            importance_sampling_ratio,
            minibatch["advantages"],
            cfg.gamma,
            cfg.gae_lambda,
            cfg.vtrace.rho_clip,
            cfg.vtrace.c_clip,
            self.device,
        )

        # Normalize advantages with distributed support, then apply prioritized weights
        adv = normalize_advantage_distributed(adv, cfg.norm_adv)
        adv = prio_weights * adv

        # Compute losses
        (
            pg_loss,
            mc_pg_loss,
            v_loss,
            entropy_loss,
            mc_entropy_loss,
            approx_kl,
            mc_approx_kl,
            clipfrac,
            mc_clipfrac,
        ) = self.compute_ppo_losses(
            minibatch,
            new_logprob,
            mc_new_logprob,
            entropy,
            mc_entropy,
            newvalue,
            importance_sampling_ratio,
            mc_importance_sampling_ratio,
            adv,
        )

        # Simple weighted sum of normal and MC heads (no normalization)
        policy_loss = pg_loss + cfg.mc_pg_coef * mc_pg_loss
        entropy_term = cfg.ent_coef * entropy_loss + cfg.mc_ent_coef * mc_entropy_loss
        loss = policy_loss - entropy_term + v_loss * cfg.vf_coef

        # Update values and ratio in experience buffer
        update_td = TensorDict(
            {"values": newvalue.view(minibatch["values"].shape).detach(), "ratio": importance_sampling_ratio.detach()},
            batch_size=minibatch.batch_size,
        )
        self.replay.update(indices, update_td)

        # Update loss tracking
        self._track("policy_loss", pg_loss)
        self._track("mc_policy_loss", mc_pg_loss)
        self._track("value_loss", v_loss)
        self._track("entropy", entropy_loss)
        self._track("mc_entropy", mc_entropy_loss)
        self._track("approx_kl", approx_kl)
        self._track("mc_approx_kl", mc_approx_kl)
        self._track("clipfrac", clipfrac)
        self._track("mc_clipfrac", mc_clipfrac)
        self._track("importance", importance_sampling_ratio.mean())
        self._track("current_logprobs", new_logprob.mean())
        self._track("mc_importance", mc_importance_sampling_ratio.mean())
        self._track("mc_current_logprobs", mc_new_logprob.mean())

        return loss

    def compute_ppo_losses(
        self,
        minibatch: TensorDict,
        new_logprob: Tensor,
        mc_new_logprob: Tensor,
        entropy: Tensor,
        mc_entropy: Tensor,
        newvalue: Tensor,
        importance_sampling_ratio: Tensor,
        mc_importance_sampling_ratio: Tensor,
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

        mc_pg_loss1 = -adv * mc_importance_sampling_ratio
        mc_pg_loss2 = -adv * torch.clamp(
            mc_importance_sampling_ratio,
            1 - self.loss_cfg.clip_coef,
            1 + self.loss_cfg.clip_coef,
        )
        mc_pg_loss = torch.max(mc_pg_loss1, mc_pg_loss2).mean()

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

        mc_entropy_loss = mc_entropy.mean()

        # Compute metrics
        with torch.no_grad():
            logratio = new_logprob - minibatch["act_log_prob"]
            mc_logratio = mc_new_logprob - minibatch["mc_act_log_prob"]
            approx_kl = ((importance_sampling_ratio - 1) - logratio).mean()
            mc_approx_kl = ((mc_importance_sampling_ratio - 1) - mc_logratio).mean()
            clipfrac = ((importance_sampling_ratio - 1.0).abs() > self.loss_cfg.clip_coef).float().mean()
            mc_clipfrac = ((mc_importance_sampling_ratio - 1.0).abs() > self.loss_cfg.clip_coef).float().mean()

        return (
            pg_loss,
            mc_pg_loss,
            v_loss,
            entropy_loss,
            mc_entropy_loss,
            approx_kl,
            mc_approx_kl,
            clipfrac,
            mc_clipfrac,
        )

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

    def _importance_ratio(self, new_logprob: Tensor, old_logprob: Tensor) -> Tensor:
        logratio = torch.clamp(new_logprob - old_logprob, -10, 10)
        return logratio.exp()

    def _track(self, key: str, value: Tensor) -> None:
        self.loss_tracker[key].append(float(value.item()))

    # ------------------- MC action curriculum helpers -------------------

    def _mc_curriculum_allowed_actions(self, epoch: int) -> list[str]:
        """Return the list of allowed MC action names for the given epoch.

        Curriculum schedule:
        - start: ["noop_1", "noop_2", "focus_1"]
        - >=100: add "focus_2"
        - >=200: add "lstm_noise_1"
        - >=300: add "lstm_clear"
        - >=500: add "lstm_noise_2"
        """
        allowed: list[str] = ["noop_1", "noop_2", "focus_1"]
        if epoch >= 100:
            allowed.append("focus_2")
        if epoch >= 200:
            allowed.append("lstm_noise_1")
        if epoch >= 300:
            allowed.append("lstm_clear")
        if epoch >= 500:
            allowed.append("lstm_noise_2")
        return allowed

    def _maybe_update_mc_action_curriculum(self, context: ComponentContext) -> None:
        """Update MC action activation only on threshold epochs and only for MC action embedding.

        Looks up the MC `ActionEmbedding` component by name ("mc_action_embedding") and calls
        its `mc_initialize_to_environment` method with the allowed MC action names when the
        epoch matches one of the curriculum thresholds.
        """
        policy_obj = self.policy
        if not hasattr(policy_obj, "components"):
            return

        epoch = int(getattr(context, "epoch", 0) or 0)
        # Call only when we cross a threshold epoch to avoid redundant resets
        threshold_epochs = {0, 100, 200, 300, 500}
        if epoch not in threshold_epochs:
            return

        desired = self._mc_curriculum_allowed_actions(epoch)
        # Filter against actual discovered MC actions if available
        all_mc_names = getattr(policy_obj, "mc_action_names", None)
        if isinstance(all_mc_names, list) and all_mc_names:
            desired = [name for name in desired if name in all_mc_names]

        try:
            # Prefer explicit component name used in MC policies. It's okay to hardcode.
            mc_embed = getattr(policy_obj, "components", {}).get("mc_action_embedding", None)
            if mc_embed is not None and hasattr(mc_embed, "mc_initialize_to_environment"):
                mc_embed.mc_initialize_to_environment(desired)
        except Exception:
            # Fail-safe: curriculum should never break training if structure differs
            pass
