from typing import Any, Tuple

import numpy as np
import torch
from pydantic import Field
from tensordict import NonTensorData, TensorDict
from torch import Tensor
from torchrl.data import Composite, UnboundedContinuous, UnboundedDiscrete

from metta.agent.policy import Policy
from metta.rl.advantage import compute_advantage, normalize_advantage_distributed
from metta.rl.loss.loss import Loss, LossConfig
from metta.rl.loss.replay_samplers import sequential_sample
from metta.rl.training import ComponentContext, TrainingEnvironment
from metta.rl.training.batch import calculate_prioritized_sampling_params
from metta.rl.utils import prepare_policy_forward_td
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


class PPOConfig(LossConfig):
    # PPO hyperparameters
    # Clip coefficient (0.1-0.3 typical; Schulman et al. 2017)
    clip_coef: float = Field(default=0.264407, gt=0, le=1.0)
    # Entropy term weight from sweep
    ent_coef: float = Field(default=0.010000, ge=0)
    # GAE lambda tuned via sweep (cf. standard 0.95)
    gae_lambda: float = Field(default=0.891477, ge=0, le=1.0)
    # Gamma tuned for shorter effective horizon
    gamma: float = Field(default=0.977, ge=0, le=1.0)

    # Training parameters
    # Value clipping mirrors policy clip
    vf_clip_coef: float = Field(default=0.1, ge=0)
    # Value term weight from sweep
    vf_coef: float = Field(default=0.897619, ge=0)

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
    ) -> "PPO":
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
            is_student_agent=UnboundedContinuous(shape=torch.Size([]), dtype=torch.float32),
        )

    def run_rollout(self, td: TensorDict, context: ComponentContext) -> None:
        dual_enabled = bool(getattr(context, "dual_policy_enabled", False))
        npc_input_td: TensorDict | None = None

        if dual_enabled:
            npc_input_td = td.select(
                *self.policy_experience_spec.keys(include_nested=True),
                strict=False,
            ).clone()

            # Preserve rollout metadata required by recurrent policies
            for meta_key in ("bptt", "batch", "training_env_ids"):
                if meta_key in td.keys(include_nested=True):
                    npc_input_td.set(meta_key, td.get(meta_key).clone())

        with torch.no_grad():
            self.policy.forward(td)

        if dual_enabled and npc_input_td is not None:
            self._inject_dual_policy_outputs(td, npc_input_td, context)
        else:
            actions_tensor = td.get("actions")
            if not isinstance(actions_tensor, torch.Tensor):
                raise RuntimeError("Policy must populate 'actions' tensor during rollout")
            td.set(
                "is_student_agent",
                torch.ones(actions_tensor.shape, device=actions_tensor.device, dtype=torch.float32),
            )

        if self.burn_in_steps_iter < self.burn_in_steps:
            self.burn_in_steps_iter += 1
            return

        # Store experience
        env_slice = context.training_env_id
        if env_slice is None:
            raise RuntimeError("ComponentContext.training_env_id is missing in rollout.")
        self.replay.store(data_td=td, env_id=env_slice)

        return

    def _inject_dual_policy_outputs(
        self,
        rollout_td: TensorDict,
        npc_input_td: TensorDict,
        context: ComponentContext,
    ) -> None:
        npc_policy = getattr(context, "npc_policy", None)
        npc_mask_per_env = getattr(context, "npc_mask_per_env", None)
        student_mask_per_env = getattr(context, "student_mask_per_env", None)

        actions = rollout_td.get("actions")
        if not isinstance(actions, torch.Tensor):
            raise RuntimeError("Policy must populate 'actions' tensor during rollout")

        if npc_policy is None or npc_mask_per_env is None or student_mask_per_env is None:
            rollout_td.set(
                "is_student_agent",
                torch.ones(actions.shape, device=actions.device, dtype=torch.float32),
            )
            return

        agents_per_env = npc_mask_per_env.numel()
        if agents_per_env == 0:
            rollout_td.set(
                "is_student_agent",
                torch.ones(actions.shape, device=actions.device, dtype=torch.float32),
            )
            return

        total_agents = actions.shape[0]
        if total_agents % agents_per_env != 0:
            rollout_td.set(
                "is_student_agent",
                torch.ones(actions.shape, device=actions.device, dtype=torch.float32),
            )
            return

        num_envs = total_agents // agents_per_env

        npc_td = npc_input_td.to(device=actions.device)
        with torch.no_grad():
            npc_policy.forward(npc_td)

        npc_mask = npc_mask_per_env.to(device=actions.device).repeat(num_envs)
        student_mask = student_mask_per_env.to(device=actions.device).repeat(num_envs)
        rollout_td.set("is_student_agent", student_mask.to(dtype=torch.float32))

        keys_to_update = ["actions", "act_log_prob", "entropy", "values", "full_log_probs"]
        for key in keys_to_update:
            if not (key in npc_td.keys() and key in rollout_td.keys()):
                continue

            source = npc_td.get(key)
            target = rollout_td.get(key)
            if not isinstance(source, torch.Tensor) or not isinstance(target, torch.Tensor):
                continue
            if source.shape != target.shape:
                continue

            source = source.to(device=target.device, dtype=target.dtype)
            updated = target.clone()
            updated[npc_mask] = source[npc_mask]
            rollout_td.set(key, updated)

    def run_train(
        self, shared_loss_data: TensorDict, context: ComponentContext, mb_idx: int
    ) -> tuple[Tensor, TensorDict, bool]:
        """This is the PPO algorithm training loop."""
        config = self.cfg
        stop_update_epoch = False
        self.burn_in_steps_iter = 0
        if config.target_kl is not None and mb_idx > 0:
            avg_kl = np.mean(self.loss_tracker["approx_kl"]) if self.loss_tracker["approx_kl"] else 0.0
            if avg_kl > config.target_kl:
                stop_update_epoch = True

        # On the first minibatch of the update epoch, compute advantages and sampling params
        if mb_idx == 0:
            self.advantages, self.anneal_beta = self._on_first_mb(context)

        # Then sample from the buffer (this happens at every minibatch)
        minibatch, indices, prio_weights = self._sample_minibatch(
            advantages=self.advantages,
            prio_alpha=config.prioritized_experience_replay.prio_alpha,
            prio_beta=self.anneal_beta,
            mb_idx=mb_idx,
        )

        shared_loss_data["sampled_mb"] = minibatch  # one loss should write the sampled mb for others to use
        shared_loss_data["indices"] = NonTensorData(indices)  # av this breaks compile

        # Then forward the policy using the sampled minibatch
        policy_td, B, TT = prepare_policy_forward_td(minibatch, self.policy_experience_spec, clone=False)

        flat_actions = minibatch["actions"].reshape(B * TT, -1)
        self.policy.reset_memory()
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
        super().on_train_phase_end(context)

    def _on_first_mb(self, context: ComponentContext) -> tuple[Tensor, float]:
        # reset importance sampling ratio
        if "ratio" in self.replay.buffer.keys():
            self.replay.buffer["ratio"].fill_(1.0)

        cfg = self.cfg
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
        cfg = self.cfg
        old_logprob = minibatch["act_log_prob"]
        new_logprob = policy_td["act_log_prob"].reshape(old_logprob.shape)
        entropy = policy_td["entropy"]
        newvalue = policy_td["values"]

        importance_sampling_ratio = self._importance_ratio(new_logprob, old_logprob)

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

        dual_cfg = getattr(self.trainer_cfg, "dual_policy", None)
        student_mask_tensor = minibatch.get("is_student_agent")
        use_dual_policy = bool(dual_cfg and getattr(dual_cfg, "enabled", False) and student_mask_tensor is not None)

        metric_importance_ratio: Tensor
        metric_new_logprob: Tensor

        if use_dual_policy:
            student_mask_bool = (student_mask_tensor.reshape(old_logprob.shape) > 0.5).to(dtype=torch.bool)

            if torch.all(student_mask_bool):
                use_dual_policy = False
            else:
                student_mask_flat = student_mask_bool.reshape(-1)
                student_indices = torch.nonzero(student_mask_flat, as_tuple=False).squeeze(-1)

                if student_indices.numel() == 0:
                    return torch.zeros((), device=self.device, dtype=torch.float32)

                old_logprob_flat = old_logprob.reshape(-1)
                new_logprob_flat = new_logprob.reshape(-1)
                entropy_flat = entropy.reshape(-1)
                adv_flat = adv.reshape(-1)
                importance_flat = importance_sampling_ratio.reshape(-1)
                newvalue_flat = newvalue.reshape(-1)
                returns_flat = minibatch["returns"].reshape(-1)
                values_flat = minibatch["values"].reshape(-1)

                student_old_logprob = old_logprob_flat[student_indices]
                student_new_logprob = new_logprob_flat[student_indices]
                student_entropy = entropy_flat[student_indices]
                student_adv = adv_flat[student_indices]
                student_importance = importance_flat[student_indices]
                student_newvalue = newvalue_flat[student_indices]
                student_returns = returns_flat[student_indices]
                student_values = values_flat[student_indices]

                student_minibatch = TensorDict(
                    {
                        "act_log_prob": student_old_logprob,
                        "values": student_values,
                        "returns": student_returns,
                    },
                    batch_size=[student_indices.shape[0]],
                )

                pg_loss, v_loss, entropy_loss, approx_kl, clipfrac = self.compute_ppo_losses(
                    student_minibatch,
                    student_new_logprob,
                    student_entropy,
                    student_newvalue,
                    student_importance,
                    student_adv,
                )

                metric_importance_ratio = student_importance
                metric_new_logprob = student_new_logprob

        if not use_dual_policy:
            pg_loss, v_loss, entropy_loss, approx_kl, clipfrac = self.compute_ppo_losses(
                minibatch,
                new_logprob,
                entropy,
                newvalue,
                importance_sampling_ratio,
                adv,
            )
            metric_importance_ratio = importance_sampling_ratio
            metric_new_logprob = new_logprob

        loss = pg_loss - cfg.ent_coef * entropy_loss + v_loss * cfg.vf_coef

        # Update values and ratio in experience buffer
        update_td = TensorDict(
            {"values": newvalue.view(minibatch["values"].shape).detach(), "ratio": importance_sampling_ratio.detach()},
            batch_size=minibatch.batch_size,
        )
        self.replay.update(indices, update_td)

        # Update loss tracking
        self._track("policy_loss", pg_loss)
        self._track("value_loss", v_loss)
        self._track("entropy", entropy_loss)
        self._track("approx_kl", approx_kl)
        self._track("clipfrac", clipfrac)
        self._track("importance", metric_importance_ratio.mean())
        self._track("current_logprobs", metric_new_logprob.mean())

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
            1 - self.cfg.clip_coef,
            1 + self.cfg.clip_coef,
        )
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()
        returns = minibatch["returns"]
        old_values = minibatch["values"]

        # Value loss
        newvalue_reshaped = newvalue.view(returns.shape)
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

        entropy_loss = entropy.mean()

        # Compute metrics
        with torch.no_grad():
            logratio = new_logprob - minibatch["act_log_prob"]
            approx_kl = ((importance_sampling_ratio - 1) - logratio).mean()
            clipfrac = ((importance_sampling_ratio - 1.0).abs() > self.cfg.clip_coef).float().mean()

        return pg_loss, v_loss, entropy_loss, approx_kl, clipfrac

    def _sample_minibatch(
        self,
        advantages: Tensor,
        prio_alpha: float,
        prio_beta: float,
        mb_idx: int,
    ) -> tuple[TensorDict, Tensor, Tensor]:
        """Sample a prioritized minibatch."""
        if prio_alpha <= 0.0:
            # Deterministic sequential sampling when alpha == 0
            minibatch, idx = sequential_sample(self.replay, mb_idx)
            with torch.no_grad():
                minibatch["advantages"] = advantages[idx]
                minibatch["returns"] = advantages[idx] + minibatch["values"]
                prio_weights = torch.ones(
                    (idx.shape[0], 1),
                    device=minibatch.device,
                    dtype=minibatch["values"].dtype,
                )
            return minibatch, idx, prio_weights

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
