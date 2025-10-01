from typing import Any

import einops
import numpy as np
import torch
import torch.nn.functional as F
from pydantic import Field
from tensordict import NonTensorData, TensorDict
from torch import Tensor
from torchrl.data import Composite, MultiCategorical, UnboundedContinuous

from metta.agent.policy import Policy
from metta.rl.advantage import compute_advantage, normalize_advantage_distributed
from metta.rl.loss import Loss
from metta.rl.training import ComponentContext, TrainingEnvironment
from mettagrid.config import Config


class CNAConfig(Config):
    """Config for Clipped Normalized Advantages (CNA) target, per Muesli.

    In Muesli, policy learning combines a regularized policy gradient with an
    MPO-style target derived from one-step lookahead Q estimates. Here we expose
    the temperature for the exponential weights and the clipping parameter used
    to normalize advantages for numerical stability.
    """

    temperature: float = Field(default=1.0, gt=0)
    clip_coef: float = Field(default=0.2, ge=0)


class MuesliVTraceConfig(Config):
    rho_clip: float = Field(default=1.0, gt=0)
    c_clip: float = Field(default=1.0, gt=0)


class MuesliConfig(Config):
    """Configuration for the Muesli loss in this codebase.

    This closely follows the PPO skeleton, while integrating:
    - CNA target for policy improvement using one-step lookahead values
    - Regularized policy gradient (entropy)
    - Value regression with optional clipping
    - Optional prioritized sampling by advantage magnitude (kept simple)
    - Auxiliary dynamics terms `returns_pred`/`reward_pred`
    """

    # Discounting and GAE
    gamma: float = Field(default=0.99, ge=0, le=1.0)
    gae_lambda: float = Field(default=0.95, ge=0, le=1.0)

    # Policy/value regularization
    ent_coef: float = Field(default=0.01, ge=0)
    vf_coef: float = Field(default=0.5, ge=0)
    vf_clip_coef: float = Field(default=0.1, ge=0)
    clip_vloss: bool = True

    # CNA/MPO-like weighting
    cna: CNAConfig = Field(default_factory=CNAConfig)

    # Normalize advantages
    norm_adv: bool = True

    # KL early stop (optional)
    target_kl: float | None = None

    # V-trace style importance sampling stabilization
    vtrace: MuesliVTraceConfig = Field(default_factory=MuesliVTraceConfig)

    # Dynamics auxiliary losses
    returns_step_look_ahead: int = Field(default=1)
    returns_pred_coef: float = Field(default=1.0, ge=0, le=1.0)
    reward_pred_coef: float = Field(default=1.0, ge=0, le=1.0)

    def create(
        self,
        policy: Policy,
        trainer_cfg: Any,
        env: TrainingEnvironment,
        device: torch.device,
        instance_name: str,
        loss_config: Any,
    ):
        return Muesli(
            policy,
            trainer_cfg,
            env,
            device,
            instance_name=instance_name,
            loss_config=loss_config,
        )


class Muesli(Loss):
    """Muesli loss with one-step lookahead and CNA-weighted policy update.

    Algorithm overview (based on Hessel et al. 2021):
    - We perform a direct policy optimization using a regularized objective.
    - Advantages are computed with V-trace style corrections for stability.
    - We build a CNA (Clipped Normalized Advantages) target by exponentiating and
      renormalizing advantages to form action weights (akin to MPO E-step weights).
    - The policy update mixes these weights with the standard policy gradient,
      yielding a robust objective that avoids brittle constrained optimization.
    - Additionally, Muesli leverages a simple one-step lookahead model to learn
      short-horizon value-improving signals. In this implementation, we require
      the policy to output: `returns_pred` (future return prediction) and
      `reward_pred` (next-step reward prediction). These predictions are trained
      with MSE losses against shifted targets in the sampled minibatch.

    Expectations from `Policy.forward(td)`:
    - Provide `values` (critic), `act_log_prob`, and `entropy`.
    - When called with `action`, compute log-prob for that action (as in PPO).
    - Provide `returns_pred` and `reward_pred` tensors for the dynamics auxiliary losses.
    """

    __slots__ = ("advantages",)

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

    # ---------- Experience spec ----------
    def get_experience_spec(self) -> Composite:
        """Augment base experience with standard RL fields used here.

        Matches your PPO spec so the CoreTrainingLoop can merge specs across losses.
        """
        act_space = self.env.single_action_space
        nvec = act_space.nvec
        act_dtype = torch.int32 if np.issubdtype(act_space.dtype, np.integer) else torch.float32
        scalar_f32 = UnboundedContinuous(shape=torch.Size([]), dtype=torch.float32)

        return Composite(
            rewards=scalar_f32,
            dones=scalar_f32,
            truncateds=scalar_f32,
            actions=MultiCategorical(nvec=nvec, dtype=act_dtype),
            act_log_prob=scalar_f32,
            values=scalar_f32,
        )

    # ---------- Rollout ----------
    def run_rollout(self, td: TensorDict, context: ComponentContext) -> None:
        with torch.no_grad():
            self.policy.forward(td)

        # Store to segmented replay
        env_slice = context.training_env_id
        if env_slice is None:
            raise RuntimeError("ComponentContext.training_env_id is required for Muesli rollout")
        self.replay.store(data_td=td, env_id=env_slice)

    # ---------- Train loop ----------
    def run_train(
        self, shared_loss_data: TensorDict, context: ComponentContext, mb_idx: int
    ) -> tuple[Tensor, TensorDict, bool]:
        cfg = self.loss_cfg
        stop_update_epoch = False

        # Early stop by KL across minibatches in an epoch
        if cfg.target_kl is not None and mb_idx > 0:
            avg_kl = torch.tensor(shared_loss_data.get("muesli_avg_kl", 0.0)).item()
            if avg_kl > cfg.target_kl:
                stop_update_epoch = True

        # Compute advantages on first mb of the update
        if mb_idx == 0:
            self.advantages = self._compute_bootstrap_advantages()

        # Sample minibatch segments (uniform here to keep simple)
        minibatch, indices = self._sample_minibatch_uniform()

        # Forward policy on sampled transitions with fixed actions for logprob.
        # Important: This asks the policy to compute the log-prob of the actions
        # drawn during rollout (off-policy compatible via IS ratio below).
        policy_td = minibatch.select(*self.policy_experience_spec.keys(include_nested=True))
        B, TT = policy_td.batch_size
        policy_td = policy_td.reshape(B * TT)
        policy_td.set("bptt", torch.full((B * TT,), TT, device=policy_td.device, dtype=torch.long))
        policy_td.set("batch", torch.full((B * TT,), B, device=policy_td.device, dtype=torch.long))

        flat_actions = minibatch["actions"].reshape(B * TT, -1)
        policy_td = self.policy.forward(policy_td, action=flat_actions)
        shared_loss_data["policy_td"] = policy_td.reshape(B, TT)
        shared_loss_data["sampled_mb"] = minibatch
        shared_loss_data["indices"] = NonTensorData(indices)

        # Compute total loss (policy + value + entropy + optional dynamics)
        loss, metrics = self._compute_losses(minibatch=minibatch, policy_td=policy_td)

        # Track and update replay with new critic values if available
        with torch.no_grad():
            new_values = policy_td["values"].view(minibatch["values"].shape)
        self.replay.update(indices, TensorDict({"values": new_values.detach()}, batch_size=minibatch.batch_size))

        # Update trackers
        for k, v in metrics.items():
            self.loss_tracker[k].append(float(v))

        # Share KL to enable target_kl stop
        shared_loss_data["muesli_avg_kl"] = metrics.get("approx_kl", 0.0)

        return loss, shared_loss_data, stop_update_epoch

    # ---------- Helper computations ----------
    def _compute_bootstrap_advantages(self) -> Tensor:
        """Compute initial advantages used to seed training for this epoch.

        We start from stored `values`, `rewards`, and `dones` in the replay
        buffer and compute generalized advantages with V-trace ratios set to 1
        (on-policy baseline). Subsequent minibatches refine via current policy
        ratios inside `_compute_losses`.
        """
        cfg = self.loss_cfg
        with torch.no_grad():
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
        return advantages

    def _sample_minibatch_uniform(self) -> tuple[TensorDict, Tensor]:
        """Uniformly sample segment rows from the segmented replay buffer.

        The buffer shape is (segments, horizon). We sample segment indices and
        take full sequences for BPTT-compatible updates. This matches your PPO
        experience handling, allowing shared infrastructure.
        """
        idx = torch.randint(
            low=0, high=self.replay.segments, size=(self.replay.minibatch_segments,), device=self.device
        )
        minibatch = self.replay.buffer[idx]
        # Attach returns/advantages from precomputed tensor for the sampled rows
        with torch.no_grad():
            minibatch["advantages"] = self.advantages[idx]
            minibatch["returns"] = minibatch["advantages"] + minibatch["values"]
        return minibatch.clone(), idx

    def _compute_losses(self, minibatch: TensorDict, policy_td: TensorDict) -> tuple[Tensor, dict[str, float]]:
        """Compute policy, value, entropy, and dynamics auxiliary losses.

        Policy: direct, CNA-weighted log-prob objective with entropy regularization.
        Value: clipped MSE against bootstrapped returns.
        Dynamics: MSE for future-returns and next-reward predictions (required).
        """
        cfg = self.loss_cfg

        old_logprob = minibatch["act_log_prob"]
        new_logprob = policy_td["act_log_prob"].reshape(old_logprob.shape)
        entropy = policy_td["entropy"]
        new_value = policy_td["values"]

        # Importance sampling ratio for off-policy correction (clamped for stability)
        # ratio = pi_new(a|s) / pi_old(a|s). In log-space we clamp differences to
        # avoid extreme exponentiation which can cause numerical issues.
        logratio = torch.clamp(new_logprob - old_logprob, -10, 10)
        ratio = logratio.exp()

        # Retrace(λ) advantages for the PG baseline (closer to Muesli paper):
        # A^ret_t = Q^ret_t - V_t, where Q^ret uses truncated IS ratios c_t = λ * min(1, ρ_t)
        adv = self._compute_retrace_advantages(minibatch, logratio)
        adv = normalize_advantage_distributed(adv, cfg.norm_adv)

        # CNA weights from one-step model-based lookahead Q estimates:
        # Q̂_t(a_t) = reward_pred_t + γ V_{t+1}. Advantages Â_t = Q̂_t - V_t.
        # We then compute scalar weights w_t = exp(clip(Â_t)/T), normalized by batch mean.
        returns_shape = minibatch["returns"].shape  # (B, T)
        B, TT = returns_shape
        V_t = new_value.view(returns_shape)
        # V_{t+1} with terminal handling: zero out next when done
        V_next = torch.zeros_like(V_t)
        V_next[:, :-1] = V_t[:, 1:]
        done = minibatch["dones"].to(dtype=V_t.dtype)
        V_next = V_next * (1.0 - done)

        reward_pred_bt = einops.rearrange(policy_td["reward_pred"].to(dtype=torch.float32), "bt 1 -> bt").view(B, TT)
        Q_hat = reward_pred_bt + self.loss_cfg.gamma * V_next
        A_cna = Q_hat - V_t
        A_cna = torch.clamp(A_cna, -cfg.cna.clip_coef, cfg.cna.clip_coef)
        weights = torch.exp(A_cna / cfg.cna.temperature)
        weights = weights / (weights.mean().detach() + 1e-8)

        # Policy gradient with CNA weighting: -E[w * log pi(a|s)].
        # We detach the weights so they act as targets rather than gradients
        # flowing back through the weighting mechanism.
        pg_loss = -(weights.detach().reshape(new_logprob.shape) * new_logprob).mean()

        # Critic loss with clipping similar to PPO: this stabilizes updates when
        # the new value deviates too far from the old estimate.
        returns = minibatch["returns"]
        old_values = minibatch["values"]
        new_value_reshaped = new_value.view(returns.shape)
        if cfg.clip_vloss:
            v_loss_unclipped = (new_value_reshaped - returns) ** 2
            v_clipped = old_values + torch.clamp(new_value_reshaped - old_values, -cfg.vf_clip_coef, cfg.vf_clip_coef)
            v_loss_clipped = (v_clipped - returns) ** 2
            v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
        else:
            v_loss = 0.5 * ((new_value_reshaped - returns) ** 2).mean()

        # Entropy regularization
        entropy_loss = entropy.mean()

        total_loss = pg_loss - cfg.ent_coef * entropy_loss + cfg.vf_coef * v_loss

        # Dynamics auxiliary losses (required by this implementation):
        # We require the policy to output two model heads:
        # - returns_pred: prediction of returns at t+look_ahead
        # - reward_pred: prediction of reward at t+1
        # Both are aligned against shifted targets sampled from the buffer.
        # This encourages learning short-horizon value structure as in Muesli's
        # one-step lookahead. See fast_dynamics policy for simple heads.
        assert "returns_pred" in policy_td.keys(), "Policy must output `returns_pred` for Muesli dynamics."
        assert "reward_pred" in policy_td.keys(), "Policy must output `reward_pred` for Muesli dynamics."

        B, TT = minibatch.batch_size
        returns_pred = policy_td["returns_pred"]
        reward_pred = policy_td["reward_pred"]

        # Flatten possible trailing unit dims from TDM heads: (BT, 1) -> (BT)
        returns_pred = einops.rearrange(returns_pred.to(dtype=torch.float32), "bt 1 -> bt")
        reward_pred = einops.rearrange(reward_pred.to(dtype=torch.float32), "bt 1 -> bt")

        future_returns = minibatch["returns"].reshape(B * TT)
        rewards_flat = minibatch["rewards"].reshape(B * TT)

        # Predict returns at t + look_ahead (>=1)
        look = max(1, int(cfg.returns_step_look_ahead))
        if future_returns.shape[0] > look:
            total_loss = total_loss + cfg.returns_pred_coef * F.mse_loss(returns_pred[:-look], future_returns[look:])

        # Predict reward at t + 1
        if rewards_flat.shape[0] > 1:
            total_loss = total_loss + cfg.reward_pred_coef * F.mse_loss(reward_pred[:-1], rewards_flat[1:])

        # Metrics (report as floats)
        with torch.no_grad():
            approx_kl = ((ratio - 1) - (new_logprob - old_logprob)).mean().item()
            clipfrac = ((ratio - 1.0).abs() > cfg.cna.clip_coef).float().mean().item()

        metrics = {
            "policy_loss": float(pg_loss.item()),
            "value_loss": float(v_loss.item()),
            "entropy": float(entropy_loss.item()),
            "approx_kl": float(approx_kl),
            "clipfrac": float(clipfrac),
            "importance": float(ratio.mean().item()),
            "current_logprobs": float(new_logprob.mean().item()),
        }

        return total_loss, metrics

    def _compute_retrace_advantages(self, minibatch: TensorDict, logratio: Tensor) -> Tensor:
        """Compute Retrace(λ) advantages A_t = Q^ret_t - V_t per segment.

        Implements: Q^ret_t = V_t + Σ_{n>=t} γ^{n-t} (Π_{i=t+1..n} c_i) δ_n,
        where δ_n = r_n + γ V_{n+1} - V_n and c_i = λ * min(1, ρ_i).

        We operate on shape (B, T) by reshaping from flattened inputs.
        """
        gamma = self.loss_cfg.gamma
        lam = self.loss_cfg.gae_lambda

        B, TT = minibatch.batch_size
        V = minibatch["values"].view(B, TT)
        r = minibatch["rewards"].view(B, TT)
        d = minibatch["dones"].view(B, TT).to(dtype=V.dtype)

        # ρ_t = exp(clamped logratio); c_t = λ * min(1, ρ_t)
        rho = logratio.exp().view(B, TT)
        c = lam * torch.clamp(rho, max=1.0)

        # V_{t+1} with terminal handling
        V_next = torch.zeros_like(V)
        V_next[:, :-1] = V[:, 1:]
        V_next = V_next * (1.0 - d)

        # TD residuals δ_t
        delta = r + gamma * V_next - V

        # Backward accumulation for Retrace
        Q_ret_minus_V = torch.zeros_like(V)
        acc = torch.zeros(B, device=V.device, dtype=V.dtype)
        for t in reversed(range(TT)):
            acc = delta[:, t] + (gamma * c[:, t]) * acc
            Q_ret_minus_V[:, t] = acc

        return Q_ret_minus_V
