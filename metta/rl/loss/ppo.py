"""PPO loss implementation."""

from typing import Any, Literal, Tuple

import numpy as np
import torch
from pydantic import Field
from tensordict import TensorDict
from torch import Tensor
from torchrl.data import Composite, MultiCategorical, UnboundedContinuous

from metta.agent.metta_agent import PolicyAgent
from metta.mettagrid.config import Config
from metta.rl.advantage import compute_advantage, normalize_advantage_distributed
from metta.rl.loss.loss import Loss
from metta.rl.trainer_state import TrainerState

# Config classes


class OptimizerConfig(Config):
    type: Literal["adam", "muon"] = "adam"
    # Learning rate: Type 2 default chosen by sweep
    learning_rate: float = Field(default=0.000457, gt=0, le=1.0)
    # Beta1: Standard Adam default from Kingma & Ba (2014) "Adam: A Method for Stochastic Optimization"
    beta1: float = Field(default=0.9, ge=0, le=1.0)
    # Beta2: Standard Adam default from Kingma & Ba (2014)
    beta2: float = Field(default=0.999, ge=0, le=1.0)
    # Epsilon: Type 2 default chosen arbitrarily
    eps: float = Field(default=1e-12, gt=0)
    # Weight decay: Disabled by default, common practice for RL to avoid over-regularization
    weight_decay: float = Field(default=0, ge=0)


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
    clip_coef: float = Field(default=0.1, gt=0, le=1.0)
    # Entropy coefficient: Type 2 default chosen from sweep
    ent_coef: float = Field(default=0.0021, ge=0)
    # GAE lambda: Type 2 default chosen from sweep, deviates from typical 0.95, bias/variance tradeoff
    gae_lambda: float = Field(default=0.916, ge=0, le=1.0)
    # Gamma: Type 2 default chosen from sweep, deviates from typical 0.99, suggests shorter
    # effective horizon for multi-agent
    gamma: float = Field(default=0.977, ge=0, le=1.0)

    # Training parameters
    # Gradient clipping: 0.5 is standard PPO default to prevent instability
    max_grad_norm: float = Field(default=0.5, gt=0)
    # Value function clipping: Matches policy clip for consistency
    vf_clip_coef: float = Field(default=0.1, ge=0)
    # Value coefficient: Type 2 default chosen from sweep, balances policy vs value loss
    vf_coef: float = Field(default=0.44, ge=0)
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

    vtrace: VTraceConfig = Field(default_factory=VTraceConfig)

    prioritized_experience_replay: PrioritizedExperienceReplayConfig = Field(
        default_factory=PrioritizedExperienceReplayConfig
    )

    def create(
        self,
        policy: PolicyAgent,
        trainer_cfg: Any,
        vec_env: Any,
        device: torch.device,
        instance_name: str,
        loss_config: Any,
    ):
        """Create PPO loss instance."""
        return PPO(
            policy,
            trainer_cfg,
            vec_env,
            device,
            instance_name=instance_name,
            loss_config=loss_config,
        )


# Loss class


class PPO(Loss):
    """This could be slightly faster by looking for repeated access to hashed vars."""

    __slots__ = (
        "advantages",
        "anneal_beta",
    )

    def __init__(
        self,
        policy: PolicyAgent,
        trainer_cfg: Any,
        vec_env: Any,
        device: torch.device,
        instance_name: str,
        loss_config: Any,
    ):
        super().__init__(policy, trainer_cfg, vec_env, device, instance_name, loss_config)
        self.advantages = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self.anneal_beta = 0.0

    def get_experience_spec(self) -> Composite:
        act_space = self.vec_env.single_action_space
        nvec = act_space.nvec
        if isinstance(nvec, np.ndarray):
            nvec = nvec.tolist()
        elif isinstance(nvec, Tensor):
            nvec = nvec.tolist()

        scalar_f32 = UnboundedContinuous(shape=(), dtype=torch.float32)

        return Composite(
            rewards=scalar_f32,
            dones=scalar_f32,
            truncateds=scalar_f32,
            actions=MultiCategorical(
                nvec=nvec,
                dtype=torch.int64,
            ),
            act_log_prob=scalar_f32,
            values=scalar_f32,
            advantages=scalar_f32,
            ratio=scalar_f32,
            returns=scalar_f32,
        )

    # Loss calls this method
    def run_rollout(self, experience: TensorDict, trainer_state: TrainerState) -> None:
        # V0 has been placed in the memory already
        self.advantages = compute_advantage(
            V0=experience["values"],
            rewards=experience["rewards"],
            dones=experience["dones"],
            ratio=torch.tensor(1.0, dtype=torch.float32, device=self.device),  # initial ratio
            advantages=torch.tensor(0.0, dtype=torch.float32, device=self.device),  # advantages don't carry over
            gamma=self.loss_cfg.gamma,
            lmbda=self.loss_cfg.gae_lambda,
            rho_clip=self.trainer_cfg.vtrace.vtrace_rho_clip,
            c_clip=self.trainer_cfg.vtrace.vtrace_c_clip,
            device=self.device,
        )
        experience.set("advantages", self.advantages, inplace=True)
        experience.set("returns", self.advantages + experience["values"], inplace=True)
        experience.set("ratio", torch.ones_like(self.advantages, device=self.device, dtype=torch.float32))

    # Loss calls this method
    def run_train(self, shared_loss_data: TensorDict, trainer_state: TrainerState) -> tuple[Tensor, TensorDict]:
        indices = shared_loss_data["indices"]
        prio_weights = shared_loss_data["prio_weights"]
        sampled_mb = shared_loss_data["sampled_mb"]
        policy_td = shared_loss_data["policy_td"]

        # Re-compute advantages with new ratios (V-trace)
        if "ratio" in sampled_mb and self.trainer_cfg.vtrace:
            # recalculate advantage w/ V-trace
            with torch.no_grad():
                self.advantages = compute_advantage(
                    sampled_mb["values"],
                    sampled_mb["rewards"],
                    sampled_mb["dones"],
                    sampled_mb["ratio"],
                    sampled_mb["advantages"],
                    self.loss_cfg.gamma,
                    self.loss_cfg.gae_lambda,
                    self.trainer_cfg.vtrace.vtrace_rho_clip,
                    self.trainer_cfg.vtrace.vtrace_c_clip,
                    self.device,
                )
        else:
            self.advantages = sampled_mb["advantages"]

        # Normalize advantages with distributed support
        if self.loss_cfg.norm_adv:
            self.advantages = normalize_advantage_distributed(self.advantages, self.loss_cfg.norm_adv)

        # Apply prioritized weights
        self.advantages = prio_weights * self.advantages

        # Re-compute returns for value loss
        sampled_mb["returns"] = self.advantages + sampled_mb["values"]

        # TODO: if we want an annealing beta, it should go here

        # run policy using policy_td
        # NOTE: metta.agent expects the last dimension to be the logits for the value head.
        # Hence, we reshape value to have shape (B, T, 1) to match the output of the value head.
        old_act_log_prob = sampled_mb["act_log_prob"]
        policy_td = self.policy(policy_td, action=sampled_mb["actions"])
        new_logprob = policy_td["act_log_prob"].reshape(old_act_log_prob.shape)
        entropy = policy_td["entropy"]
        newvalue = policy_td["value"].reshape(sampled_mb["values"].shape)

        logratio = new_logprob - old_act_log_prob
        importance_sampling_ratio = logratio.exp()

        # Policy loss
        if self.loss_cfg.norm_adv:
            norm_adv = self.advantages / (self.advantages.var() + 1e-8).sqrt()
        else:
            norm_adv = self.advantages
        pg_loss1 = -norm_adv * importance_sampling_ratio
        pg_loss2 = -norm_adv * torch.clamp(
            importance_sampling_ratio, 1 - self.loss_cfg.clip_coef, 1 + self.loss_cfg.clip_coef
        )
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Value loss
        newvalue = newvalue.view(-1)
        # This has been extensively benchmarked to help with stability in PvP.
        # For details, see https://github.com/vwxyzjn/ppo-implementation-details/blob/main/ppo_implementation_details.ipynb
        if self.loss_cfg.clip_vloss:
            v_loss_unclipped = (newvalue - sampled_mb["returns"]) ** 2
            v_clipped = sampled_mb["values"] + torch.clamp(
                newvalue - sampled_mb["values"],
                -self.loss_cfg.vf_clip_coef,
                self.loss_cfg.vf_clip_coef,
            )
            v_loss_clipped = (v_clipped - sampled_mb["returns"]) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()
        else:
            v_loss = 0.5 * ((newvalue - sampled_mb["returns"]) ** 2).mean()

        # Entropy loss
        entropy_loss = entropy.mean()

        # L2 init loss
        # TODO: enable. There's a bug that's breaking this
        # l2_init_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        # if self.trainer_cfg.ppo.l2_init_loss_coef > 0:
        #     l2_init_loss = self.trainer_cfg.ppo.l2_init_loss_coef * self.policy.l2_init_loss().to(self.device)
        l2_init_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)

        # L2 regularization loss (turned off in a code sweep)
        l2_reg_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        if self.loss_cfg.l2_reg_loss_coef > 0.0:
            for param in self.policy.parameters():
                l2_reg_loss += param.norm(2)
            l2_reg_loss *= self.loss_cfg.l2_reg_loss_coef

        # Total loss
        loss = (
            pg_loss
            - self.loss_cfg.ent_coef * entropy_loss
            + v_loss * self.loss_cfg.vf_coef
            + l2_init_loss
            + l2_reg_loss
        )

        # Update values and ratio in experience buffer
        update_td = TensorDict(
            {"values": newvalue.view(sampled_mb["values"].shape).detach(), "ratio": importance_sampling_ratio.detach()},
            batch_size=sampled_mb.batch_size,
        )
        # we have to use the object directly, not the sampled_mb object
        self.experience.update(indices, update_td)

        # Update loss tracking
        with torch.no_grad():
            logratio = new_logprob - sampled_mb["act_log_prob"]
            approx_kl = ((importance_sampling_ratio - 1) - logratio).mean()
            clipfrac = ((importance_sampling_ratio - 1.0).abs() > self.loss_cfg.clip_coef).float().mean()

        # Update with the tracking version
        self.loss_tracker["policy_loss"].append(float(pg_loss.item()))
        self.loss_tracker["value_loss"].append(float(v_loss.item()))
        self.loss_tracker["entropy_loss"].append(float(entropy_loss.item()))
        self.loss_tracker["approx_kl"].append(float(approx_kl.item()))
        self.loss_tracker["clipfrac"].append(float(clipfrac.item()))
        self.loss_tracker["l2_init_loss"].append(float(l2_init_loss.item()))
        self.loss_tracker["l2_reg_loss"].append(float(l2_reg_loss.item()))
        self.loss_tracker["importance"].append(float(importance_sampling_ratio.mean().item()))

        # av: save some extra data for downstream use, possibly with prioritized experience replay
        if sampled_mb.get("priorities", None) is not None:
            td_errors = (sampled_mb["returns"] - sampled_mb["values"]).abs()
            _, indices_np = indices.data.cpu().numpy(), td_errors.detach().cpu().numpy()
            shared_loss_data["td_errors"] = td_errors
            shared_loss_data["indices"] = indices_np

        return loss, shared_loss_data


def compute_ppo_losses(
    minibatch: TensorDict,
    new_logprob: Tensor,
    entropy: Tensor,
    newvalue: Tensor,
    importance_sampling_ratio: Tensor,
    adv: Tensor,
    trainer_cfg: Any,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Compute PPO losses.

    Returns:
        Tuple of (policy_loss, value_loss, entropy_loss, approx_kl, clipfrac)
    """
    # Policy loss
    if trainer_cfg.ppo.norm_adv:
        norm_adv = adv / (adv.var() + 1e-8).sqrt()
    else:
        norm_adv = adv
    pg_loss1 = -norm_adv * importance_sampling_ratio
    pg_loss2 = -norm_adv * torch.clamp(
        importance_sampling_ratio, 1 - trainer_cfg.ppo.clip_coef, 1 + trainer_cfg.ppo.clip_coef
    )
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

    # Value loss
    newvalue = newvalue.view(-1)
    if trainer_cfg.ppo.clip_vloss:
        v_loss_unclipped = (newvalue - minibatch["returns"]) ** 2
        v_clipped = minibatch["values"] + torch.clamp(
            newvalue - minibatch["values"],
            -trainer_cfg.ppo.vf_clip_coef,
            trainer_cfg.ppo.vf_clip_coef,
        )
        v_loss_clipped = (v_clipped - minibatch["returns"]) ** 2
        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
        v_loss = 0.5 * v_loss_max.mean()
    else:
        v_loss = 0.5 * ((newvalue - minibatch["returns"]) ** 2).mean()

    entropy_loss = entropy.mean()

    # Compute metrics
    with torch.no_grad():
        logratio = new_logprob - minibatch["act_log_prob"]
        approx_kl = ((importance_sampling_ratio - 1) - logratio).mean()
        clipfrac = ((importance_sampling_ratio - 1.0).abs() > trainer_cfg.ppo.clip_coef).float().mean()

    return pg_loss, v_loss, entropy_loss, approx_kl, clipfrac
