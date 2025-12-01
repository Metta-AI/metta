# metta/rl/loss/crl_actor.py
"""Contrastive RL Actor Loss.

Trains the policy to maximize the CRL critic output, effectively teaching
the agent to reach commanded goals. The actor loss is:

max_θ E_{s,g~p, a~π(a|s,g)}[f(s, a, g)]

where f(s, a, g) = -||φ(s,a) - ψ(g)||₂ is the critic output.

This is a policy gradient method where the learned temporal distance
function serves as the reward signal.
"""

from typing import Any, Optional

import torch
from pydantic import Field
from tensordict import TensorDict
from torch import Tensor
from torchrl.data import Composite, UnboundedContinuous

from metta.agent.policy import Policy
from metta.rl.loss.crl_critic import CRLCritic
from metta.rl.loss.loss import Loss, LossConfig
from metta.rl.training import ComponentContext, TrainingEnvironment
from metta.rl.utils import prepare_policy_forward_td


class CRLActorConfig(LossConfig):
    """Configuration for CRL Actor Loss.

    Trains the policy to maximize the learned temporal distance function (critic).
    """

    # Actor training hyperparameters
    actor_coef: float = Field(
        default=1.0,
        ge=0,
        description="Coefficient for actor loss",
    )
    entropy_coef: float = Field(
        default=0.01,
        ge=0,
        description="Entropy regularization coefficient",
    )

    # Goal sampling for actor
    discount: float = Field(
        default=0.99,
        ge=0,
        lt=1,
        description="Discount for geometric goal sampling",
    )

    # Goal conditioning
    goal_dims: Optional[list[int]] = Field(
        default=None,
        description="Dimensions of state to use as goal",
    )

    # Hindsight experience replay (HER) style relabeling
    her_ratio: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="Fraction of goals to relabel with achieved future states",
    )

    def create(
        self,
        policy: Policy,
        trainer_cfg: Any,
        env: TrainingEnvironment,
        device: torch.device,
        instance_name: str,
        loss_config: Any,
    ) -> "CRLActorLoss":
        return CRLActorLoss(
            policy,
            trainer_cfg,
            env,
            device,
            instance_name=instance_name,
            loss_config=loss_config,
        )


class CRLActorLoss(Loss):
    """CRL Actor Loss.

    Trains the policy to maximize the CRL critic, using policy gradients
    with the critic output as the reward signal.

    This loss requires the CRLLoss to run first (to provide the critic).
    """

    __slots__ = (
        "actor_coef",
        "entropy_coef",
        "discount",
        "goal_dims",
        "her_ratio",
        "_state_dim",
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

        self.actor_coef = self.cfg.actor_coef
        self.entropy_coef = self.cfg.entropy_coef
        self.discount = self.cfg.discount
        self.goal_dims = self.cfg.goal_dims
        self.her_ratio = self.cfg.her_ratio

        # Infer state dimension
        obs_space = env.single_observation_space
        if hasattr(obs_space, "shape"):
            self._state_dim = int(torch.tensor(obs_space.shape).prod().item())
        else:
            raise ValueError("Cannot infer state dimension from observation space")

    def get_experience_spec(self) -> Composite:
        """Define experience spec (shares with CRL loss)."""
        # CRL loss already stores crl_obs, we don't need additional storage
        return Composite()

    def run_train(
        self,
        shared_loss_data: TensorDict,
        context: ComponentContext,
        mb_idx: int,
    ) -> tuple[Tensor, TensorDict, bool]:
        """Compute CRL actor loss.

        Uses the critic from CRLLoss to compute policy gradients.
        """
        # Get critic from shared data (set by CRLLoss)
        critic: Optional[CRLCritic] = shared_loss_data.get("crl_critic", None)
        if critic is None:
            # CRL loss hasn't run or isn't enabled
            self.loss_tracker["crl_actor_loss"].append(0.0)
            return self._zero(), shared_loss_data, False

        minibatch = shared_loss_data["sampled_mb"]
        policy_td = shared_loss_data.get("policy_td", None)

        if policy_td is None:
            # Need to forward the policy
            B, TT = minibatch.batch_size
            policy_forward_td, _, _ = prepare_policy_forward_td(
                minibatch, self.policy_experience_spec, clone=False
            )
            self.policy.reset_memory()
            policy_td = self.policy.forward(policy_forward_td)
            policy_td = policy_td.reshape(B, TT)
            shared_loss_data["policy_td"] = policy_td

        batch_shape = minibatch.batch_size
        if len(batch_shape) != 2:
            raise ValueError("CRL actor loss expects minibatch with 2D batch size")

        segments, horizon = batch_shape

        # Get observations
        if "crl_obs" in minibatch.keys():
            obs = minibatch["crl_obs"]  # (segments, horizon, state_dim)
        else:
            obs = minibatch["env_obs"].reshape(segments, horizon, -1).float()

        # Get actions from policy output
        actions = policy_td["actions"]  # (segments, horizon) or (segments, horizon, action_dim)
        if actions.dim() == 1:
            actions = actions.reshape(segments, horizon)
        elif actions.shape[0] == segments * horizon:
            actions = actions.reshape(segments, horizon, -1).squeeze(-1)

        # Get log probs and entropy for policy gradient
        log_probs = policy_td["act_log_prob"]
        if log_probs.dim() == 1:
            log_probs = log_probs.reshape(segments, horizon)
        elif log_probs.shape[0] == segments * horizon:
            log_probs = log_probs.reshape(segments, horizon)

        entropy = policy_td.get("entropy", None)

        # Get done mask
        dones = minibatch.get("dones", None)
        if dones is not None:
            dones = dones.squeeze(-1) if dones.dim() == 3 else dones
            done_mask = dones.to(dtype=torch.bool)
        else:
            done_mask = torch.zeros(segments, horizon, dtype=torch.bool, device=self.device)

        truncateds = minibatch.get("truncateds", None)
        if truncateds is not None:
            truncateds = truncateds.squeeze(-1) if truncateds.dim() == 3 else truncateds
            done_mask = torch.logical_or(done_mask, truncateds.to(dtype=torch.bool))

        # Compute actor loss
        actor_loss, metrics = self._compute_actor_loss(
            obs, actions, log_probs, done_mask, critic, segments, horizon
        )

        # Add entropy bonus
        if entropy is not None:
            entropy_loss = entropy.mean()
            actor_loss = actor_loss - self.entropy_coef * entropy_loss
            metrics["crl_entropy"] = entropy_loss.item()

        # Track metrics
        self.loss_tracker["crl_actor_loss"].append(float(actor_loss.item()))
        for key, value in metrics.items():
            if key not in self.loss_tracker:
                self.loss_tracker[key] = []
            self.loss_tracker[key].append(float(value))

        return actor_loss * self.actor_coef, shared_loss_data, False

    def _compute_actor_loss(
        self,
        obs: Tensor,
        actions: Tensor,
        log_probs: Tensor,
        done_mask: Tensor,
        critic: CRLCritic,
        segments: int,
        horizon: int,
    ) -> tuple[Tensor, dict]:
        """Compute policy gradient loss using CRL critic as reward.

        For each state, we sample a goal (future state) and compute:
        L = -E[log π(a|s) * Q(s,a,g)]

        where Q(s,a,g) = -||φ(s,a) - ψ(g)||₂ is the critic output.
        """
        # Sample goals for each state-action pair
        prob = max(1.0 - float(self.discount), 1e-8)
        geom_dist = torch.distributions.Geometric(
            probs=torch.tensor(prob, device=self.device, dtype=obs.dtype)
        )

        done_mask_cpu = done_mask.detach().cpu()

        # Collect valid (state, action, goal) triplets
        state_indices: list[tuple[int, int]] = []
        goal_indices: list[tuple[int, int]] = []

        for batch_idx in range(segments):
            done_row = done_mask_cpu[batch_idx].view(-1)
            episode_bounds: list[tuple[int, int]] = []
            start = 0

            for step, done in enumerate(done_row.tolist()):
                if done:
                    episode_bounds.append((start, step))
                    start = step + 1
            if start < horizon:
                episode_bounds.append((start, horizon - 1))

            for episode_start, episode_end in episode_bounds:
                if episode_end - episode_start < 1:
                    continue

                for anchor in range(episode_start, episode_end):
                    max_future = episode_end - anchor

                    # Sample delta
                    delta = int(geom_dist.sample().item())
                    if delta > max_future:
                        delta = max_future

                    goal_step = anchor + delta
                    state_indices.append((batch_idx, anchor))
                    goal_indices.append((batch_idx, goal_step))

        num_samples = len(state_indices)
        if num_samples < 1:
            return torch.tensor(0.0, device=self.device), {
                "crl_critic_values": 0.0,
                "crl_actor_samples": 0,
            }

        # Extract tensors
        batch_idx = torch.tensor([s[0] for s in state_indices], device=self.device, dtype=torch.long)
        step_idx = torch.tensor([s[1] for s in state_indices], device=self.device, dtype=torch.long)
        goal_batch_idx = torch.tensor([g[0] for g in goal_indices], device=self.device, dtype=torch.long)
        goal_step_idx = torch.tensor([g[1] for g in goal_indices], device=self.device, dtype=torch.long)

        states = obs[batch_idx, step_idx]  # (N, state_dim)
        sampled_actions = actions[batch_idx, step_idx]  # (N,) or (N, action_dim)
        goals = obs[goal_batch_idx, goal_step_idx]  # (N, state_dim)
        sampled_log_probs = log_probs[batch_idx, step_idx]  # (N,)

        # Apply goal_dims if specified
        if self.goal_dims is not None:
            goals = goals[..., self.goal_dims]

        # Compute critic values (these are our "rewards")
        with torch.no_grad():
            # critic.forward returns negative L2 distance (higher = better)
            critic_values = critic(states, sampled_actions, goals)  # (N,)

        # Policy gradient: minimize -log_prob * advantage
        # Here advantage ≈ Q(s,a,g) = critic_values
        # We want to maximize E[log π(a|s) * Q], so minimize -E[log π(a|s) * Q]
        # Normalize critic values for stability
        critic_values_normalized = (critic_values - critic_values.mean()) / (critic_values.std() + 1e-8)

        # Policy gradient loss
        pg_loss = -(sampled_log_probs * critic_values_normalized.detach()).mean()

        metrics = {
            "crl_critic_values": critic_values.mean().item(),
            "crl_actor_samples": num_samples,
        }

        return pg_loss, metrics
