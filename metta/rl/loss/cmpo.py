"""Conservative Model-Based Policy Optimization (CMPO) loss."""

from __future__ import annotations

import copy
import random
from collections import deque
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import Field
from tensordict import TensorDict
from torch import Tensor
from torchrl.data import Composite, UnboundedContinuous, UnboundedDiscrete

from metta.agent.policy import Policy
from metta.rl.loss.loss import Loss, LossConfig
from metta.rl.training import ComponentContext, Experience, TrainingEnvironment
from metta.rl.utils import add_dummy_loss_for_unused_params, ensure_sequence_metadata, forward_policy_for_training
from mettagrid.base_config import Config


def _build_mlp(input_dim: int, hidden_dims: list[int], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = input_dim
    for hidden in hidden_dims:
        layers.append(nn.Linear(last_dim, hidden))
        layers.append(nn.LayerNorm(hidden))
        layers.append(nn.ReLU())
        last_dim = hidden
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


class WorldModelConfig(Config):
    ensemble_size: int = Field(default=5, ge=1)
    hidden_dims: list[int] = Field(default_factory=lambda: [512, 512])
    learning_rate: float = Field(default=3e-4, gt=0)
    batch_size: int = Field(default=256, gt=0)
    train_steps: int = Field(default=50, ge=1)
    buffer_size: int = Field(default=50000, gt=1)
    warmup_transitions: int = Field(default=2048, ge=0)
    gradient_clip: float = Field(default=1.0, gt=0)


class CMPOConfig(LossConfig):
    # CMPO policy parameters
    temperature: float = Field(default=1.0, gt=0)
    adv_clip: float = Field(default=10.0, gt=0)
    norm_adv: bool = True
    ent_coef: float = Field(default=0.0, ge=0)
    vf_coef: float = Field(default=0.5, ge=0)
    target_kl: Optional[float] = None

    # Prior policy via EMA (None disables)
    prior_ema_decay: Optional[float] = Field(default=None, ge=0, le=1.0)

    # Cap for (state, action) pairs processed at once in CMPO Q-value computation.
    # Default: match PPO minibatch size to keep memory in line with PPO runs.
    q_value_batch_size: Optional[int] = Field(default=None, gt=0)

    # Fraction of minibatch segments to train CMPO on (reduces compute for large runs).
    minibatch_fraction: float = Field(default=1.0, ge=0.0, le=1.0)

    world_model: WorldModelConfig = Field(default_factory=WorldModelConfig)

    def create(
        self,
        policy: Policy,
        trainer_cfg: Any,
        env: TrainingEnvironment,
        device: torch.device,
        instance_name: str,
    ) -> "CMPO":
        return CMPO(policy, trainer_cfg, env, device, instance_name, self)


class FeedForwardDynamics(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list[int]) -> None:
        super().__init__()
        self.net = _build_mlp(state_dim + action_dim, hidden_dims, 2 * state_dim + 1)
        self.state_dim = state_dim

    def forward(self, state: Tensor, action: Tensor) -> tuple[Tensor, Tensor]:
        x = torch.cat([state, action], dim=-1)
        output = self.net(x)
        delta = output[..., : self.state_dim]
        reward = output[..., -1]
        next_state = state + delta
        return next_state, reward


class WorldModelEnsemble(nn.Module):
    def __init__(self, cfg: WorldModelConfig, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.members = nn.ModuleList(
            FeedForwardDynamics(state_dim=state_dim, action_dim=action_dim, hidden_dims=cfg.hidden_dims)
            for _ in range(cfg.ensemble_size)
        )

    def _predict(self, state: Tensor, action: Tensor) -> tuple[Tensor, Tensor]:
        mean_state: Optional[Tensor] = None
        mean_reward: Optional[Tensor] = None
        for member in self.members:
            next_state, reward = member(state, action)
            if mean_state is None:
                mean_state = next_state
                mean_reward = reward
            else:
                mean_state = mean_state + next_state
                mean_reward = mean_reward + reward
        assert mean_state is not None and mean_reward is not None
        scale = 1.0 / len(self.members)
        return mean_state * scale, mean_reward * scale

    def forward(self, state: Tensor, action: Tensor) -> tuple[Tensor, Tensor]:
        return self._predict(state, action)


class TransitionBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._buffer: deque[dict[str, Tensor]] = deque(maxlen=capacity)

    def add_batch(
        self,
        states: Tensor,
        actions_enc: Tensor,
        rewards: Tensor,
        next_states: Tensor,
    ) -> None:
        states = states.detach().cpu()
        actions_enc = actions_enc.detach().cpu()
        rewards = rewards.detach().cpu()
        next_states = next_states.detach().cpu()
        for state, action, reward, next_state in zip(states, actions_enc, rewards, next_states, strict=False):
            self._buffer.append(
                {
                    "state": state,
                    "action_enc": action,
                    "reward": reward,
                    "next_state": next_state,
                }
            )

    def __len__(self) -> int:
        return len(self._buffer)

    def sample(self, batch_size: int, device: torch.device) -> TensorDict:
        actual = min(batch_size, len(self._buffer))
        batch = random.sample(self._buffer, actual)
        stacked = {k: torch.stack([item[k] for item in batch], dim=0).to(device=device) for k in batch[0].keys()}
        return TensorDict(stacked, batch_size=(actual,))


class CMPO(Loss):
    def __init__(
        self,
        policy: Policy,
        trainer_cfg: Any,
        env: TrainingEnvironment,
        device: torch.device,
        instance_name: str,
        cfg: CMPOConfig,
    ) -> None:
        super().__init__(policy, trainer_cfg, env, device, instance_name, cfg)
        self.burn_in_steps = getattr(self.policy, "burn_in_steps", 0)
        self.burn_in_steps_iter = 0

        obs_space = env.single_observation_space
        self.obs_shape = tuple(int(dim) for dim in obs_space.shape)
        self.obs_dim = int(torch.tensor(self.obs_shape).prod().item())
        action_space = env.single_action_space
        self.action_dim = int(action_space.n)

        self.world_model = WorldModelEnsemble(cfg.world_model, self.obs_dim, self.action_dim).to(device)
        self.world_model_opt = torch.optim.Adam(self.world_model.parameters(), lr=cfg.world_model.learning_rate)
        self.transition_buffer = TransitionBuffer(cfg.world_model.buffer_size)

        self.prior_model: Optional[Policy] = None
        if cfg.prior_ema_decay is not None:
            # π_prior in CMPO; EMA helps stabilize off-policy updates.
            self.prior_model = copy.deepcopy(self.policy).to(device)
            for param in self.prior_model.parameters():
                param.requires_grad = False

        self._prev_obs = torch.empty((0, self.obs_dim), dtype=torch.float32, device=device)
        self._prev_action_enc = torch.empty((0, self.action_dim), dtype=torch.float32, device=device)
        self._has_prev = torch.empty((0,), dtype=torch.bool, device=device)
        self._valid_action_mask: Optional[Tensor] = None

    def attach_replay_buffer(self, experience: Experience) -> None:  # type: ignore[override]
        super().attach_replay_buffer(experience)
        segments = experience.segments
        device = self.device
        self._prev_obs = torch.zeros((segments, self.obs_dim), dtype=torch.float32, device=device)
        self._prev_action_enc = torch.zeros((segments, self.action_dim), dtype=torch.float32, device=device)
        self._has_prev = torch.zeros((segments,), dtype=torch.bool, device=device)

    def get_experience_spec(self) -> Composite:
        scalar_f32 = UnboundedContinuous(shape=torch.Size([]), dtype=torch.float32)
        return Composite(
            actions=UnboundedDiscrete(shape=torch.Size([]), dtype=torch.int32),
            values=scalar_f32,
            rewards=scalar_f32,
            dones=scalar_f32,
            truncateds=scalar_f32,
            act_log_prob=scalar_f32,
        )

    def policy_output_keys(self, policy_td: Optional[TensorDict] = None) -> set[str]:
        return {"full_log_probs", "values"}

    def run_rollout(self, td: TensorDict, context: ComponentContext) -> None:
        with torch.no_grad():
            if "actions" in td.keys():
                self.policy.forward(td, action=td["actions"])
            else:
                self.policy.forward(td)

        env_slice = self._training_env_id(
            context,
            error="ComponentContext.training_env_id is required for CMPO rollout",
        )

        rewards = td["rewards"].to(dtype=torch.float32)
        dones = td["dones"].to(dtype=torch.float32)
        truncateds = td["truncateds"].to(dtype=torch.float32)
        terminals = torch.logical_or(dones > 0.5, truncateds > 0.5)

        obs = td["env_obs"]
        obs_flat = self._flatten_obs(obs)
        actions = td["actions"]
        actions_enc = self._encode_action(actions)

        if self._has_prev[env_slice].any():
            mask = self._has_prev[env_slice]
            prev_states = self._prev_obs[env_slice][mask]
            prev_actions_enc = self._prev_action_enc[env_slice][mask]

            current_states = obs_flat[mask]
            current_rewards = rewards[mask]

            self.transition_buffer.add_batch(
                states=prev_states,
                actions_enc=prev_actions_enc,
                rewards=current_rewards,
                next_states=current_states,
            )

        if self.burn_in_steps_iter < self.burn_in_steps:
            self.burn_in_steps_iter += 1
        else:
            self.replay.store(data_td=td, env_id=env_slice)

        self._prev_obs[env_slice] = obs_flat.detach()
        self._prev_action_enc[env_slice] = actions_enc.detach()
        self._has_prev[env_slice] = ~terminals.bool()

    def run_train(
        self,
        shared_loss_data: TensorDict,
        context: ComponentContext,
        mb_idx: int,
    ) -> tuple[Tensor, TensorDict, bool]:
        stop_update_epoch = False
        if mb_idx > 0 and self.cfg.target_kl is not None:
            approx_kl = self.loss_tracker["approx_kl"]
            avg_kl = sum(approx_kl) / len(approx_kl) if approx_kl else 0.0
            if avg_kl > self.cfg.target_kl:
                stop_update_epoch = True

        if mb_idx == 0:
            self._train_world_model()

        if len(self.transition_buffer) < self.cfg.world_model.warmup_transitions:
            self._update_prior_model()
            return self._zero(), shared_loss_data, stop_update_epoch

        minibatch = shared_loss_data["sampled_mb"]
        if minibatch.batch_size.numel() == 0:
            return self._zero(), shared_loss_data, stop_update_epoch

        policy_td = shared_loss_data["policy_td"]
        B, TT = minibatch.batch_size
        if self.cfg.minibatch_fraction < 1.0:
            keep = torch.rand(B, device=minibatch.device) < self.cfg.minibatch_fraction
            if not keep.any():
                keep[torch.randint(B, (1,), device=minibatch.device)] = True
            minibatch = minibatch[keep]
            policy_td = policy_td[keep]
            B, TT = minibatch.batch_size

        log_pi = policy_td["full_log_probs"].reshape(B, TT, -1)
        if self._valid_action_mask is None:
            self._valid_action_mask = (log_pi > -1e8).any(dim=(0, 1))
        prior_log_probs = self._get_prior_log_probs(minibatch, policy_td)
        q_values = self._compute_q_values(minibatch["env_obs"], valid_action_mask=self._valid_action_mask)  # [B, T, A]
        pi_prior = prior_log_probs.exp()
        v_prior = (pi_prior * q_values).sum(dim=-1, keepdim=True)
        advantages = q_values - v_prior

        if self.cfg.norm_adv:
            # Muesli Sec. 4.5: normalize advantages to reduce reward-scale sensitivity.
            adv_std = advantages.std(dim=-1, keepdim=True).clamp(min=1e-6)
            advantages = advantages / adv_std

        # Eq. (7): CMPO target π_CMPO ∝ π_prior · exp(clip(Â/τ, c)); τ=1 matches the paper.
        advantages_scaled = (advantages / self.cfg.temperature).clamp(-self.cfg.adv_clip, self.cfg.adv_clip)
        pi_cmpo = pi_prior * torch.exp(advantages_scaled)
        pi_cmpo = pi_cmpo / pi_cmpo.sum(dim=-1, keepdim=True)

        log_pi = log_pi.reshape(pi_cmpo.shape)
        # Eq. (9) KL regularizer: KL(π_CMPO || π) up to a constant.
        kl_loss = -(pi_cmpo.detach() * log_pi).sum(dim=-1).mean()

        value_pred = policy_td["values"].reshape(pi_cmpo.shape[0], pi_cmpo.shape[1])
        v_target = (pi_cmpo.detach() * q_values).sum(dim=-1)
        value_loss = 0.5 * F.mse_loss(value_pred, v_target)

        with torch.no_grad():
            kl = (pi_cmpo * (pi_cmpo.clamp(min=1e-8).log() - log_pi)).sum(dim=-1).mean()
            self.loss_tracker["approx_kl"].append(float(kl.item()))
            self.loss_tracker["policy_loss"].append(float(kl_loss.item()))
            self.loss_tracker["value_loss"].append(float(value_loss.item()))

        entropy = -(log_pi.exp() * log_pi).sum(dim=-1).mean()
        # This omits the Eq. (9) policy-gradient term; we train via CMPO distillation + value loss.
        loss = kl_loss + self.cfg.vf_coef * value_loss - self.cfg.ent_coef * entropy
        loss = add_dummy_loss_for_unused_params(loss, td=policy_td, used_keys=["full_log_probs", "values"])

        self._update_prior_model()
        return loss, shared_loss_data, stop_update_epoch

    def _update_prior_model(self) -> None:
        if self.prior_model is None or self.cfg.prior_ema_decay is None:
            return
        decay = self.cfg.prior_ema_decay
        with torch.no_grad():
            for prior_param, online_param in zip(self.prior_model.parameters(), self.policy.parameters(), strict=False):
                prior_param.data = decay * prior_param.data + (1 - decay) * online_param.data
            for prior_buf, online_buf in zip(self.prior_model.buffers(), self.policy.buffers(), strict=False):
                prior_buf.copy_(online_buf)

    def _get_prior_log_probs(self, minibatch: TensorDict, policy_td: TensorDict) -> Tensor:
        B, TT = minibatch.batch_size
        if self.prior_model is None:
            return policy_td["full_log_probs"].reshape(B, TT, -1).detach()

        with torch.no_grad():
            prior_td = forward_policy_for_training(self.prior_model, minibatch, self.policy_experience_spec)
        return prior_td["full_log_probs"].reshape(B, TT, -1).detach()

    def _compute_q_values(self, obs: Tensor, valid_action_mask: Tensor) -> Tensor:
        if self.prior_model is not None:
            value_model = self.prior_model
        else:
            value_model = self.policy

        B, TT = obs.shape[:2]
        obs_flat = obs.reshape(B * TT, *obs.shape[2:])
        state_flat = self._flatten_obs(obs_flat)
        num_states = state_flat.shape[0]

        valid_action_mask = valid_action_mask.to(device=obs.device)
        valid_action_indices = valid_action_mask.nonzero(as_tuple=False).view(-1)
        num_actions = int(valid_action_indices.numel())
        if num_actions == 0:
            return torch.zeros((B, TT, self.action_dim), device=obs.device, dtype=torch.float32)

        # Match PPO default memory footprint by limiting (state, action) pairs per chunk.
        max_pairs = self.cfg.q_value_batch_size or self.trainer_cfg.minibatch_size
        states_per_chunk = max(1, max_pairs // num_actions)

        q_values = torch.zeros((num_states, self.action_dim), device=obs.device, dtype=torch.float32)
        action_eye = torch.eye(self.action_dim, device=obs.device, dtype=torch.float32)[valid_action_indices]

        for start in range(0, num_states, states_per_chunk):
            end = min(start + states_per_chunk, num_states)
            state_chunk = state_flat[start:end]

            actions = action_eye.unsqueeze(0).expand(end - start, -1, -1)
            states = state_chunk.unsqueeze(1).expand(end - start, num_actions, self.obs_dim)
            states_flat = states.reshape((end - start) * num_actions, self.obs_dim)
            actions_flat = actions.reshape((end - start) * num_actions, self.action_dim)

            with torch.no_grad():
                # Eq. (14): one-step model lookahead q̂(s,a)=r̂+γ v̂(s').
                next_state, reward = self.world_model(states_flat, actions_flat)
                next_obs = self._unflatten_obs(next_state)
                next_values = self._value_from_obs(value_model, next_obs)

            q_chunk = reward.view(end - start, num_actions) + (
                self.trainer_cfg.advantage.gamma * next_values.view(end - start, num_actions)
            )
            q_values[start:end].index_copy_(1, valid_action_indices, q_chunk)

        return q_values.view(B, TT, self.action_dim)

    def _value_from_obs(self, model: Policy, obs: Tensor) -> Tensor:
        batch = obs.shape[0]
        td = TensorDict(
            {
                "env_obs": obs,
                "dones": torch.zeros(batch, device=obs.device, dtype=torch.float32),
                "truncateds": torch.zeros(batch, device=obs.device, dtype=torch.float32),
            },
            batch_size=(batch,),
        )
        ensure_sequence_metadata(td, batch_size=batch, time_steps=1)
        dummy_actions = torch.zeros(batch, device=obs.device, dtype=torch.long)
        model.reset_memory()
        with torch.no_grad():
            td = model.forward(td, action=dummy_actions)
        return td["values"].reshape(batch)

    def _flatten_obs(self, obs: Tensor) -> Tensor:
        obs_f = obs.to(dtype=torch.float32) / 255.0
        return obs_f.view(obs.shape[0], -1)

    def _unflatten_obs(self, obs_flat: Tensor) -> Tensor:
        obs = obs_flat.view(obs_flat.shape[0], *self.obs_shape)
        obs = (obs * 255.0).clamp(0, 255)
        return obs.to(dtype=torch.uint8)

    def _encode_action(self, action: Tensor) -> Tensor:
        action = action.view(-1).long()
        return F.one_hot(action, num_classes=self.action_dim).to(dtype=torch.float32)

    def _train_world_model(self) -> None:
        cfg = self.cfg.world_model
        if len(self.transition_buffer) < cfg.warmup_transitions:
            return

        for _ in range(cfg.train_steps):
            batch = self.transition_buffer.sample(cfg.batch_size, self.device)
            states = batch["state"]
            actions = batch["action_enc"]
            rewards = batch["reward"]
            targets = batch["next_state"]

            # Muesli Sec. 4.4: auxiliary model learning (here simplified to one-step dynamics).
            self.world_model_opt.zero_grad()
            pred_next, pred_reward = self.world_model(states, actions)
            loss_state = F.mse_loss(pred_next, targets)
            loss_reward = F.mse_loss(pred_reward, rewards)
            loss = loss_state + loss_reward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), cfg.gradient_clip)
            self.world_model_opt.step()

            self.loss_tracker["world_model_loss"].append(float(loss.item()))
