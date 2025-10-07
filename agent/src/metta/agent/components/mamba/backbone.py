from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from typing import Dict, List, Literal, Optional

import torch
import torch.nn as nn
from einops import rearrange
from tensordict import TensorDict
from torchrl.data import Composite, UnboundedDiscrete

from metta.agent.components.mamba_ssm.modules.block import Block
from metta.agent.components.mamba_ssm.modules.mamba import Mamba
from metta.agent.components.mamba_ssm.modules.mamba2 import Mamba2
from metta.agent.components.mamba_ssm.modules.mha import MHA
from metta.agent.components.mamba_ssm.modules.mlp import GatedMLP
from metta.agent.components.mamba_ssm.ops.triton.layer_norm import RMSNorm
from metta.agent.components.mamba_ssm.utils.generation import InferenceParams
from metta.rl.training import EnvironmentMetaData

from .config import MambaBackboneConfig

TORCH_DTYPE = torch.float32


def _sinusoidal_positional_encoding(positions: torch.Tensor, dim: int) -> torch.Tensor:
    device = positions.device
    div_term = torch.exp(torch.arange(0, dim, 2, device=device) * -(math.log(10000.0) / dim))
    pe = torch.zeros(*positions.shape, dim, device=device, dtype=TORCH_DTYPE)
    pe[..., 0::2] = torch.sin(positions.unsqueeze(-1) * div_term)
    pe[..., 1::2] = torch.cos(positions.unsqueeze(-1) * div_term)
    return pe


def _create_block(
    d_model: int,
    d_intermediate: int,
    layer_idx: int,
    ssm_cfg: dict,
    attn_layer_idx: List[int],
    attn_cfg: dict,
    norm_epsilon: float,
    rms_norm: bool,
) -> Block:
    cfg = dict(ssm_cfg) if ssm_cfg else {}
    attn_cfg = dict(attn_cfg) if attn_cfg else {}

    if layer_idx in attn_layer_idx:
        mixer_cls = partial(MHA, layer_idx=layer_idx, **attn_cfg)
    else:
        ssm_layer = cfg.pop("layer", "Mamba1")
        mixer = Mamba2 if ssm_layer == "Mamba2" else Mamba
        mixer_cls = partial(mixer, layer_idx=layer_idx, **cfg)

    if d_intermediate == 0:
        mlp_cls = nn.Identity
    else:
        mlp_cls = partial(GatedMLP, hidden_features=d_intermediate, out_features=d_model)

    norm_base = nn.LayerNorm if not rms_norm else RMSNorm
    norm_cls = partial(norm_base, eps=norm_epsilon)
    block = Block(d_model, mixer_cls, mlp_cls, norm_cls=norm_cls, fused_add_norm=False, residual_in_fp32=False)
    block.layer_idx = layer_idx
    return block


def _zero_state(cache: torch.Tensor | Dict | List | tuple) -> None:
    if torch.is_tensor(cache):
        cache.zero_()
        return
    if isinstance(cache, dict):
        for value in cache.values():
            _zero_state(value)
        return
    if isinstance(cache, (list, tuple)):
        for value in cache:
            _zero_state(value)


@dataclass
class _EnvState:
    inference_params: InferenceParams
    position: int
    tokens_per_step: int

    def reset(self) -> None:
        for cache in self.inference_params.key_value_memory_dict.values():
            _zero_state(cache)
        self.position = 0
        self.inference_params.seqlen_offset = 0


class MambaBackboneComponent(nn.Module):
    """Sliding-window style Mamba backbone with per-env KV caches."""

    def __init__(self, config: MambaBackboneConfig, env: Optional[EnvironmentMetaData] = None):
        super().__init__()
        self.config = config
        self.in_key = config.in_key
        self.out_key = config.out_key
        self.pool: Literal["cls", "mean", "none"] = config.pool
        self.use_aux_tokens = config.use_aux_tokens
        self.last_action_dim = config.last_action_dim
        self.max_cache_size = max(1, config.max_cache_size)

        self.input_proj = nn.Linear(config.input_dim, config.d_model)

        if self.use_aux_tokens:
            self.reward_proj = nn.Linear(1, config.d_model)
            self.reset_proj = nn.Linear(1, config.d_model)
            self.action_proj = nn.Linear(self.last_action_dim, config.d_model)
        else:
            self.reward_proj = None
            self.reset_proj = None
            self.action_proj = None

        self.cls_token = nn.Parameter(torch.randn(1, 1, config.d_model))

        self.layers = nn.ModuleList(
            [
                _create_block(
                    d_model=config.d_model,
                    d_intermediate=config.d_intermediate,
                    layer_idx=i,
                    ssm_cfg=config.ssm_cfg,
                    attn_layer_idx=config.attn_layer_idx,
                    attn_cfg=config.attn_cfg,
                    norm_epsilon=config.norm_epsilon,
                    rms_norm=config.rms_norm,
                )
                for i in range(config.n_layer)
            ]
        )
        norm_cls = nn.LayerNorm if not config.rms_norm else RMSNorm
        self.norm_f = norm_cls(config.d_model, eps=config.norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_p)

        self._env_states: List[Optional[_EnvState]] = []
        self._tokens_per_step: Optional[int] = None

    def _ensure_capacity(self, num_envs: int) -> None:
        if num_envs <= len(self._env_states):
            return
        self._env_states.extend([None] * (num_envs - len(self._env_states)))

    def _ensure_env_state(
        self, env_id: int, device: torch.device, dtype: torch.dtype, tokens_per_step: int
    ) -> _EnvState:
        state = self._env_states[env_id] if env_id < len(self._env_states) else None
        if state is not None and state.tokens_per_step == tokens_per_step:
            return state

        max_seqlen = max(self.max_cache_size * tokens_per_step, tokens_per_step)
        inference_params = InferenceParams(max_seqlen=max_seqlen, max_batch_size=1)
        key_value_memory: Dict[int, torch.Tensor | tuple] = {}
        for layer in self.layers:
            cache = layer.allocate_inference_cache(batch_size=1, max_seqlen=max_seqlen, dtype=dtype)
            key_value_memory[layer.layer_idx] = cache
            _zero_state(cache)
        inference_params.key_value_memory_dict = key_value_memory
        state = _EnvState(inference_params=inference_params, position=0, tokens_per_step=tokens_per_step)
        self._env_states[env_id] = state
        return state

    def _reset_env_state(self, env_id: int) -> None:
        if env_id < len(self._env_states) and self._env_states[env_id] is not None:
            self._env_states[env_id].reset()

    # Token preparation -------------------------------------------------
    def _build_tokens(self, td: TensorDict) -> torch.Tensor:
        x = td[self.in_key]
        device = x.device

        if x.dim() == 2:
            x = x.unsqueeze(-2)
        x = self.input_proj(x)

        batch_flat = td.batch_size.numel()
        tt = int(td.get("bptt", torch.ones(1, device=device))[0].item())
        batch = max(batch_flat // tt, 1)

        x = rearrange(x, "(b tt) s d -> b tt s d", b=batch, tt=tt)

        if self.use_aux_tokens:
            zeros = torch.zeros(batch_flat, device=device)
            rewards = td.get("rewards", zeros)
            rewards = rearrange(rewards, "(b tt) -> b tt 1 1", b=batch, tt=tt).float()
            reward_token = self.reward_proj(rewards)

            dones = td.get("dones", zeros)
            truncateds = td.get("truncateds", zeros)
            resets = torch.logical_or(dones.bool(), truncateds.bool()).float()
            resets = rearrange(resets, "(b tt) -> b tt 1 1", b=batch, tt=tt)
            reset_token = self.reset_proj(resets)

            last_actions = td.get("last_actions", torch.zeros(batch_flat, self.last_action_dim, device=device))
            last_actions = rearrange(last_actions, "(b tt) d -> b tt 1 d", b=batch, tt=tt)
            action_token = self.action_proj(last_actions.float())
        else:
            reward_token = torch.zeros_like(x[..., :1, :])
            reset_token = torch.zeros_like(x[..., :1, :])
            action_token = torch.zeros_like(x[..., :1, :])

        cls = self.cls_token.to(device).expand(x.size(0), x.size(1), -1, -1)
        tokens = torch.cat([cls, x, reward_token + reset_token, action_token], dim=2)
        return tokens

    def _apply_layers(self, tokens: torch.Tensor, inference_params=None) -> torch.Tensor:
        hidden_states = tokens
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual, inference_params=inference_params)
        if residual is not None:
            hidden_states = hidden_states + residual
        hidden_states = self.norm_f(self.dropout(hidden_states))
        return hidden_states

    def _pool_output(self, output: torch.Tensor) -> torch.Tensor:
        if self.pool == "cls":
            return output[:, 0, :]
        if self.pool == "mean":
            return output.mean(dim=1)
        if self.pool == "none":
            return rearrange(output, "b s d -> b (s d)")
        raise ValueError(f"Unsupported pool mode: {self.pool}")

    # Forward -----------------------------------------------------------
    def forward(self, td: TensorDict) -> TensorDict:
        tokens = self._build_tokens(td)
        device = tokens.device

        tt = tokens.size(1)
        batch = tokens.size(0)
        S = tokens.size(2)
        self._tokens_per_step = S

        training_env_ids = td.get("training_env_ids", None)
        if training_env_ids is None:
            env_ids = torch.arange(batch, device=device, dtype=torch.long)
        else:
            env_ids = training_env_ids.reshape(-1).to(device=device, dtype=torch.long)
        self._ensure_capacity(int(env_ids.max().item()) + 1)

        if tt == 1:
            outputs = []
            positions = []
            if not self._env_states:
                raise RuntimeError(
                    "MambaBackboneComponent requires initialize_to_environment or a prior training pass before rollout"
                )
            dones = td.get("dones", torch.zeros(batch, device=device))
            truncateds = td.get("truncateds", torch.zeros(batch, device=device))
            resets = torch.logical_or(dones.bool(), truncateds.bool())

            for idx in range(batch):
                env_id = int(env_ids[idx].item())
                state = self._ensure_env_state(env_id, device, tokens.dtype, S)
                step_tokens = tokens[idx, 0]

                hidden_stack = []
                position = state.position
                for token_idx in range(S):
                    token = step_tokens[token_idx].unsqueeze(0).unsqueeze(0)
                    pos_enc = _sinusoidal_positional_encoding(
                        torch.tensor([position], device=device, dtype=torch.long),
                        token.size(-1),
                    )
                    token = token + pos_enc.unsqueeze(0)
                    hidden = self._apply_layers(token, inference_params=state.inference_params)
                    hidden_stack.append(hidden.squeeze(0).squeeze(0))
                    position += 1
                    state.inference_params.seqlen_offset = position
                    state.inference_params.max_seqlen = max(state.inference_params.max_seqlen, position)

                hidden_tensor = torch.stack(hidden_stack, dim=0).unsqueeze(0)
                pooled = self._pool_output(hidden_tensor)
                outputs.append(pooled.squeeze(0))
                positions.append(torch.tensor(position - 1, device=device, dtype=torch.long))

                state.position = position
                if resets[idx]:
                    self._reset_env_state(env_id)

            out_tensor = torch.stack(outputs, dim=0)
            td.set(self.out_key, out_tensor)
            td.set("transformer_position", torch.stack(positions, dim=0))
            return td

        # training path
        pos = torch.arange(tt, device=device, dtype=torch.long)
        pos_enc = _sinusoidal_positional_encoding(pos, tokens.size(-1))
        pos_enc = pos_enc.unsqueeze(1).expand(-1, S, -1)
        seq = tokens + pos_enc
        seq = rearrange(seq, "b tt s d -> b (tt s) d")

        hidden = self._apply_layers(seq)
        hidden = hidden.reshape(batch, tt, S, -1)
        pooled = hidden.reshape(batch * tt, S, -1)
        flat = self._pool_output(pooled)
        td.set(self.out_key, flat)

        # update caches to match training sequence
        dones = td.get("dones", torch.zeros(batch * tt, device=device))
        truncateds = td.get("truncateds", torch.zeros(batch * tt, device=device))
        resets = torch.logical_or(dones.bool(), truncateds.bool()).reshape(batch, tt)

        for idx in range(batch):
            env_id = int(env_ids[idx].item())
            state = self._ensure_env_state(env_id, device, tokens.dtype, S)
            position = state.position
            for step in range(tt):
                step_tokens = tokens[idx, step]
                for token_idx in range(S):
                    token = step_tokens[token_idx].unsqueeze(0).unsqueeze(0)
                    pos_enc = _sinusoidal_positional_encoding(
                        torch.tensor([position], device=device, dtype=torch.long),
                        token.size(-1),
                    )
                    token = token + pos_enc.unsqueeze(0)
                    self._apply_layers(token, inference_params=state.inference_params)
                    position += 1
                if resets[idx, step]:
                    self._reset_env_state(env_id)
                    state = self._ensure_env_state(env_id, device, tokens.dtype, S)
                    position = state.position
            state.position = position

        return td

    # ------------------------------------------------------------------
    # Integration hooks
    # ------------------------------------------------------------------
    def get_agent_experience_spec(self) -> Composite:
        return Composite({"transformer_position": UnboundedDiscrete(shape=torch.Size([]), dtype=torch.long)})

    def initialize_to_environment(self, env: EnvironmentMetaData, device: torch.device) -> Optional[str]:
        self._env_states = []
        return None

    def reset_memory(self) -> None:
        for env_id in range(len(self._env_states)):
            self._reset_env_state(env_id)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
