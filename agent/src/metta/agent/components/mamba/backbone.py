from __future__ import annotations

import math
from typing import Dict, List, Literal, Optional

import torch
import torch.nn as nn
from einops import rearrange
from tensordict import TensorDict
from torchrl.data import Composite, UnboundedDiscrete

from metta.rl.training import EnvironmentMetaData

from .config import MambaBackboneConfig
from metta.agent.components.mamba_ssm.modules.block import Block
from metta.agent.components.mamba_ssm.modules.mamba import Mamba
Mamba2 = None
from metta.agent.components.mamba_ssm.modules.mha import MHA
from metta.agent.components.mamba_ssm.modules.mlp import GatedMLP


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
        mixer_cls = lambda dim: MHA(dim, layer_idx=layer_idx, **attn_cfg)
    else:
        ssm_layer = cfg.pop("layer", "Mamba1")
        mixer = Mamba2 if ssm_layer == "Mamba2" else Mamba
        mixer_cls = lambda dim: mixer(dim, layer_idx=layer_idx, **cfg)

    if d_intermediate == 0:
        mlp_cls = nn.Identity
    else:
        mlp_cls = lambda dim: GatedMLP(dim, hidden_features=d_intermediate, out_features=d_model)

    norm_cls = lambda dim: (nn.LayerNorm if not rms_norm else nn.LayerNorm)(dim, eps=norm_epsilon)
    block = Block(d_model, mixer_cls, mlp_cls, norm_cls=norm_cls, fused_add_norm=False, residual_in_fp32=False)
    block.layer_idx = layer_idx
    return block


class _TokenCache:
    """Simple per-environment rolling window of Mamba tokens."""

    def __init__(self, max_steps: int) -> None:
        self.max_steps = max_steps
        self._store: List[torch.Tensor] = []
        self.position: int = 0

    def append(self, token: torch.Tensor) -> None:
        token = token.detach()
        self._store.append(token)
        if len(self._store) > self.max_steps:
            self._store.pop(0)
        self.position += 1

    def extend(self, tokens: List[torch.Tensor]) -> None:
        for tok in tokens:
            self.append(tok)

    def reset(self) -> None:
        self._store.clear()
        self.position = 0

    @property
    def size(self) -> int:
        return len(self._store)

    @property
    def data(self) -> torch.Tensor:
        if not self._store:
            raise ValueError("Token cache is empty")
        return torch.stack(self._store, dim=0)


class MambaBackboneComponent(nn.Module):
    """Sliding-window style Mamba backbone with simple token caching."""

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
        self.norm_f = nn.LayerNorm(config.d_model, eps=config.norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_p)

        self._token_caches: List[_TokenCache] = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _ensure_capacity(self, num_envs: int) -> None:
        if num_envs <= len(self._token_caches):
            return
        for _ in range(num_envs - len(self._token_caches)):
            self._token_caches.append(_TokenCache(self.max_cache_size))

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

            last_actions = td.get(
                "last_actions", torch.zeros(batch_flat, self.last_action_dim, device=device)
            )
            last_actions = rearrange(last_actions, "(b tt) d -> b tt 1 d", b=batch, tt=tt)
            action_token = self.action_proj(last_actions.float())
        else:
            reward_token = torch.zeros_like(x[..., :1, :])
            reset_token = torch.zeros_like(x[..., :1, :])
            action_token = torch.zeros_like(x[..., :1, :])

        cls = self.cls_token.to(device).expand(x.size(0), x.size(1), -1, -1)
        tokens = torch.cat([cls, x, reward_token + reset_token, action_token], dim=2)
        return tokens

    def _run_layers(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden_states = tokens
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual)
        if residual is not None:
            hidden_states = hidden_states + residual
        hidden_states = self.norm_f(self.dropout(hidden_states))
        return hidden_states

    def _pool_output(self, output: torch.Tensor, tt: int) -> torch.Tensor:
        if self.pool == "cls":
            if tt == 1:
                return output[:, 0, :]
            return output[:, :, 0, :]
        if self.pool == "mean":
            if tt == 1:
                return output.mean(dim=1)
            return output.mean(dim=2)
        if tt == 1:
            return rearrange(output, "b s d -> b (s d)")
        return rearrange(output, "b tt s d -> b tt (s d)")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, td: TensorDict) -> TensorDict:
        tokens = self._build_tokens(td)
        device = tokens.device

        tt = tokens.size(1)
        batch = tokens.size(0)
        S = tokens.size(2)

        training_env_ids = td.get("training_env_ids", None)
        if training_env_ids is None:
            env_ids = torch.arange(batch, device=device, dtype=torch.long)
        else:
            env_ids = training_env_ids.reshape(-1).to(device=device, dtype=torch.long)
        self._ensure_capacity(int(env_ids.max().item()) + 1)

        if tt == 1:
            outputs = []
            positions = []
            dones = td.get("dones", torch.zeros(batch, device=device))
            truncateds = td.get("truncateds", torch.zeros(batch, device=device))
            resets = torch.logical_or(dones.bool(), truncateds.bool())

            for idx in range(batch):
                env_id = int(env_ids[idx].item())
                cache = self._token_caches[env_id]
                current = tokens[idx, 0]  # (S, D)
                cache.append(current)

                history = cache.data  # (steps, S, D)
                steps = history.size(0)
                seq = history.reshape(1, steps * S, -1)

                pos = torch.arange(steps, device=device, dtype=torch.long)
                pos_enc = _sinusoidal_positional_encoding(pos, seq.size(-1))
                pos_enc = pos_enc.unsqueeze(1).expand(-1, S, -1)
                seq = seq + pos_enc.reshape(1, steps * S, -1)

                hidden = self._run_layers(seq)
                hidden = hidden.reshape(steps, S, -1)
                last_hidden = hidden[-1].unsqueeze(0)  # (1, S, D)
                pooled = self._pool_output(last_hidden, tt=1)
                outputs.append(pooled.squeeze(0))

                positions.append(torch.tensor(cache.position - 1, device=device, dtype=torch.long))

                if resets[idx]:
                    cache.reset()

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

        hidden = self._run_layers(seq)
        hidden = hidden.reshape(batch, tt, S, -1)
        pooled = self._pool_output(hidden, tt=tt)
        flat = rearrange(pooled, "b tt ... -> (b tt) ...")
        if flat.shape[0] != td.batch_size.numel():
            raise RuntimeError(f"pooled batch {flat.shape[0]} != td batch {td.batch_size.numel()}")
        td.set(self.out_key, flat)

        # update caches
        dones = td.get("dones", torch.zeros(batch * tt, device=device))
        truncateds = td.get("truncateds", torch.zeros(batch * tt, device=device))
        resets = torch.logical_or(dones.bool(), truncateds.bool()).reshape(batch, tt)

        for idx in range(batch):
            env_id = int(env_ids[idx].item())
            cache = self._token_caches[env_id]
            for step in range(tt):
                cache.append(tokens[idx, step])
                if resets[idx, step]:
                    cache.reset()

        td.set(
            "transformer_position",
            torch.tensor(
                [self._token_caches[int(env_ids[idx].item())].position - 1 for idx in range(batch)],
                device=device,
                dtype=torch.long,
            ),
        )
        return td

    # ------------------------------------------------------------------
    # Integration hooks
    # ------------------------------------------------------------------
    def get_agent_experience_spec(self) -> Composite:
        return Composite({"transformer_position": UnboundedDiscrete(shape=torch.Size([]), dtype=torch.long)})

    def initialize_to_environment(self, env: EnvironmentMetaData, device: torch.device) -> Optional[str]:
        self._token_caches = []
        return None

    def reset_memory(self) -> None:
        for cache in self._token_caches:
            cache.reset()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
