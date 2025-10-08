from __future__ import annotations

import math
from functools import partial
from typing import List, Literal, Optional

import torch
import torch.nn as nn
from einops import rearrange
from tensordict import TensorDict
from torchrl.data import Composite

from metta.agent.components.mamba_ssm.modules.block import Block
from metta.agent.components.mamba_ssm.modules.mamba import Mamba
from metta.agent.components.mamba_ssm.modules.mamba2 import Mamba2
from metta.agent.components.mamba_ssm.modules.mha import MHA
from metta.agent.components.mamba_ssm.modules.mlp import GatedMLP
from metta.agent.components.mamba_ssm.ops.triton.layer_norm import RMSNorm
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


class MambaBackboneComponent(nn.Module):
    """Simplified Mamba backbone following the Transformer policy flow."""

    def __init__(self, config: MambaBackboneConfig, env: Optional[EnvironmentMetaData] = None):
        super().__init__()
        self.config = config
        self.in_key = config.in_key
        self.out_key = config.out_key
        self.pool: Literal["cls", "mean", "none"] = config.pool
        self.use_aux_tokens = config.use_aux_tokens
        self.last_action_dim = config.last_action_dim

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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
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

    def _apply_layers(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden_states = tokens
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual)
        if residual is not None:
            hidden_states = hidden_states + residual
        hidden_states = self.norm_f(self.dropout(hidden_states))
        return hidden_states

    def _pool_output(self, output: torch.Tensor) -> torch.Tensor:
        leading = output.shape[:-2]
        seq_len = output.shape[-2]
        hidden_dim = output.shape[-1]

        flat = output.reshape(-1, seq_len, hidden_dim)

        if self.pool == "cls":
            pooled = flat[:, 0, :]
        elif self.pool == "mean":
            pooled = flat.mean(dim=1)
        elif self.pool == "none":
            pooled = flat.reshape(flat.size(0), seq_len * hidden_dim)
        else:
            raise ValueError(f"Unsupported pool mode: {self.pool}")

        return pooled.reshape(*leading, -1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, td: TensorDict) -> TensorDict:
        tokens = self._build_tokens(td)
        device = tokens.device

        tt = tokens.size(1)
        batch = tokens.size(0)
        S = tokens.size(2)

        if tt == 1:
            seq_tokens = tokens.reshape(batch, S, -1)
            pos = torch.arange(S, device=device, dtype=torch.long)
            seq_tokens = seq_tokens + _sinusoidal_positional_encoding(pos, seq_tokens.size(-1))

            hidden = self._apply_layers(seq_tokens)
            pooled = self._pool_output(hidden)
            td.set(self.out_key, pooled)
            return td

        seq_tokens = tokens.reshape(batch * tt, S, -1)
        pos = torch.arange(S, device=device, dtype=torch.long)
        pos_enc = _sinusoidal_positional_encoding(pos, seq_tokens.size(-1))
        seq_tokens = seq_tokens + pos_enc.unsqueeze(0)

        hidden = self._apply_layers(seq_tokens)
        pooled = self._pool_output(hidden)
        td.set(self.out_key, pooled.reshape(batch * tt, -1))
        return td

    # ------------------------------------------------------------------
    # Integration hooks
    # ------------------------------------------------------------------
    def get_agent_experience_spec(self) -> Composite:
        return Composite({})

    def initialize_to_environment(self, env: EnvironmentMetaData, device: torch.device) -> Optional[str]:
        return None

    def reset_memory(self) -> None:
        return

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
