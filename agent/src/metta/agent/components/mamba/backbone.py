from __future__ import annotations

import math
from typing import Literal, Optional

import torch
import torch.nn as nn
from einops import rearrange
from tensordict import TensorDict
from torchrl.data import Composite, UnboundedDiscrete

from metta.rl.training import EnvironmentMetaData

from .config import MambaBackboneConfig
from metta.agent.components.mamba_ssm.modules.block import Block
from metta.agent.components.mamba_ssm.modules.mamba import Mamba
from metta.agent.components.mamba_ssm.modules.mamba2 import Mamba as Mamba2
from metta.agent.components.mamba_ssm.modules.mha import MHA
from metta.agent.components.mamba_ssm.modules.mlp import GatedMLP

def _create_block(
    d_model: int,
    d_intermediate: int,
    layer_idx: int,
    ssm_cfg: dict,
    attn_layer_idx: list[int],
    attn_cfg: dict,
    norm_epsilon: float,
    rms_norm: bool,
):
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


class MambaBackboneComponent(nn.Module):
    """Sliding-window style Mamba backbone."""

    def __init__(self, config: MambaBackboneConfig, env: Optional[EnvironmentMetaData] = None):
        super().__init__()
        self.config = config
        self.in_key = config.in_key
        self.out_key = config.out_key
        self.pool: Literal["cls", "mean", "none"] = config.pool
        self.use_aux_tokens = config.use_aux_tokens
        self.last_action_dim = config.last_action_dim
        self.max_cache_size = config.max_cache_size

        self.input_proj = nn.Linear(config.input_dim, config.d_model)

        if self.use_aux_tokens:
            self.reward_proj = nn.Linear(1, config.d_model)
            self.reset_proj = nn.Linear(1, config.d_model)
            self.action_proj = nn.Linear(self.last_action_dim, config.d_model)
        else:
            self.register_parameter("reward_proj", None)
            self.register_parameter("reset_proj", None)
            self.register_parameter("action_proj", None)

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

        self.register_buffer("position_counter", torch.zeros(0, dtype=torch.long), persistent=False)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _ensure_capacity(self, num_envs: int, device: torch.device) -> None:
        if num_envs <= self.position_counter.size(0):
            return
        diff = num_envs - self.position_counter.size(0)
        filler = torch.zeros(diff, dtype=torch.long, device=device)
        if self.position_counter.numel() == 0:
            self.position_counter = filler
        else:
            self.position_counter = torch.cat([self.position_counter.to(device), filler], dim=0)

    def _sinusoidal_positional_encoding(self, positions: torch.Tensor, dim: int) -> torch.Tensor:
        device = positions.device
        div_term = torch.exp(torch.arange(0, dim, 2, device=device) * -(math.log(10000.0) / dim))
        pe = torch.zeros(*positions.shape, dim, device=device)
        pe[..., 0::2] = torch.sin(positions.unsqueeze(-1) * div_term)
        pe[..., 1::2] = torch.cos(positions.unsqueeze(-1) * div_term)
        return pe

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
        x = td[self.in_key]
        device = x.device
        batch_flat = td.batch_size.numel()
        tt = int(td.get("bptt", torch.ones(1, device=device))[0].item())
        batch = batch_flat // tt if tt > 0 else batch_flat

        if x.dim() == 2:
            x = x.unsqueeze(-2)
        # project observations
        x = self.input_proj(x)

        x = rearrange(x, "(b tt) s d -> b tt s d", b=batch, tt=tt)

        if self.use_aux_tokens:
            rewards = td.get("rewards", torch.zeros(batch_flat, device=device))
            rewards = rearrange(rewards, "(b tt) -> b tt 1 1", b=batch, tt=tt)
            reward_token = self.reward_proj(rewards)

            dones = td.get("dones", torch.zeros(batch_flat, device=device))
            truncateds = td.get("truncateds", torch.zeros(batch_flat, device=device))
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

        cls = self.cls_token.to(device).expand(batch, tt, -1, -1)
        tokens = torch.cat([cls, x, reward_token + reset_token, action_token], dim=2)
        seq_tokens = rearrange(tokens, "b tt s d -> b (tt s) d")

        training_env_ids = td.get("training_env_ids", None)
        if training_env_ids is None:
            env_ids = torch.arange(batch, device=device)
        else:
            env_ids = training_env_ids.reshape(-1).to(device)

        self._ensure_capacity(int(env_ids.max().item()) + 1, device)

        if tt == 1:
            positions = self.position_counter[env_ids]
            pos_enc = self._sinusoidal_positional_encoding(positions, tokens.size(-1))
            pos_enc = pos_enc.unsqueeze(1).expand(-1, tokens.size(2), -1)
            seq_tokens = seq_tokens + pos_enc
            self.position_counter[env_ids] += 1

            hidden_states = seq_tokens
            residual = None
            for layer in self.layers:
                hidden_states, residual = layer(hidden_states, residual)
            residual = hidden_states + (residual if residual is not None else 0)
            hidden_states = self.norm_f(self.dropout(residual))
            pooled = self._pool_output(hidden_states.unsqueeze(1), tt=1)
            td.set(self.out_key, pooled)
            td.set("transformer_position", positions.detach())

            resets_mask = td.get("dones", torch.zeros(batch, device=device))
            truncated_mask = td.get("truncateds", torch.zeros(batch, device=device))
            reset_mask = torch.logical_or(resets_mask.bool(), truncated_mask.bool())
            self.position_counter[env_ids] *= (~reset_mask).long()
            return td

        # training path (TT > 1)
        start_pos = td.get("transformer_position", torch.zeros(batch, device=device).long())
        positions = start_pos.unsqueeze(1) + torch.arange(tt, device=device)
        pos_enc = self._sinusoidal_positional_encoding(positions, tokens.size(-1))
        pos_enc = pos_enc.unsqueeze(2).expand(-1, -1, tokens.size(2), -1)
        seq_tokens = seq_tokens + rearrange(pos_enc, "b tt s d -> b (tt s) d")

        hidden_states = seq_tokens
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual)
        residual = hidden_states + (residual if residual is not None else 0)
        hidden_states = self.norm_f(self.dropout(residual))
        hidden_states = rearrange(hidden_states, "b (tt s) d -> b tt s d", tt=tt)

        pooled = self._pool_output(hidden_states, tt=tt)
        pooled = rearrange(pooled, "b tt ... -> (b tt) ...")
        td.set(self.out_key, pooled)
        td.set("transformer_position", positions[:, -1].detach())
        return td

    # ------------------------------------------------------------------
    # Integration hooks
    # ------------------------------------------------------------------
    def get_agent_experience_spec(self) -> Composite:
        return Composite({"transformer_position": UnboundedDiscrete(shape=torch.Size([]), dtype=torch.long)})

    def initialize_to_environment(self, env: EnvironmentMetaData, device: torch.device) -> Optional[str]:
        self.position_counter = torch.zeros(env.num_envs, dtype=torch.long, device=device)
        return None

    def reset_memory(self) -> None:
        self.position_counter.zero_()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
