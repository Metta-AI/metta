import math
from typing import Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tensordict import TensorDict
from torchrl.data import Composite, UnboundedDiscrete

from metta.agent.components.component_config import ComponentConfig


class TransformerBlock(nn.Module):
    """A single block of the transformer architecture."""

    def __init__(self, output_dim, num_heads, ff_mult):
        super().__init__()
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads

        self.q_proj = nn.Linear(output_dim, output_dim)
        self.k_proj = nn.Linear(output_dim, output_dim)
        self.v_proj = nn.Linear(output_dim, output_dim)
        self.out_proj = nn.Linear(output_dim, output_dim)

        self.norm1 = nn.LayerNorm(output_dim)
        self.norm2 = nn.LayerNorm(output_dim)

        ff_hidden_dim = output_dim * ff_mult
        self.ffn = nn.Sequential(
            nn.Linear(output_dim, ff_hidden_dim),
            nn.GELU(),
            nn.Linear(ff_hidden_dim, output_dim),
        )

        self.norm_factor = self.head_dim**-0.5

    def forward(self, x, pk, pv, mask):
        B, T, E = x.shape  # Batch, Tokens, EmbedDim

        # LayerNorm and Self-Attention
        x_norm = self.norm1(x)
        q = self.q_proj(x_norm).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_norm).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Combine with past KV cache for this layer
        if pk is not None and pv is not None:
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)

        # Attention calculation
        attn_weights = (q @ k.transpose(-2, -1)) * self.norm_factor
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask, float("-inf"))
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = (attn_weights @ v).transpose(1, 2).contiguous().view(B, T, E)

        # Residual connection and FFN
        x = x + self.out_proj(attn_output)
        x = x + self.ffn(self.norm2(x))

        return x, k, v


class SlidingTransformerConfig(ComponentConfig):
    in_key: str
    out_key: str
    name: str = "sliding_transformer"
    output_dim: int = 16
    input_dim: int = 64
    num_heads: int = 1
    ff_mult: int = 4
    num_layers: int
    max_cache_size: int = 80
    pool: Literal["cls", "mean", "none"] = "mean"
    last_action_dim: int = 2

    def make_component(self, env=None):
        return SlidingTransformer(config=self, env=env)


class SlidingTransformer(nn.Module):
    """A sliding window transformer with a KV cache.

    This component processes sequences of observations, actions, and rewards.
    It maintains a KV cache to support efficient rollouts and training with BPTT.
    It has two modes of operation, determined by the time dimension (TT) of the input TensorDict:
    - Rollout (TT=1): Processes a single timestep for a batch of environments, updating the KV cache.
    - Training (TT>1): Processes a batch of sequences, using a causal mask for self-attention.
    """

    def __init__(self, config: SlidingTransformerConfig, env):
        super().__init__()
        self.config = config
        self.max_cache_size = self.config.max_cache_size
        self.in_key = self.config.in_key
        self.out_key = self.config.out_key

        self.output_dim = self.config.output_dim
        self.num_layers = self.config.num_layers

        input_dim = self.config.input_dim
        if input_dim == self.output_dim:
            self.input_proj = nn.Identity()
        else:
            self.input_proj = nn.Linear(input_dim, self.output_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    output_dim=self.output_dim,
                    num_heads=self.config.num_heads,
                    ff_mult=self.config.ff_mult,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.last_action_proj = nn.Linear(self.config.last_action_dim, self.output_dim)
        self.reward_proj = nn.Linear(1, self.output_dim)
        self.dones_truncateds_proj = nn.Linear(1, self.output_dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.output_dim))
        self.final_norm = nn.LayerNorm(self.output_dim)

        # State buffers
        head_dim = self.config.output_dim // self.config.num_heads
        cache_shape = (0, self.num_layers, self.config.num_heads, self.max_cache_size, head_dim)
        self.register_buffer("k_cache_rollout", torch.empty(cache_shape))
        self.register_buffer("v_cache_rollout", torch.empty(cache_shape))
        self.register_buffer("k_cache_training", torch.empty(cache_shape))
        self.register_buffer("v_cache_training", torch.empty(cache_shape))
        self.register_buffer("position_counter", torch.zeros(0, dtype=torch.long))

    def _get_positional_encoding(self, positions: torch.Tensor, d_model: int) -> torch.Tensor:
        """Generates sinusoidal positional encodings."""
        device = positions.device
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(*positions.shape, d_model, device=device)
        pe[..., 0::2] = torch.sin(positions.unsqueeze(-1) * div_term)
        pe[..., 1::2] = torch.cos(positions.unsqueeze(-1) * div_term)
        return pe

    def _pool_output(self, output: torch.Tensor, tt: int) -> torch.Tensor:
        """Applies pooling to the transformer output."""
        if self.config.pool == "cls":
            if tt == 1:
                return output[:, 0, :]  # Select CLS token output
            return output[:, :, 0, :]  # Select CLS token
        elif self.config.pool == "mean":
            if tt == 1:
                return output.mean(dim=1)
            return output.mean(dim=2)  # pool over S dimension
        elif self.config.pool == "none":
            if tt == 1:
                return rearrange(output, "b s d -> b (s d)")
            return rearrange(output, "b tt s d -> b tt (s d)")
        else:
            raise ValueError(f"Unsupported pool mode: {self.config.pool}")

    def _prepare_inputs(self, td: TensorDict) -> Tuple[torch.Tensor, int, int, int]:
        """Reads and prepares input tensors from the TensorDict."""
        B_flat = td.batch_size.numel()
        TT = td.get("bptt", [1])[0]
        B = B_flat // TT

        # 1. Read inputs and prepare them for the transformer
        x = td[self.in_key]  # observation token(s). Can be [B*TT, E] or [B*TT, seq_len, E]
        x = self.input_proj(x)

        empty_tensor = torch.zeros(B * TT, device=td.device)
        reward = td.get("reward", empty_tensor)  # scalar
        last_actions = td.get("last_actions", torch.zeros(B * TT, self.config.last_action_dim, device=td.device))
        dones = td.get("dones", empty_tensor)
        truncateds = td.get("truncateds", empty_tensor)

        # Handle variable observation shapes [B, E] -> [B, 1, E]
        if x.dim() == 2:
            x = x.unsqueeze(-2)

        # Reshape all inputs to be [B, TT, Seq, Dims]
        x = rearrange(x, "(b tt) ... d -> b tt (...) d", tt=TT)
        reward = rearrange(reward, "(b tt) ... -> b tt (...) 1", tt=TT)
        last_actions = rearrange(last_actions, "(b tt) ... d -> b tt (...) d", tt=TT)
        resets = torch.logical_or(dones.bool(), truncateds.bool()).float()
        resets = rearrange(resets, "(b tt) ... -> b tt (...) 1", tt=TT)

        # Project inputs to tokens
        reward_token = self.reward_proj(reward)
        reset_token = self.dones_truncateds_proj(resets)
        reward_reset_token = (reward_token + reset_token).view(B, TT, 1, self.output_dim)
        action_token = self.last_action_proj(last_actions.float()).view(B, TT, 1, self.output_dim)

        # Combine all tokens for each timestep
        cls_token = self.cls_token.expand(B, TT, -1, -1)
        # Final sequence shape: [B, TT, S, E] where S is num tokens per step
        x = torch.cat([cls_token, x, reward_reset_token, action_token], dim=2)

        S = x.shape[2]  # Number of tokens per time-step
        x = rearrange(x, "b tt s d -> b (tt s) d")
        return x, B, TT, S

    def forward(self, td: TensorDict):
        x, B, TT, S = self._prepare_inputs(td)

        training_env_ids = td.get("training_env_ids", None)
        if training_env_ids is None:
            training_env_ids = torch.arange(B, device=td.device)
        else:
            training_env_ids = training_env_ids.reshape(B * TT)

        if TT == 1:  # rollout
            return self._forward_rollout(td, x, B, S, training_env_ids)
        else:  # training
            return self._forward_training(td, x, B, TT, S, training_env_ids)

    def _forward_rollout(self, td: TensorDict, x: torch.Tensor, B: int, S: int, training_env_ids: torch.Tensor):
        # 1. Add positional encoding
        max_num_envs = training_env_ids.max() + 1
        if max_num_envs > self.position_counter.size(0):
            # this block dynamically builds the cache buffers so that we don't need to know the num of envs upfront
            num_new_envs = max_num_envs - self.position_counter.size(0)
            # Expand all state buffers for new environments
            filler = torch.zeros(num_new_envs, *self.k_cache_rollout.shape[1:], device=self.k_cache_rollout.device)
            self.k_cache_rollout = torch.cat([self.k_cache_rollout, filler])
            self.v_cache_rollout = torch.cat([self.v_cache_rollout, filler])
            self.k_cache_training = self.k_cache_rollout.clone()
            self.v_cache_training = self.v_cache_rollout.clone()
            pos_filler = torch.zeros(num_new_envs, device=self.position_counter.device, dtype=torch.long)
            self.position_counter = torch.cat([self.position_counter, pos_filler])

        current_pos = self.position_counter[training_env_ids]
        pos_enc = self._get_positional_encoding(current_pos, self.output_dim)
        x = x + pos_enc.unsqueeze(1)  # Add to all S tokens for this timestep

        # 2. Process through transformer layers
        new_k_cache_list, new_v_cache_list = [], []
        pk_layers = self.k_cache_rollout[training_env_ids]
        pv_layers = self.v_cache_rollout[training_env_ids]

        for i, block in enumerate(self.blocks):
            pk = pk_layers[:, i]
            pv = pv_layers[:, i]
            x, new_k, new_v = block(x, pk, pv, mask=None)
            new_k_cache_list.append(new_k.unsqueeze(1))
            new_v_cache_list.append(new_v.unsqueeze(1))

        # 3. Truncate and update cache
        updated_k_cache = torch.cat(new_k_cache_list, dim=1)
        updated_v_cache = torch.cat(new_v_cache_list, dim=1)

        if updated_k_cache.shape[3] > self.max_cache_size:
            updated_k_cache = updated_k_cache[:, :, :, -self.max_cache_size :]
            updated_v_cache = updated_v_cache[:, :, :, -self.max_cache_size :]

        self.k_cache_rollout[training_env_ids] = updated_k_cache.detach()
        self.v_cache_rollout[training_env_ids] = updated_v_cache.detach()

        # 4. Store cache for training and get final output
        td.set("transformer_position", current_pos.detach())

        output = self.final_norm(x)
        pooled_output = self._pool_output(output, tt=1)
        td.set(self.out_key, pooled_output)

        # 5. Update and reset position counter
        self.position_counter[training_env_ids] += 1
        dones = td.get("dones", torch.zeros(B, device=td.device))
        truncateds = td.get("truncateds", torch.zeros(B, device=td.device))
        resets = torch.logical_or(dones.bool(), truncateds.bool()).squeeze()
        if resets.dim() > 1:
            resets = resets.squeeze(-1)
        self.position_counter[training_env_ids] *= (~resets.bool()).long()

        return td

    def _forward_training(
        self, td: TensorDict, x: torch.Tensor, B: int, TT: int, S: int, training_env_ids: torch.Tensor
    ):
        # 1. Add positional encoding
        # The cache is from the step before this sequence started
        start_pos = td["transformer_position"].view(B, TT)[:, 0]
        positions = start_pos.unsqueeze(1) + torch.arange(TT, device=x.device)
        pos_enc = self._get_positional_encoding(positions, self.output_dim)
        pos_enc = pos_enc.unsqueeze(2).expand(-1, -1, S, -1)
        pos_enc = rearrange(pos_enc, "b tt s d -> b (tt s) d")
        x = x + pos_enc

        # 2. Prepare causal mask
        q_len_tokens = TT * S
        cache_len_tokens = self.k_cache_training.shape[3]

        causal_mask_timesteps = torch.triu(torch.ones(TT, TT, device=x.device, dtype=torch.bool), diagonal=1)
        causal_mask = causal_mask_timesteps.repeat_interleave(S, dim=0).repeat_interleave(S, dim=1)
        full_mask = torch.cat(
            [
                torch.zeros(q_len_tokens, cache_len_tokens, device=x.device, dtype=torch.bool),
                causal_mask,
            ],
            dim=1,
        )

        # 3. Process through transformer layers
        pk_layers_at_start = self.k_cache_training[training_env_ids]
        pv_layers_at_start = self.v_cache_training[training_env_ids]

        new_k_cache_list, new_v_cache_list = [], []
        for i, block in enumerate(self.blocks):
            pk = pk_layers_at_start[:, i]
            pv = pv_layers_at_start[:, i]
            x, new_k, new_v = block(x, pk, pv, mask=full_mask)
            new_k_cache_list.append(new_k.unsqueeze(1))
            new_v_cache_list.append(new_v.unsqueeze(1))

        # 4. Truncate and update cache
        updated_k_cache = torch.cat(new_k_cache_list, dim=1)
        updated_v_cache = torch.cat(new_v_cache_list, dim=1)

        if updated_k_cache.shape[3] > self.max_cache_size:
            updated_k_cache = updated_k_cache[:, :, :, -self.max_cache_size :]
            updated_v_cache = updated_v_cache[:, :, :, -self.max_cache_size :]

        self.k_cache_training[training_env_ids] = updated_k_cache.detach()
        self.v_cache_training[training_env_ids] = updated_v_cache.detach()
        self.k_cache_rollout = self.k_cache_training.clone()
        self.v_cache_rollout = self.v_cache_training.clone()

        # 5. Get final output
        output = self.final_norm(x)
        output = rearrange(output, "b (tt s) d -> b tt s d", tt=TT)
        pooled_output = self._pool_output(output, tt=TT)

        pooled_output = rearrange(pooled_output, "b tt ... -> (b tt) ...")
        td.set(self.out_key, pooled_output)

        return td

    def get_agent_experience_spec(self) -> Composite:
        return Composite(
            {
                "transformer_position": UnboundedDiscrete(shape=torch.Size([]), dtype=torch.long),
                "dones": UnboundedDiscrete(shape=torch.Size([]), dtype=torch.float32),
                "truncateds": UnboundedDiscrete(shape=torch.Size([]), dtype=torch.float32),
            }
        )

    def get_memory(self):
        return self.k_cache_training, self.v_cache_training

    def set_memory(self, memory):
        """Cannot be called at the Policy level - use policy.<path_to_this_layer>.set_memory()"""
        self.k_cache_training, self.v_cache_training = memory[0], memory[1]

    def reset_memory(self):
        pass

    def initialize_to_environment(
        self,
        env,
        device,
    ) -> None:
        pass
