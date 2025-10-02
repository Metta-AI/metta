from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tensordict import TensorDict
from torchrl.data import Composite, UnboundedDiscrete

from metta.agent.components.component_config import ComponentConfig

from metta.agent.components.transformer_module import sinusoidal_position_encoding


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
    num_layers: int = 2
    max_cache_size: int = 80
    pool: Literal["cls", "mean", "none"] = "mean"

    def make_component(self, env=None):
        return SlidingTransformer(config=self, env=env)


class SlidingTransformer(nn.Module):
    """ """

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

        self.last_action_proj = nn.Linear(2, self.output_dim)
        self.reward_proj = nn.Linear(1, self.output_dim)
        self.dones_truncateds_proj = nn.Linear(1, self.output_dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.output_dim))
        self.final_norm = nn.LayerNorm(self.output_dim)

        # State buffers (KV cache per layer)
        self.register_buffer(
            "k_cache",
            torch.empty(
                0,
                self.num_layers,
                self.config.num_heads,
                self.max_cache_size,
                self.config.output_dim // self.config.num_heads,
            ),
        )
        self.register_buffer(
            "v_cache",
            torch.empty(
                0,
                self.num_layers,
                self.config.num_heads,
                self.max_cache_size,
                self.config.output_dim // self.config.num_heads,
            ),
        )
        self.register_buffer(
            "k_cache_training",
            torch.empty(
                0,
                self.num_layers,
                self.config.num_heads,
                self.max_cache_size,
                self.config.output_dim // self.config.num_heads,
            ),
        )
        self.register_buffer(
            "v_cache_training",
            torch.empty(
                0,
                self.num_layers,
                self.config.num_heads,
                self.max_cache_size,
                self.config.output_dim // self.config.num_heads,
            ),
        )
        self.register_buffer("position_counter", torch.zeros(0, dtype=torch.long))

        # av need to handle switching between rollout/train and eval

    def forward(self, td: TensorDict):
        B = td.batch_size.numel()
        if td["bptt"][0] != 1:
            TT = td["bptt"][0]
        else:
            TT = 1
        B = B // TT

        training_env_ids = td.get("training_env_ids", None)
        if training_env_ids is None:
            training_env_ids = torch.arange(B, device=td.device)
        else:
            training_env_ids = training_env_ids.reshape(B * TT)  # av why reshape this? should already be B*TT

        # 1. Read inputs and prepare them for the transformer
        x = td[self.in_key]  # observation token(s)

        x = self.input_proj(x)

        empty_tensor = torch.zeros(B * TT, device=td.device)
        reward = td.get("reward", empty_tensor)  # scalar
        last_actions = td.get("last_actions", torch.zeros(B * TT, 2, device=td.device))
        dones = td.get("dones", empty_tensor)
        truncateds = td.get("truncateds", empty_tensor)

        # Handle variable observation shapes [B, E] -> [B, 1, E]
        if x.dim() == 2 + (TT > 1):
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

        if TT == 1:  # rollout
            # 1. Add positional encoding
            max_num_envs = training_env_ids.max() + 1
            if max_num_envs > self.position_counter.size(0):
                num_new_envs = max_num_envs - self.position_counter.size(0)
                # Expand all state buffers for new environments
                k_filler = torch.zeros(num_new_envs, *self.k_cache.shape[1:], device=self.k_cache.device)
                self.k_cache = torch.cat([self.k_cache, k_filler])
                self.v_cache = torch.cat([self.v_cache, k_filler])
                self.k_cache_training = self.k_cache.clone()
                self.v_cache_training = self.v_cache.clone()
                pos_filler = torch.zeros(num_new_envs, device=self.position_counter.device, dtype=torch.long)
                self.position_counter = torch.cat([self.position_counter, pos_filler])

            current_pos = self.position_counter[training_env_ids]
            pos_enc = sinusoidal_position_encoding(current_pos, self.output_dim)
            x = x + pos_enc.unsqueeze(1)  # Add to all S tokens for this timestep

            # 2. Process through transformer layers
            new_k_cache_list, new_v_cache_list = [], []
            pk_layers = self.k_cache[training_env_ids]
            pv_layers = self.v_cache[training_env_ids]

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

            self.k_cache[training_env_ids] = updated_k_cache.detach()
            self.v_cache[training_env_ids] = updated_v_cache.detach()

            # 4. Store cache for training and get final output
            td.set("transformer_position", current_pos.detach())

            output = self.final_norm(x)
            if self.config.pool == "cls":
                pooled_output = output[:, 0, :]  # Select CLS token output
            elif self.config.pool == "mean":
                pooled_output = output.mean(dim=1)
            elif self.config.pool == "none":
                pooled_output = rearrange(output, "b s d -> b (s d)")
            else:
                raise ValueError(f"Unsupported pool mode: {self.config.pool}")
            td.set(self.out_key, pooled_output)

            # 5. Update and reset position counter
            self.position_counter[training_env_ids] += 1
            resets = torch.logical_or(dones.bool(), truncateds.bool()).squeeze()
            if resets.dim() > 1:
                resets = resets.squeeze(-1)
            self.position_counter[training_env_ids] *= (~resets.bool()).long()

            return td

        else:  # training
            # 1. Add positional encoding
            # The cache is from the step before this sequence started
            start_pos = td["transformer_position"].view(B, TT)[:, 0]
            positions = start_pos.unsqueeze(1) + torch.arange(TT, device=x.device)
            pos_enc = sinusoidal_position_encoding(positions, self.output_dim)
            pos_enc = pos_enc.unsqueeze(2).expand(-1, -1, S, -1)
            pos_enc = rearrange(pos_enc, "b tt s d -> b (tt s) d")
            x = x + pos_enc

            # 2. Prepare causal mask
            q_len_tokens = TT * S
            # pk_layers_at_start = td["past_key"].view(B, TT, *td["past_key"].shape[1:])[:, 0]
            cache_len_tokens = self.k_cache_training.shape[3]  # av should this use some of the code in the line above?

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

            # 3. Truncate and update cache
            updated_k_cache = torch.cat(new_k_cache_list, dim=1)
            updated_v_cache = torch.cat(new_v_cache_list, dim=1)

            if updated_k_cache.shape[3] > self.max_cache_size:
                updated_k_cache = updated_k_cache[:, :, :, -self.max_cache_size :]
                updated_v_cache = updated_v_cache[:, :, :, -self.max_cache_size :]

            self.k_cache_training[training_env_ids] = updated_k_cache.detach()
            self.v_cache_training[training_env_ids] = updated_v_cache.detach()
            self.k_cache = self.k_cache_training.clone()
            self.v_cache = self.v_cache_training.clone()

            # 4. Get final output
            output = self.final_norm(x)
            output = rearrange(output, "b (tt s) d -> b tt s d", tt=TT)
            if self.config.pool == "cls":
                pooled_output = output[:, :, 0, :]  # Select CLS token
            elif self.config.pool == "mean":
                pooled_output = output.mean(dim=2)  # pool over S dimension
            elif self.config.pool == "none":
                pooled_output = rearrange(output, "b tt s d -> b tt (s d)")
            else:
                raise ValueError(f"Unsupported pool mode: {self.config.pool}")

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
        return self.k_cache, self.v_cache

    def set_memory(self, memory):
        """Cannot be called at the Policy level - use policy.<path_to_this_layer>.set_memory()"""
        self.k_cache, self.v_cache = memory[0], memory[1]

    def reset_memory(self):
        pass

    def _reset_memory(self):
        self.k_cache.fill_(0)
        self.v_cache.fill_(0)

    def initialize_to_environment(
        self,
        env,
        device,
    ) -> None:
        pass
