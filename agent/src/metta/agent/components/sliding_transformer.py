from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tensordict import TensorDict
from torchrl.data import Composite, UnboundedDiscrete

from metta.agent.components.component_config import ComponentConfig

from .transformer_common import (
    SinusoidalPositionEmbedding,
    ensure_mask_on_device,
    make_layer_norm,
)


class TransformerBlock(nn.Module):
    """A single block of the sliding-window transformer architecture."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        *,
        use_fused_layernorm: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.norm1 = make_layer_norm(d_model, use_fused_layernorm)
        self.norm2 = make_layer_norm(d_model, use_fused_layernorm)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

        self.norm_factor = self.head_dim**-0.5

    def forward(
        self,
        x: torch.Tensor,
        pk: Optional[torch.Tensor],
        pv: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, token_count, _ = x.shape

        x_norm = self.norm1(x)
        q = self.q_proj(x_norm).view(batch, token_count, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_norm).view(batch, token_count, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).view(batch, token_count, self.num_heads, self.head_dim).transpose(1, 2)

        if pk is not None and pv is not None:
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)

        attn_weights = (q @ k.transpose(-2, -1)) * self.norm_factor
        if mask is not None:
            mask = ensure_mask_on_device(mask.to(dtype=torch.bool), device=attn_weights.device)
            attn_weights = attn_weights.masked_fill(mask, float("-inf"))
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = (attn_weights @ v).transpose(1, 2).contiguous().view(batch, token_count, self.d_model)

        x = x + self.out_proj(attn_output)
        x = x + self.ffn(self.norm2(x))

        return x, k, v


class SlidingTransformer(nn.Module):
    """Sliding-window transformer with cached key/value memory."""

    def __init__(
        self,
        *,
        in_key: str,
        out_key: str,
        input_dim: int,
        hidden_size: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        max_cache_size: int,
        pool: Literal["cls", "mean", "none"],
        use_fused_layernorm: bool = False,
        env=None,
    ) -> None:
        super().__init__()
        self.in_key = in_key
        self.out_key = out_key
        self.hidden_size = hidden_size
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_cache_size = max_cache_size
        self.pool = pool

        if input_dim == hidden_size:
            self.input_proj = nn.Identity()
        else:
            self.input_proj = nn.Linear(input_dim, hidden_size)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size,
                    num_heads,
                    d_ff,
                    use_fused_layernorm=use_fused_layernorm,
                )
                for _ in range(num_layers)
            ]
        )

        self.last_action_proj = nn.Linear(2, hidden_size)
        self.reward_proj = nn.Linear(1, hidden_size)
        self.dones_truncateds_proj = nn.Linear(1, hidden_size)

        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.final_norm = make_layer_norm(hidden_size, use_fused_layernorm)
        self.position_embedding = SinusoidalPositionEmbedding(hidden_size)

        head_dim = hidden_size // num_heads
        cache_shape = (0, num_layers, num_heads, max_cache_size, head_dim)
        self.register_buffer("k_cache", torch.empty(cache_shape))
        self.register_buffer("v_cache", torch.empty(cache_shape))
        self.register_buffer("k_cache_training", torch.empty(cache_shape))
        self.register_buffer("v_cache_training", torch.empty(cache_shape))
        self.register_buffer("position_counter", torch.zeros(0, dtype=torch.long))

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

        tokens_per_step = x.shape[2]
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
            pos_enc = self.position_embedding(current_pos, dtype=x.dtype)
            x = x + pos_enc.unsqueeze(1)

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
            if self.pool == "cls":
                pooled_output = output[:, 0, :]  # Select CLS token output
            elif self.pool == "mean":
                pooled_output = output.mean(dim=1)
            elif self.pool == "none":
                pooled_output = rearrange(output, "b s d -> b (s d)")
            else:
                raise ValueError(f"Unsupported pool mode: {self.pool}")
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
            pos_enc = self.position_embedding(positions, dtype=x.dtype)
            pos_enc = pos_enc.unsqueeze(2).expand(-1, -1, tokens_per_step, -1)
            pos_enc = rearrange(pos_enc, "b tt s d -> b (tt s) d")
            x = x + pos_enc

            # 2. Prepare causal mask
            q_len_tokens = TT * tokens_per_step
            # pk_layers_at_start = td["past_key"].view(B, TT, *td["past_key"].shape[1:])[:, 0]
            cache_len_tokens = self.k_cache_training.shape[3]

            causal_mask_timesteps = torch.triu(torch.ones(TT, TT, device=x.device, dtype=torch.bool), diagonal=1)
            causal_mask = causal_mask_timesteps.repeat_interleave(tokens_per_step, dim=0).repeat_interleave(
                tokens_per_step, dim=1
            )
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
            if self.pool == "cls":
                pooled_output = output[:, :, 0, :]  # Select CLS token
            elif self.pool == "mean":
                pooled_output = output.mean(dim=2)  # pool over S dimension
            elif self.pool == "none":
                pooled_output = rearrange(output, "b tt s d -> b tt (s d)")
            else:
                raise ValueError(f"Unsupported pool mode: {self.pool}")

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
    use_fused_layernorm: bool = False

    def make_component(self, env=None):
        from .transformer_core import TransformerBackboneConfig, TransformerBackboneVariant

        hidden_size = self.output_dim
        feedforward_dim = hidden_size * self.ff_mult
        backbone_cfg = TransformerBackboneConfig(
            name=self.name,
            in_key=self.in_key,
            out_key=self.out_key,
            variant=TransformerBackboneVariant.SLIDING,
            latent_size=self.input_dim,
            hidden_size=hidden_size,
            num_layers=self.num_layers,
            n_heads=self.num_heads,
            d_ff=feedforward_dim,
            dropout=0.0,
            attn_dropout=0.0,
            max_cache_size=self.max_cache_size,
            pool=self.pool,
            use_fused_layernorm=self.use_fused_layernorm,
        )
        return backbone_cfg.make_component(env)
