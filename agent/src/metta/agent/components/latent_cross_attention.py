from typing import Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tensordict import TensorDict
from torchrl.data import Composite, UnboundedDiscrete

from metta.agent.components.component_config import ComponentConfig


class CrossAttentionBlock(nn.Module):
    """A transformer block with cross-attention and latent updates."""

    def __init__(self, output_dim, num_heads, ff_mult):
        super().__init__()
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        assert self.head_dim * num_heads == output_dim, "output_dim must be divisible by num_heads"

        # Projections for input x
        self.q_proj_x = nn.Linear(output_dim, output_dim)
        self.k_proj_x = nn.Linear(output_dim, output_dim)
        self.v_proj_x = nn.Linear(output_dim, output_dim)
        self.out_proj_x = nn.Linear(output_dim, output_dim)

        # Projections for latents
        self.q_proj_l = nn.Linear(output_dim, output_dim)
        self.k_proj_l = nn.Linear(output_dim, output_dim)
        self.v_proj_l = nn.Linear(output_dim, output_dim)
        self.out_proj_l = nn.Linear(output_dim, output_dim)

        # LayerNorms
        self.norm_x1 = nn.LayerNorm(output_dim)
        self.norm_x2 = nn.LayerNorm(output_dim)
        self.norm_l1 = nn.LayerNorm(output_dim)
        self.norm_l2 = nn.LayerNorm(output_dim)

        # Feedforward networks
        ff_hidden_dim = output_dim * ff_mult
        self.ffn_x = nn.Sequential(
            nn.Linear(output_dim, ff_hidden_dim),
            nn.GELU(),
            nn.Linear(ff_hidden_dim, output_dim),
        )
        self.ffn_l = nn.Sequential(
            nn.Linear(output_dim, ff_hidden_dim),
            nn.GELU(),
            nn.Linear(ff_hidden_dim, output_dim),
        )

        self.norm_factor = self.head_dim**-0.5

    def forward(self, x, latents, x_mask=None, latent_mask=None):
        B, T, E = x.shape
        _B, N, _E = latents.shape

        # 1. Update latents based on x (cross-attention)
        latents_norm = self.norm_l1(latents)
        x_norm_for_latents = self.norm_x1(x)

        q_l = self.q_proj_l(latents_norm).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k_x = self.k_proj_x(x_norm_for_latents).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v_x = self.v_proj_x(x_norm_for_latents).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights_l = (q_l @ k_x.transpose(-2, -1)) * self.norm_factor
        if x_mask is not None:
            # x_mask should be (B, 1, 1, T)
            attn_weights_l = attn_weights_l.masked_fill(x_mask, float("-inf"))
        attn_weights_l = F.softmax(attn_weights_l, dim=-1)
        attn_output_l = (attn_weights_l @ v_x).transpose(1, 2).contiguous().view(B, N, E)
        latents = latents + self.out_proj_l(attn_output_l)

        # 2. Update x based on updated latents (cross-attention)
        x_norm = self.norm_x1(x)
        latents_norm_for_x = self.norm_l1(latents)

        q_x = self.q_proj_x(x_norm).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k_l = self.k_proj_l(latents_norm_for_x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v_l = self.v_proj_l(latents_norm_for_x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights_x = (q_x @ k_l.transpose(-2, -1)) * self.norm_factor
        if latent_mask is not None:
            # latent_mask should be (B, 1, T, N)
            attn_weights_x = attn_weights_x.masked_fill(latent_mask, float("-inf"))
        attn_weights_x = F.softmax(attn_weights_x, dim=-1)
        attn_output_x = (attn_weights_x @ v_l).transpose(1, 2).contiguous().view(B, T, E)
        x = x + self.out_proj_x(attn_output_x)

        # 3. Apply FFNs
        x = x + self.ffn_x(self.norm_x2(x))
        latents = latents + self.ffn_l(self.norm_l2(latents))

        return x, latents


class LatentCrossAttentionConfig(ComponentConfig):
    in_key: str
    out_key: str
    name: str = "latent_cross_attention"
    output_dim: int = 16
    input_dim: int = 64
    num_heads: int = 1
    ff_mult: int = 4
    num_layers: int
    num_latents: int = 3
    max_latents: int = 50
    pool: Literal["cls", "mean", "none"] = "mean"
    last_action_dim: int = 2

    def make_component(self, env=None):
        return LatentCrossAttention(config=self, env=env)


class LatentCrossAttention(nn.Module):
    def __init__(self, config: LatentCrossAttentionConfig, env):
        super().__init__()
        self.config = config
        self.in_key = self.config.in_key
        self.out_key = self.config.out_key

        self.output_dim = self.config.output_dim
        self.num_layers = self.config.num_layers

        input_dim = self.config.input_dim
        if input_dim == self.output_dim:
            self.input_proj = nn.Identity()
        else:
            self.input_proj = nn.Linear(input_dim, self.output_dim)

        self.latent_embeddings = nn.Embedding(self.config.max_latents, self.output_dim)
        self.latent_embeddings.weight.data.uniform_(-0.02, 0.02)
        self.register_buffer("latent_indices", torch.arange(self.config.num_latents))

        self.blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
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
        latents_shape = (0, self.num_layers, self.config.num_latents, self.output_dim)
        self.register_buffer("latents_cache_rollout", torch.empty(latents_shape))
        self.register_buffer("latents_cache_training", torch.empty(latents_shape))

    def _get_initial_latents(self, batch_size, device):
        initial_latents = self.latent_embeddings(self.latent_indices)
        return initial_latents.unsqueeze(0).expand(batch_size, -1, -1)

    def _pool_output(self, output: torch.Tensor, tt: int) -> torch.Tensor:
        if self.config.pool == "cls":
            return output[:, :, 0, :] if tt > 1 else output[:, 0, :]
        elif self.config.pool == "mean":
            return output.mean(dim=2) if tt > 1 else output.mean(dim=1)
        elif self.config.pool == "none":
            return rearrange(output, "b tt s d -> b tt (s d)") if tt > 1 else rearrange(output, "b s d -> b (s d)")
        else:
            raise ValueError(f"Unsupported pool mode: {self.config.pool}")

    def _prepare_inputs(self, td: TensorDict) -> Tuple[torch.Tensor, int, int, int]:
        B_flat = td.batch_size.numel()
        TT = td.get("bptt", [1])[0]
        B = B_flat // TT

        x = td[self.in_key]
        x = self.input_proj(x)

        empty_tensor = torch.zeros(B * TT, device=td.device)
        reward = td.get("reward", empty_tensor)
        last_actions = td.get("last_actions", torch.zeros(B * TT, self.config.last_action_dim, device=td.device))
        dones = td.get("dones", empty_tensor)
        truncateds = td.get("truncateds", empty_tensor)

        if x.dim() == 2:
            x = x.unsqueeze(-2)

        x = rearrange(x, "(b tt) ... d -> b tt (...) d", tt=TT)
        reward = rearrange(reward, "(b tt) ... -> b tt (...) 1", tt=TT)
        last_actions = rearrange(last_actions, "(b tt) ... d -> b tt (...) d", tt=TT)
        resets = torch.logical_or(dones.bool(), truncateds.bool()).float()
        resets = rearrange(resets, "(b tt) ... -> b tt (...) 1", tt=TT)

        reward_token = self.reward_proj(reward)
        reset_token = self.dones_truncateds_proj(resets)
        reward_reset_token = (reward_token + reset_token).view(B, TT, 1, self.output_dim)
        action_token = self.last_action_proj(last_actions.float()).view(B, TT, 1, self.output_dim)

        cls_token = self.cls_token.expand(B, TT, -1, -1)
        x = torch.cat([cls_token, x, reward_reset_token, action_token], dim=2)

        S = x.shape[2]
        x = rearrange(x, "b tt s d -> (b tt) s d")
        return x, B, TT, S

    def forward(self, td: TensorDict):
        x, B, TT, S = self.prepare_inputs_and_expand_cache(td)

        if TT == 1:
            latents = self.latents_cache_rollout
            new_latents = []
            for i, block in enumerate(self.blocks):
                x, layer_latents = block(x, latents[:, i])
                new_latents.append(layer_latents.unsqueeze(1))
            self.latents_cache_rollout = torch.cat(new_latents, dim=1).detach()
        else:
            latents = self.latents_cache_training
            x = rearrange(x, "(b tt) s d -> b tt s d", tt=TT)
            outputs = []
            for t in range(TT):
                x_t = x[:, t]
                new_latents_t = []
                for i, block in enumerate(self.blocks):
                    x_t, layer_latents = block(x_t, latents[:, i])
                    new_latents_t.append(layer_latents.unsqueeze(1))
                latents = torch.cat(new_latents_t, dim=1)
                outputs.append(x_t.unsqueeze(1))
            x = torch.cat(outputs, dim=1)
            self.latents_cache_training = latents.detach()
            self.latents_cache_rollout = self.latents_cache_training.clone()

        output = self.final_norm(x)
        pooled_output = self._pool_output(output, tt=TT)
        td.set(self.out_key, rearrange(pooled_output, "b tt ... -> (b tt) ...") if TT > 1 else pooled_output)
        return td

    def prepare_inputs_and_expand_cache(self, td: TensorDict):
        x, B, TT, S = self._prepare_inputs(td)

        if B > self.latents_cache_rollout.size(0):
            num_new = B - self.latents_cache_rollout.size(0)
            initial_latents = self._get_initial_latents(num_new, x.device)
            new_cache = initial_latents.unsqueeze(1).expand(-1, self.num_layers, -1, -1)
            self.latents_cache_rollout = torch.cat([self.latents_cache_rollout, new_cache], dim=0)
            self.latents_cache_training = self.latents_cache_rollout.clone()

        if TT == 1:
            resets = (
                td.get("dones", torch.zeros(B, 1, device=x.device)).bool()
                | td.get("truncateds", torch.zeros(B, 1, device=x.device)).bool()
            )
            if resets.any():
                initial_latents = self._get_initial_latents(resets.sum(), x.device)
                initial_cache = initial_latents.unsqueeze(1).expand(-1, self.num_layers, -1, -1)
                self.latents_cache_rollout[resets.squeeze()] = initial_cache

        return x, B, TT, S

    def get_agent_experience_spec(self) -> Composite:
        return Composite(
            {
                "dones": UnboundedDiscrete(shape=torch.Size([]), dtype=torch.float32),
                "truncateds": UnboundedDiscrete(shape=torch.Size([]), dtype=torch.float32),
            }
        )

    def get_memory(self):
        return self.latents_cache_training

    def set_memory(self, memory):
        self.latents_cache_training = memory

    def reset_memory(self):
        pass

    def initialize_to_environment(self, env, device) -> None:
        pass
