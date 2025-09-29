from typing import Literal

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

    def forward(self, x, context, mask):
        B, T, E = x.shape  # Batch, Tokens, EmbedDim
        _, S, _ = context.shape  # Batch, Latents, EmbedDim

        # LayerNorm and Q projection from input
        x_norm = self.norm1(x)
        q = self.q_proj(x_norm).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # For cross attention, K and V come from context (latents)
        k = self.k_proj(context).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention calculation
        attn_weights = (q @ k.transpose(-2, -1)) * self.norm_factor
        if mask is not None:
            # Mask shape is (B, T, S) -> needs to be (B, 1, T, S) for broadcasting
            attn_weights = attn_weights.masked_fill(mask.unsqueeze(1), float("-inf"))
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = (attn_weights @ v).transpose(1, 2).contiguous().view(B, T, E)

        # Residual connection and FFN
        x = x + self.out_proj(attn_output)
        x = x + self.ffn(self.norm2(x))

        return x


class CrossAttentionTransformerConfig(ComponentConfig):
    in_key: str
    out_key: str
    name: str = "cross_attention_transformer"
    output_dim: int = 16
    input_dim: int = 64
    num_heads: int = 1
    ff_mult: int = 4
    num_layers: int = 2
    pool: Literal["cls", "mean", "none"] = "mean"
    num_latents: int = 3
    num_embeddings: int = 50

    def make_component(self, env=None):
        return CrossAttentionTransformer(config=self, env=env)


class CrossAttentionTransformer(nn.Module):
    """ """

    def __init__(self, config: CrossAttentionTransformerConfig, env):
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

        # Latent tokens for cross-attention
        self.latents = nn.Embedding(self.config.num_embeddings, self.output_dim)
        # Initialize with small weights
        self.latents.weight.data.uniform_(-0.01, 0.01)

    def forward(self, td: TensorDict):
        B_times_TT = td.batch_size.numel()
        if "bptt" in td.keys() and td["bptt"][0] != 1:
            TT = td["bptt"][0]
        else:
            TT = 1
        B = B_times_TT // TT

        # 1. Read inputs and prepare them for the transformer
        x = td[self.in_key]  # observation token(s)
        x = self.input_proj(x)

        empty_tensor = torch.zeros(B_times_TT, device=td.device)
        reward = td.get("reward", empty_tensor)
        last_actions = td.get("last_actions", torch.zeros(B_times_TT, 2, device=td.device))
        dones = td.get("dones", empty_tensor)
        truncateds = td.get("truncateds", empty_tensor)

        # Handle variable observation shapes [B*TT, E] -> [B*TT, 1, E]
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

        x = rearrange(x, "b tt s d -> b (tt s) d")

        # Get latent tokens for cross-attention
        latents = self.latents.weight[: self.config.num_latents]
        latents = latents.unsqueeze(0).expand(B, -1, -1)

        # Process through transformer layers
        for block in self.blocks:
            x = block(x, latents, mask=None)

        # Get final output
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
                "dones": UnboundedDiscrete(shape=torch.Size([]), dtype=torch.float32),
                "truncateds": UnboundedDiscrete(shape=torch.Size([]), dtype=torch.float32),
            }
        )
