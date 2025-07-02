from typing import List

import pufferlib.models
import pufferlib.pytorch
import torch
import torch.nn as nn
from einops import rearrange


class Recurrent(pufferlib.models.LSTMWrapper):
    def __init__(self, env, policy=None, cnn_channels=128, input_size=384, hidden_size=384):
        if policy is None:
            policy = Policy(env, cnn_channels=cnn_channels, hidden_size=hidden_size)
        super().__init__(env, policy, input_size, hidden_size)

    def forward(self, observations, state):
        """Forward function for inference. 3x faster than using LSTM directly"""
        # Either [B, T, H, W, C] or [B, H, W, C]
        if len(observations.shape) == 5:
            x = rearrange(observations, "b t h w c -> b t c h w").float()
            x[:] /= self.policy.max_vec
            return self.forward_train(x, state)
        else:
            x = rearrange(observations, "b h w c -> b c h w").float() / self.policy.max_vec
        hidden = self.policy.encode_observations(x, state=state)
        h = state.lstm_h
        c = state.lstm_c

        # TODO: Don't break compile
        if h is not None:
            if len(h.shape) == 3:
                h = h.squeeze()
            if len(c.shape) == 3:
                c = c.squeeze()
            assert h.shape[0] == c.shape[0] == observations.shape[0], "LSTM state must be (h, c)"
            lstm_state = (h, c)
        else:
            lstm_state = None

        # hidden = self.pre_layernorm(hidden)
        hidden, c = self.cell(hidden, lstm_state)
        # hidden = self.post_layernorm(hidden)
        state.hidden = hidden
        state.lstm_h = hidden
        state.lstm_c = c
        logits, values = self.policy.decode_actions(hidden)
        return logits, values


class Policy(nn.Module):
    """Stronger drop‑in replacement for the original CNN+MLP policy.

    **Key ideas**
    -------------
    1. **Richer spatial features.**  A small Conv‑Stem extracts low‑level texture
       information before patchification.
    2. **Lightweight ViT encoder.**  A class token summarises the visual scene
       through multi‑head self‑attention.  Depth/width are modest so it still
       runs in real time on a single 4090.
    3. **Separate proprioceptive stream.**  The 34‑dim agent‑centric vector is
       encoded with a two‑layer MLP, mirroring the original design.
    4. **Late fusion & projection.**  Visual CLS + self vector → linear project
       to a unified embedding that matches *hidden_size* expected by the LSTM
       wrapper.
    5. **Actor‑Critic heads unchanged.**  Keeping the interface identical means
       you can swap the old policy for this one with **no other code changes**.
    """

    def __init__(
        self,
        env,
        patch_size: int = 2,
        cnn_channels: int = 128,
        hidden_size: int = 384,
        depth: int = 3,
        num_heads: int = 6,
        mlp_ratio: float = 3.0,
        **kw,
    ):
        super().__init__()
        self.is_continuous = False
        self.hidden_size = hidden_size

        # ------------------------------------------------------------------
        #  Image → local features → patches → Transformer
        # ------------------------------------------------------------------
        in_channels = 34  # observation feature planes
        self.conv_stem = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Conv2d(in_channels, cnn_channels, 5, stride=2, padding=2)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(cnn_channels, cnn_channels, 3, stride=1, padding=1)),
            nn.ReLU(),
        )

        self.patch_size = patch_size
        self.proj = pufferlib.pytorch.layer_init(
            nn.Conv2d(cnn_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        )

        # class token & positional embeddings -------------------------------------------------
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.pos_emb = nn.Parameter(torch.zeros(1, 1000, hidden_size))  # 1000 >> max patch count
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        # Transformer encoder ----------------------------------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=int(hidden_size * mlp_ratio),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # ------------------------------------------------------------------
        #  Proprioceptive / self features
        # ------------------------------------------------------------------
        self.self_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(34, hidden_size)),
            nn.GELU(),
        )

        # ------------------------------------------------------------------
        #  Fusion + heads
        # ------------------------------------------------------------------
        self.fuse_proj = pufferlib.pytorch.layer_init(nn.Linear(hidden_size * 2, hidden_size))

        action_nvec = env.single_action_space.nvec
        self.actor: List[nn.Linear] = nn.ModuleList(
            [pufferlib.pytorch.layer_init(nn.Linear(hidden_size, n), std=0.01) for n in action_nvec]
        )
        self.value = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, 1), std=1)

        # Keep the original scaling vector so normalization remains identical
        # TODO - fix magic numbers!
        # fmt: off
        max_vec = torch.tensor([
            1, 10, 30, 1, 1, 255,
            100, 100, 100, 100, 100, 100, 100, 100,
            1, 1, 1, 10, 1,
            100, 100, 100, 100, 100, 100, 100, 100,
            1, 1, 1, 1, 1, 1, 1
        ], dtype=torch.float).reshape(1, 34, 1, 1)
        # fmt: on

        self.register_buffer("max_vec", max_vec)

    def encode_observations(self, observations: torch.Tensor, state=None) -> torch.Tensor:
        """Maps raw env observations → latent *hidden_size* vector."""
        x = observations
        B = x.size(0)

        # 1) Conv stem -----------------------------------------------------------------------
        x = self.conv_stem(x)  # [B, C', H', W']

        # 2) Patchify ------------------------------------------------------------------------
        x = self.proj(x)  # [B, D, h, w]
        x = rearrange(x, "b d h w -> b (h w) d")

        # 3) Add CLS & positional encoding ---------------------------------------------------
        cls = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        tokens = torch.cat([cls, x], dim=1)  # [B, 1+N, D]
        tokens = tokens + self.pos_emb[:, : tokens.size(1)]  # broadcast

        # 4) Transformer ---------------------------------------------------------------------
        tokens = self.transformer(tokens)  # [B, 1+N, D]
        vis_feat = tokens[:, 0]  # CLS token

        # 5) Self vector ---------------------------------------------------------------------
        self_vec = rearrange(observations, "b c h w -> b h w c")[:, 5, 5, :].float()
        self_feat = self.self_encoder(self_vec)

        # 6) Fuse & project ------------------------------------------------------------------
        fused = torch.cat([vis_feat, self_feat], dim=1)  # [B, 2D]
        fused = self.fuse_proj(fused)  # [B, D]
        return fused

    def decode_actions(self, hidden: torch.Tensor):
        logits = [dec(hidden) for dec in self.actor]
        value = self.value(hidden)
        return logits, value

    # Convenience entry point if you want to use the policy *without* a wrapper
    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations, state)
        logits, value = self.decode_actions(hidden)
        return (logits, value), hidden
