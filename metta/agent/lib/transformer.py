import os
from functools import lru_cache

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

from metta.agent.lib.attention import AttentionBlock
from metta.agent.lib.metta_layer import LayerBase


class FFBlock(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, *, activation: nn.Module | None = None):
        super().__init__()
        self.d_model = in_features
        self.hidden_features = hidden_features

        self.activation = activation if activation is not None else nn.ReLU()
        self.up_proj = nn.Linear(in_features, hidden_features, bias=True)
        self.down_proj = nn.Linear(hidden_features, in_features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up_proj(x)
        x = self.activation(x)
        x = self.down_proj(x)
        return x


class GLUBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        *,
        activation: nn.Module | None = None,
    ):
        super().__init__()
        self.d_model = in_features
        self.hidden_features = hidden_features
        self.activation = activation if activation is not None else nn.ReLU()

        self.up_proj = nn.Linear(in_features, hidden_features * 2, bias=True)
        self.down_proj = nn.Linear(hidden_features, in_features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up_proj(x)

        x, gate = torch.chunk(x, 2, -1)
        x = self.activation(x) * gate
        x = self.down_proj(x)
        return x


class GatingMechanism(torch.nn.Module):
    def __init__(self, hidden_features, bg=0.1):
        super(GatingMechanism, self).__init__()
        self.Wr = torch.nn.Linear(hidden_features, hidden_features)
        self.Ur = torch.nn.Linear(hidden_features, hidden_features)
        self.Wz = torch.nn.Linear(hidden_features, hidden_features)
        self.Uz = torch.nn.Linear(hidden_features, hidden_features)
        self.Wg = torch.nn.Linear(hidden_features, hidden_features)
        self.Ug = torch.nn.Linear(hidden_features, hidden_features)
        self.bg = bg

        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()

    def forward(self, x, y):
        r = self.sigmoid(self.Wr(y) + self.Ur(x))
        z = self.sigmoid(self.Wz(y) + self.Uz(x) - self.bg)
        h = self.tanh(self.Wg(y) + self.Ug(torch.mul(r, x)))
        g = torch.mul(1 - z, x) + torch.mul(z, h)
        return g


class TransformerLayer(nn.Module):
    def __init__(
        self,
        num_heads: int,
        hidden_features: int,
        ffn_size: int,
        *,
        activation: nn.Module | None = None,
        glu: bool = True,
        gtrxl_gate: bool = True,
    ):
        super().__init__()
        self.gtrxl_gate = gtrxl_gate

        self.attention_norm = nn.LayerNorm(hidden_features)
        self.attention = AttentionBlock(num_heads, hidden_features)

        self.ffn_norm = nn.LayerNorm(hidden_features)
        ff_block = GLUBlock if glu else FFBlock
        self.ffn = ff_block(hidden_features, ffn_size, activation=activation)

        if gtrxl_gate:
            self.attention_gate = GatingMechanism(hidden_features)
            self.ffn_gate = GatingMechanism(hidden_features)

    def forward(self, x: Tensor, time_steps: Tensor, block_mask: BlockMask | None = None) -> Tensor:
        attention_input = self.attention_norm(x)
        attention = self.attention(attention_input, time_steps, block_mask=block_mask)
        x = self.attention_gate(x, attention) if self.gtrxl_gate else x + attention

        feed_forward_input = self.ffn_norm(x)
        feed_forward = self.ffn(feed_forward_input)
        x = self.ffn_gate(x, feed_forward) if self.gtrxl_gate else x + feed_forward

        return x


@lru_cache()
def _create_block_mask(seq_len: int) -> BlockMask:
    """
    Returns a block mask for flex attention, this is expensive and can't be generated inside torch compile.
    """

    def causal(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    return create_block_mask(causal, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len)


_compile = os.name == 'posix'
_flex_attention = os.name == 'posix'


class TransformerCore(LayerBase):
    def __init__(self, obs_shape, hidden_size: int, **cfg):
        super().__init__(**cfg)
        self.obs_shape = obs_shape
        self.hidden_size = hidden_size

        self.num_layers = self._nn_params["num_layers"]
        self.num_heads = self._nn_params["num_heads"]
        self.ffn_size = self._nn_params["ffn_size"]
        self.context_size = self._nn_params["context_size"]

        self.activation = nn.ReLU()
        self.glu = False
        self.gtrxl_gate = self._nn_params.get("gtrxl_gate", False)

    def _initialize(self):
        self._out_tensor_shape = [self.hidden_size]

        layers = []
        for _ in range(self.num_layers):
            layers.append(
                TransformerLayer(
                    self.num_heads,
                    self.hidden_size,
                    self.ffn_size,
                    activation=self.activation,
                    glu=self.glu,
                    gtrxl_gate=self.gtrxl_gate,
                )
            )
        self.layers = nn.ModuleList(layers)
        self.layer_norm = nn.LayerNorm(self.hidden_size)

    def ensure_kv_cache(self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32):
        for layer in self.layers:
            layer.attention.ensure_kv_cache(batch_size, self.context_size, device, dtype)

    def _forward(self, td: TensorDict) -> TensorDict:
        x = td["x"]
        hidden = td[self._sources[0]["name"]]

        x_shape, space_shape = x.shape, self.obs_shape
        x_n, space_n = len(x_shape), len(space_shape)
        if x_shape[-space_n:] != space_shape:
            raise ValueError("Invalid input tensor shape", x.shape)

        if x_n == space_n + 1:
            B, TT = x_shape[0], 1
        elif x_n == space_n + 2:
            B, TT = x_shape[:2]
        else:
            raise ValueError("Invalid input tensor shape", x.shape)

        hidden = hidden.reshape(B, TT, self._in_tensor_shapes[0][0])
        time_steps = td.get("time_steps")

        if self.training:
            assert TT == self.context_size
            block_mask = _create_block_mask(self.context_size) if _flex_attention else None
            if time_steps is None:
                time_steps = torch.arange(self.context_size, dtype=torch.long, device=hidden.device)[None, :]
            self._inner_forward(hidden.clone(), time_steps, block_mask)
        else:
            self.ensure_kv_cache(B, hidden.device, hidden.dtype)
            self._inner_forward(hidden.clone(), time_steps, None)

        hidden = hidden.reshape(B * TT, self.hidden_size)

        td[self._name] = hidden
        return td

    @torch.compile(mode="max-autotune", dynamic=True, fullgraph=True, disable=not _compile)
    def _inner_forward(self, hidden: torch.Tensor, time_steps: torch.Tensor, block_mask: BlockMask | None) -> torch.Tensor:
        for layer in self.layers:
            hidden = layer(hidden, time_steps, block_mask)
        hidden = self.layer_norm(hidden)

        return hidden
