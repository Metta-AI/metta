"""Post-processing block that applies cell then FFN with residual connections."""


import typing

import cortex.blocks.base
import cortex.blocks.registry
import cortex.cells.base
import cortex.config
import cortex.types
import tensordict
import torch.nn as nn


@cortex.blocks.registry.register_block(cortex.config.PostUpBlockConfig)
class PostUpBlock(cortex.blocks.base.BaseBlock):
    """Block that applies cell then FFN sublayer with residual connections."""

    def __init__(
        self, config: cortex.config.PostUpBlockConfig, d_hidden: int, cell: cortex.cells.base.MemoryCell
    ) -> None:
        super().__init__(d_hidden=d_hidden, cell=cell)
        self.config = config
        self.d_inner = int(config.proj_factor * d_hidden)
        assert cell.hidden_size == d_hidden, "PostUpBlock requires cell.hidden_size == d_hidden"
        self.norm = nn.LayerNorm(d_hidden, elementwise_affine=True, bias=False)
        self.ffn_norm = nn.LayerNorm(d_hidden, elementwise_affine=True, bias=False)
        self.out1 = nn.Linear(d_hidden, self.d_inner)
        self.act = nn.SiLU()
        self.out2 = nn.Linear(self.d_inner, d_hidden)

    def forward(
        self,
        x: cortex.types.Tensor,
        state: cortex.types.MaybeState,
        *,
        resets: typing.Optional[cortex.types.ResetMask] = None,
    ) -> typing.Tuple[cortex.types.Tensor, cortex.types.MaybeState]:
        cell_key = self.cell.__class__.__name__
        cell_state = state.get(cell_key, None) if state is not None else None
        batch_size = state.batch_size[0] if state is not None and state.batch_size else x.shape[0]

        residual = x
        x_normed = self.norm(x)
        y_cell, new_cell_state = self.cell(x_normed, cell_state, resets=resets)
        y = residual + y_cell

        ffn_residual = y
        y_ffn_normed = self.ffn_norm(y)
        is_step = y_ffn_normed.dim() == 2
        if is_step:
            y_ffn = self.out1(y_ffn_normed)
            y_ffn = self.act(y_ffn)
            y_ffn = self.out2(y_ffn)
        else:
            B, T, H = y_ffn_normed.shape
            y_ = self.out1(y_ffn_normed.reshape(B * T, H))
            y_ = self.act(y_)
            y_ffn = self.out2(y_).reshape(B, T, self.d_hidden)

        y = ffn_residual + y_ffn
        return y, tensordict.TensorDict({cell_key: new_cell_state}, batch_size=[batch_size])


__all__ = ["PostUpBlock"]
