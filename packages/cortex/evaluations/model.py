from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from cortex.stacks import CortexStack
from cortex.types import MaybeState


class SequenceClassifier(nn.Module):
    """Wraps a `CortexStack` with an embedding and classifier head.

    Inputs are integer tokens `[B, T]` coming from synthetic datasets.
    """

    def __init__(self, stack: CortexStack, vocab_size: int, d_hidden: int, n_classes: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_hidden)
        self.stack = stack
        self.classifier = nn.Linear(d_hidden, n_classes)

    def forward(
        self, tokens: torch.Tensor, state: MaybeState | None = None
    ) -> Tuple[torch.Tensor, MaybeState]:  # logits, next_state
        # tokens: [B, T]
        x = self.embed(tokens)  # [B, T, H]

        # Match stack dtype (e.g., bfloat16 stacks) to avoid matmul dtype mismatches.
        target_stack_dtype = next(self.stack.parameters()).dtype
        if x.dtype != target_stack_dtype:
            x = x.to(dtype=target_stack_dtype)

        y, next_state = self.stack(x, state)

        # Cast to classifier dtype before the output head.
        target_head_dtype = self.classifier.weight.dtype
        if y.dtype != target_head_dtype:
            y = y.to(dtype=target_head_dtype)
        # Last timestep pooling by default
        logits = self.classifier(y[:, -1, :])
        return logits, next_state


__all__ = ["SequenceClassifier"]
