import typing

import cortex.stacks
import cortex.types
import torch
import torch.nn as nn


class SequenceClassifier(nn.Module):
    """Wraps a `CortexStack` with an embedding and classifier head.

    Inputs are integer tokens `[B, T]` coming from synthetic datasets.
    """

    def __init__(self, stack: cortex.stacks.CortexStack, vocab_size: int, d_hidden: int, n_classes: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_hidden)
        self.stack = stack
        self.classifier = nn.Linear(d_hidden, n_classes)

    def forward(
        self, tokens: torch.Tensor, state: cortex.types.MaybeState | None = None
    ) -> typing.Tuple[torch.Tensor, cortex.types.MaybeState]:  # logits, next_state
        # tokens: [B, T]
        x = self.embed(tokens)  # [B, T, H]
        y, next_state = self.stack(x, state)
        # Last timestep pooling by default
        logits = self.classifier(y[:, -1, :])
        return logits, next_state


__all__ = ["SequenceClassifier"]
