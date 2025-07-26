from typing import Tuple

import torch
from heavyball import ForeachMuon

__all__ = ["Optimizer"]


class Optimizer:
    """Unified wrapper for Adam and Heavyball-Muon optimizers."""

    def __init__(
        self,
        optimizer_type: str,
        policy: torch.nn.Module,
        learning_rate: float,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        max_grad_norm: float = 1.0,
    ) -> None:
        opt_cls = torch.optim.Adam if optimizer_type == "adam" else ForeachMuon
        if optimizer_type != "adam":
            weight_decay = int(weight_decay)

        self.optimizer = opt_cls(
            policy.parameters(),
            lr=learning_rate,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        self.max_grad_norm = max_grad_norm
        self.param_groups = self.optimizer.param_groups

    def step(self, loss: torch.Tensor, epoch: int, accumulate_minibatches: int = 1) -> None:
        """Back-propagate *loss* and update parameters with optional gradient accumulation."""
        self.optimizer.zero_grad()
        loss.backward()

        if (epoch + 1) % accumulate_minibatches == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for group in self.optimizer.param_groups for p in group["params"]],
                self.max_grad_norm,
            )
            self.optimizer.step()

            if torch.cuda.is_available():
                torch.cuda.synchronize()

    # Thin wrappers to mimic torch.optim.Optimizer API
    def state_dict(self):  # noqa: D401  # "Returns" docstring skipped for brevity
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)
