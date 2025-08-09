from dataclasses import dataclass


@dataclass(slots=True)
class TrainerState:
    """Lightweight, fast, mutable container for training loop state. Also provides a way for losses to influence
    trainer's control flow."""

    agent_step: int = 0
    epoch: int = 0
    update_epoch: int = 0
    mb_idx: int = 0
    num_minibatches: int = 0

    # Control flags and scratch metrics
    early_stop_update_epoch: bool = False

    def start_update_epoch(self, update_epoch: int, num_minibatches: int) -> None:
        self.update_epoch = update_epoch
        self.num_minibatches = num_minibatches
        self.mb_idx = 0
        self.early_stop_update_epoch = False

    def is_last_minibatch(self) -> bool:
        return self.num_minibatches > 0 and self.mb_idx == self.num_minibatches - 1
