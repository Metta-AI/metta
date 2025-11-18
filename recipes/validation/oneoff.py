from __future__ import annotations

from pathlib import Path

from recipes.prod.arena_basic_easy_shaped import evaluate_latest_in_dir
from recipes.prod.arena_basic_easy_shaped import train as arena_train


def run_arena_train_and_eval(run_name: str) -> None:
    """Run a tiny arena training to produce a checkpoint, then evaluate it.

    This keeps train+eval in the same job so the checkpoint is available locally.
    """
    # Train briefly with frequent checkpoints
    arena_train(
        run=run_name,
        trainer={"total_timesteps": 100, "epoch_length": 10},
        checkpointer={"epoch_interval": 1},
    )

    # Evaluate latest checkpoint from the run's checkpoints dir
    ckpt_dir = Path("./train_dir") / run_name / "checkpoints"
    evaluate_latest_in_dir(ckpt_dir)
