from __future__ import annotations

from pathlib import Path

from metta.rl.trainer_config import TrainerConfig
from metta.rl.training import CheckpointerConfig, EvaluatorConfig, TrainingEnvironmentConfig
from metta.tools.train import TrainTool
from recipes.prod.arena_basic_easy_shaped import evaluate_latest_in_dir, make_curriculum, simulations


def run_arena_train_and_eval(run_name: str) -> None:
    """Run a tiny arena training to produce a checkpoint, then evaluate it.

    This keeps train+eval in the same job so the checkpoint is available locally.
    """
    # Train briefly with frequent checkpoints
    train_tool = TrainTool(
        run=run_name,
        trainer=TrainerConfig(total_timesteps=100),
        checkpointer=CheckpointerConfig(epoch_interval=1),
        training_env=TrainingEnvironmentConfig(curriculum=make_curriculum()),
        evaluator=EvaluatorConfig(simulations=simulations()),
    )
    train_tool.run()

    # Evaluate latest checkpoint from the run's checkpoints dir
    ckpt_dir = Path("./train_dir") / run_name / "checkpoints"
    evaluate_latest_in_dir(ckpt_dir)
