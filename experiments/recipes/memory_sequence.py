import os
from datetime import datetime
from typing import Optional

import metta.cogworks.curriculum as cc
import metta.mettagrid.config.envs as eb
from metta.cogworks.curriculum.curriculum import CurriculumConfig
from metta.mettagrid.mettagrid_config import EnvConfig
from metta.rl.trainer_config import EvaluationConfig, TrainerConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.sim import SimTool
from metta.tools.train import TrainTool

from experiments.evals import memory_sequence as mem_evals


def _get_user_identifier() -> str:
    """Get user identifier from USER environment variable."""
    return os.getenv("USER", "unknown")


def _default_run_name() -> str:
    """Generate a robust run name following the pattern: memory_sequence.{user}.{date}.{unique_id}

    Format: memory_sequence.{username}.MMDD-HHMMSS.{git_hash_short} or memory_sequence.{username}.MMDD-HHMMSS
    Example: memory_sequence.alice.0820-143052.a1b2c3d or memory_sequence.alice.0820-143052
    """
    user = _get_user_identifier()
    now = datetime.now()
    timestamp = now.strftime("%m%d-%H%M%S")

    try:
        from metta.common.util.git import get_current_commit

        git_hash = get_current_commit()[:7]
        return f"memory_sequence.{user}.{timestamp}.{git_hash}"
    except Exception:
        return f"memory_sequence.{user}.{timestamp}"


def make_env(num_agents: int = 1) -> EnvConfig:
    return eb.make_memory_sequence(num_agents=num_agents)


def make_memory_eval_suite() -> list[SimulationConfig]:
    return mem_evals.make_memory_sequence_eval_suite()


def make_curriculum(env: Optional[EnvConfig] = None) -> CurriculumConfig:
    env = env or make_env()

    # Minimal curriculum structure for memory sequence tasks
    tasks = cc.bucketed(env)

    return tasks.to_curriculum()


def train(
    run: Optional[str] = None, curriculum: Optional[CurriculumConfig] = None
) -> TrainTool:
    if run is None:
        run = _default_run_name()
    trainer_cfg = TrainerConfig(
        curriculum=curriculum or make_curriculum(),
        evaluation=EvaluationConfig(
            simulations=mem_evals.make_memory_sequence_eval_suite(),
        ),
    )

    return TrainTool(
        trainer=trainer_cfg,
        run=run,
    )


def play(env: Optional[EnvConfig] = None) -> PlayTool:
    eval_env = env or make_env()
    return PlayTool(
        sim=SimulationConfig(
            env=eval_env,
            name="memory_sequence",
        ),
    )


def replay(env: Optional[EnvConfig] = None) -> ReplayTool:
    eval_env = env or make_env()
    return ReplayTool(
        sim=SimulationConfig(
            env=eval_env,
            name="memory_sequence",
        ),
    )


def evaluate(policy_uri: str) -> SimTool:
    return SimTool(simulations=make_memory_eval_suite(), policy_uris=[policy_uri])
