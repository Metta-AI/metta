"""Machina v1 open-world recipe using the full vibe set and sweep helpers."""

from typing import Optional, Sequence

from metta.agent.policies.vit import ViTDefaultConfig
from metta.agent.policy import PolicyArchitecture
from metta.rl.training.teacher import TeacherConfig
from metta.sim.simulation_config import SimulationConfig
from metta.sweep.core import Distribution as D
from metta.sweep.core import SweepParameters as SP
from metta.sweep.core import make_sweep
from metta.tools.stub import StubTool
from metta.tools.sweep import SweepTool
from metta.tools.train import TrainTool
from mettagrid.config import vibes
from recipes.experiment.cogs_v_clips import (
    apply_cvc_sweep_defaults,
    make_training_env,
    train_single_mission,
)


def train(
    num_cogs: int = 4,
    variants: Optional[Sequence[str]] = None,
    eval_variants: Optional[Sequence[str]] = None,
    eval_difficulty: str | None = "standard",
    policy_architecture: PolicyArchitecture | None = None,
    teacher: TeacherConfig | None = None,
    map_seed: int = 50,
    ppo_learning_rate: float = 3e-4,
) -> TrainTool:
    """Train on machina_1.open_world with sweep-tuned defaults and single-map eval."""

    tt = train_single_mission(
        mission="machina_1.open_world",
        num_cogs=num_cogs,
        variants=variants,
        eval_variants=eval_variants,
        eval_difficulty=eval_difficulty,
        teacher=teacher,
    )
    tt.policy_architecture = policy_architecture or ViTDefaultConfig()

    tt.training_env.maps_cache_size = 1
    tt.training_env.seed = map_seed

    # Keep the environment action space full; restrict actor sampling to the first 21 actions.
    if tt.policy_architecture.action_probs_config:
        tt.policy_architecture.action_probs_config.max_action_index = 21
    # Explicitly keep full vibe/action definitions so saved checkpoints remain compatible.
    full_vibes = [v.name for v in vibes.VIBES]
    env_cfg = tt.training_env.curriculum.task_generator.env
    env_cfg.game.vibe_names = full_vibes
    change_vibe = getattr(env_cfg.game.actions, "change_vibe", None)
    if change_vibe is not None:
        change_vibe.number_of_vibes = len(full_vibes)
    if env_cfg.game.agent.initial_vibe >= len(full_vibes):
        env_cfg.game.agent.initial_vibe = 0

    apply_cvc_sweep_defaults(tt.trainer)
    tt.trainer.optimizer.learning_rate = ppo_learning_rate

    eval_env = make_training_env(num_cogs=num_cogs, mission="machina_1.open_world", variants=eval_variants)
    eval_env.game.vibe_names = list(full_vibes)
    change_vibe_eval = getattr(eval_env.game.actions, "change_vibe", None)
    if change_vibe_eval is not None:
        change_vibe_eval.number_of_vibes = len(full_vibes)
    if eval_env.game.agent.initial_vibe >= len(full_vibes):
        eval_env.game.agent.initial_vibe = 0
    tt.evaluator.simulations = [
        SimulationConfig(
            suite="cogs_vs_clips",
            name=f"machina_1_open_world_{num_cogs}cogs",
            env=eval_env,
        )
    ]
    # Slow down evals for long runs
    tt.evaluator.epoch_interval = 3000
    return tt


def train_sweep(
    num_cogs: int = 4,
    variants: Optional[Sequence[str]] = None,
    eval_variants: Optional[Sequence[str]] = None,
    eval_difficulty: str | None = "standard",
    policy_architecture: PolicyArchitecture | None = None,
    teacher: TeacherConfig | None = None,
    ppo_learning_rate: float = 3e-4,
) -> TrainTool:
    """Sweep-friendly train with heart_chorus baked in."""

    base_variants = ["heart_chorus"]
    if variants:
        for v in variants:
            if v not in base_variants:
                base_variants.append(v)

    return train(
        num_cogs=num_cogs,
        variants=base_variants,
        eval_variants=eval_variants or base_variants,
        eval_difficulty=eval_difficulty,
        policy_architecture=policy_architecture,
        teacher=teacher,
        ppo_learning_rate=ppo_learning_rate,
    )


def evaluate_stub(*args, **kwargs) -> StubTool:
    """No-op evaluator for sweeps."""

    return StubTool()


def sweep(
    sweep_name: str,
    num_cogs: int = 4,
    eval_difficulty: str | None = "standard",
    max_trials: int = 80,
    num_parallel_trials: int = 4,
) -> SweepTool:
    """Hyperparameter sweep targeting train_sweep (heart_chorus baked in)."""

    search_space = {
        **SP.LEARNING_RATE,
        **SP.PPO_CLIP_COEF,
        **SP.PPO_GAE_LAMBDA,
        **SP.PPO_VF_COEF,
        **SP.ADAM_EPS,
        **SP.param(
            "trainer.total_timesteps",
            D.INT_UNIFORM,
            min=5e8,
            max=2e9,
            search_center=1e9,
        ),
    }

    return make_sweep(
        name=sweep_name,
        recipe="recipes.experiment.machina_1",
        train_entrypoint="train_sweep",
        eval_entrypoint="evaluate_stub",
        metric_key="env_agent/heart.gained",
        search_space=search_space,
        cost_key="metric/total_time",
        max_trials=max_trials,
        num_parallel_trials=num_parallel_trials,
    )


__all__ = [
    "train",
    "train_sweep",
    "evaluate_stub",
    "sweep",
]
