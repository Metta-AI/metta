"""Machina v1 open-world recipe using the full vibe set and sweep helpers with a Cortex (~100M) policy."""

from typing import Optional, Sequence

from cortex.stacks import build_cortex_auto_config

from metta.agent.policies.cortex import CortexBaseConfig
from metta.agent.policy import PolicyArchitecture
from metta.rl.trainer_config import AdvantageConfig, OptimizerConfig
from metta.rl.training.teacher import TeacherConfig
from metta.sim.simulation_config import SimulationConfig
from metta.sweep.core import make_sweep
from metta.tools.stub import StubTool
from metta.tools.sweep import SweepTool
from metta.tools.train import TrainTool
from mettagrid.config import vibes
from recipes.experiment.cogs_v_clips import (
    _normalize_variant_names,
    get_cvc_sweep_search_space,
    make_training_env,
    train_single_mission,
)


def _trainer_and_env_overrides() -> tuple[dict[str, object], dict[str, object]]:
    trainer_updates = {
        # "compile": False,
        # "batch_size": 4_194_304 ,
        # "minibatch_size": 8192,
        # "bptt_horizon": 256,
        # "optimizer": OptimizerConfig(learning_rate=1e-4),
        "advantage": AdvantageConfig(gae_lambda=0.997),
    }

    env_updates = {
        # "forward_pass_minibatch_target_size": 16384,
        # "auto_workers": False,
        # "num_workers": 1,
        # "async_factor": 1,
    }

    return trainer_updates, env_updates


def train(
    num_cogs: int = 4,
    variants: Optional[Sequence[str]] = None,
    eval_variants: Optional[Sequence[str]] = None,
    eval_difficulty: str | None = None,
    dtype: str = "float32",
    policy_architecture: PolicyArchitecture | None = None,
    teacher: TeacherConfig | None = None,
) -> TrainTool:
    """Train on machina_1.open_world with leaderboard-aligned defaults and single-map eval."""
    if eval_variants is None:
        eval_variants = variants

    teacher = teacher or TeacherConfig(
        policy_uri="metta://policy/dinky:v15",
        mode="sliced_cloner",
        steps=2_500_000_000,
        teacher_led_proportion=0.3,
        kwargs={"restrict_ppo_to_ppo_mask": False},
    )

    tt = train_single_mission(
        mission="machina_1.open_world",
        num_cogs=num_cogs,
        variants=variants,
        eval_variants=eval_variants,
        eval_difficulty=eval_difficulty,
        teacher=teacher,
        maps_cache_size=None,
    )

    trainer_updates, env_updates = _trainer_and_env_overrides()
    tt.trainer = tt.trainer.model_copy(update=trainer_updates)
    tt.training_env = tt.training_env.model_copy(update=env_updates)

    if policy_architecture is None:
        stack_cfg = build_cortex_auto_config(
            d_hidden=256,
            num_layers=8,
            pattern=[
                "Ag,A",
                "Ag,A",
                "Ag,A",
                "Ag,A,S",
            ]
            * 2,
            post_norm=True,
            compile_blocks=True,
        )
        policy_architecture = CortexBaseConfig(stack_cfg=stack_cfg, dtype=dtype)
    tt.policy_architecture = policy_architecture

    losses_cfg = tt.trainer.losses
    needs_full_log_probs = any(
        getattr(losses_cfg, name).enabled
        for name in (
            "supervisor",
            "sliced_scripted_cloner",
            "eer_kickstarter",
            "eer_cloner",
        )
    )
    action_probs_config = getattr(tt.policy_architecture, "action_probs_config", None)
    if action_probs_config is not None and not needs_full_log_probs:
        action_probs_config.emit_full_log_probs = False
    tt.system.torch_deterministic = False

    # Explicitly keep full vibe/action definitions so saved checkpoints remain compatible.
    env_cfg = tt.training_env.curriculum.task_generator.env
    env_cfg.game.vibe_names = [v.name for v in vibes.VIBES]
    change_vibe = getattr(env_cfg.game.actions, "change_vibe", None)
    if change_vibe is not None:
        change_vibe.vibes = list(vibes.VIBES)
    if env_cfg.game.agent.initial_vibe >= len(vibes.VIBES):
        env_cfg.game.agent.initial_vibe = 0

    eval_variant_names = _normalize_variant_names(
        initial=[eval_difficulty] if eval_difficulty else None,
        variants=eval_variants,
    )
    eval_env = make_training_env(
        num_cogs=num_cogs,
        mission="machina_1.open_world",
        variants=eval_variant_names or None,
    )
    tt.evaluator.simulations = [
        SimulationConfig(
            suite="cogs_vs_clips",
            name=f"machina_1_open_world_{num_cogs}cogs",
            env=eval_env,
        )
    ]
    # Run evals periodically during long runs
    tt.evaluator.epoch_interval = 0
    return tt


def train_sweep(
    num_cogs: int = 4,
    variants: Optional[Sequence[str]] = None,
    eval_variants: Optional[Sequence[str]] = None,
    eval_difficulty: str | None = None,
    dtype: str = "float32",
    policy_architecture: PolicyArchitecture | None = None,
    teacher: TeacherConfig | None = None,
) -> TrainTool:
    """Sweep-friendly train with heart_chorus baked in."""

    base_variants = ["heart_chorus"]
    if variants:
        for v in variants:
            if v not in base_variants:
                base_variants.append(v)

    tt = train(
        num_cogs=num_cogs,
        variants=base_variants,
        eval_variants=eval_variants or base_variants,
        eval_difficulty=eval_difficulty,
        dtype=dtype,
        policy_architecture=policy_architecture,
        teacher=teacher,
    )
    # Sweep-friendly default (kept consistent with the shared CvC sweep search space).
    tt.trainer.total_timesteps = 1_000_000_000
    return tt


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

    search_space = get_cvc_sweep_search_space()

    return make_sweep(
        name=sweep_name,
        recipe="recipes.experiment.cvc.machina1_cortex_100m",
        train_entrypoint="train_sweep",
        eval_entrypoint="evaluate_stub",
        metric_key="env_game/assembler.heart.created",
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
