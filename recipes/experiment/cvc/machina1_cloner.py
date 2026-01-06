"""Machina v1 open-world recipe using the full vibe set and sweep helpers with a Cortex policy."""

from typing import Optional, Sequence

from cortex.stacks import build_cortex_auto_config

from metta.agent.policies.cortex import CortexBaseConfig
from metta.agent.policy import PolicyArchitecture
from metta.rl.trainer_config import AdvantageConfig
from metta.rl.training.scheduler import LossRunGate, ScheduleRule
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
        "update_epochs": 2,
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
    variants: Optional[Sequence[str]] = ("heart_chorus",),
    eval_variants: Optional[Sequence[str]] = None,
    eval_difficulty: str | None = None,
    dtype: str = "float32",
    policy_architecture: PolicyArchitecture | None = None,
    teacher: TeacherConfig | None = None,
    evaluate_local: bool = False,
    teacher_led_proportion: float = 0.0,
    student_led_proportion: float = 1.0,
    bc_steps: int = 2_500_000_000,
    anneal_steps: int = 3_000_000_000,
    cortex_d_hidden: int = 128,
    cortex_num_layers: int = 2,
    cortex_pattern: Sequence[str] = ("Ag,A,S",),
    ppo_during_bc: bool = True,
) -> TrainTool:
    """Train on machina_1.open_world with leaderboard-aligned defaults and single-map eval."""
    if eval_variants is None:
        eval_variants = variants

    if bc_steps <= 0:
        raise ValueError("bc_steps must be > 0")
    if anneal_steps <= 0:
        raise ValueError("anneal_steps must be > 0")

    teacher_total_steps = bc_steps + anneal_steps
    if teacher is None:
        teacher = TeacherConfig(
            policy_uri="metta://policy/dinky:v15",
            mode="sliced_cloner",
            steps=teacher_total_steps,
            teacher_led_proportion=teacher_led_proportion,
            student_led_proportion=student_led_proportion,
            # When PPO is enabled, let it train on all slices (teacher/student/ppo).
            kwargs={"restrict_ppo_to_ppo_mask": False},
        )
    else:
        if teacher.steps is None:
            teacher.steps = teacher_total_steps
        if "teacher_led_proportion" not in teacher.model_fields_set:
            teacher.teacher_led_proportion = teacher_led_proportion
        if "student_led_proportion" not in teacher.model_fields_set:
            teacher.student_led_proportion = student_led_proportion
        teacher = TeacherConfig.model_validate(teacher.model_dump())

    if teacher.steps != teacher_total_steps:
        raise ValueError(f"Expected teacher.steps={teacher_total_steps} (bc_steps+anneal_steps), got {teacher.steps}")

    tt = train_single_mission(
        mission="machina_1.open_world",
        num_cogs=num_cogs,
        variants=variants,
        eval_variants=eval_variants,
        eval_difficulty=eval_difficulty,
        teacher=teacher,
        maps_cache_size=None,
    )
    tt.evaluator.evaluate_local = evaluate_local

    if tt.scheduler is None:
        raise RuntimeError("Expected TrainTool.scheduler to be set by train_single_mission")

    # --------------------- BC + PPO schedule (sliced_cloner) ---------------------
    #
    # Goal:
    # 1) Keep PPO active for the entire run.
    # 2) [0 .. bc_steps)        : BC slices run alongside PPO.
    # 3) [bc_steps .. end)      : anneal teacher/student slice proportions -> 0 linearly.
    # 4) [end .. inf)           : pure PPO (cloner gated off by teacher.steps).
    #
    # Mechanism:
    # - We use TeacherConfig(mode="sliced_cloner") to enable the sliced cloner loss and supervisor actions.
    # - We override teacher.py's default anneal (0..end) so annealing begins at bc_steps instead.
    teacher_proportion_path = "losses.sliced_scripted_cloner.teacher_led_proportion"
    student_proportion_path = "losses.sliced_scripted_cloner.student_led_proportion"

    # teacher.py delays PPO critic rollout until teacher.steps; keep PPO running from step 0 instead.
    # (Leaving this gate in place would still work due to OR semantics, but it's misleading in configs.)
    ppo_gate_names = {"ppo_critic", "quantile_ppo_critic"}
    tt.scheduler.run_gates = [
        gate
        for gate in tt.scheduler.run_gates
        if not (
            gate.phase == "rollout"
            and gate.begin_at_step is not None
            and gate.loss_instance_name in ppo_gate_names
        )
    ]
    if not ppo_during_bc:
        # Optional: keep PPO gated off until bc_steps to run a pure-BC warmup.
        for loss_name in ("ppo_actor", "ppo_critic"):
            for phase in ("rollout", "train"):
                tt.scheduler.run_gates.append(
                    LossRunGate(loss_instance_name=loss_name, phase=phase, begin_at_step=bc_steps)
                )

    # Remove teacher.py's default anneal-from-zero rules; we re-add anneals starting at bc_steps.
    blocked_paths = {teacher_proportion_path, student_proportion_path}
    tt.scheduler.rules = [rule for rule in tt.scheduler.rules if rule.target_path not in blocked_paths]

    # Phase 2: linearly anneal BC slice proportions down to 0, growing the PPO slice over time.
    tt.scheduler.rules.append(
        ScheduleRule(
            target_path=teacher_proportion_path,
            mode="progress",
            style="linear",
            start_value=float(teacher.teacher_led_proportion),
            end_value=0.0,
            start_agent_step=bc_steps,
            end_agent_step=teacher_total_steps,
        )
    )
    tt.scheduler.rules.append(
        ScheduleRule(
            target_path=student_proportion_path,
            mode="progress",
            style="linear",
            start_value=float(teacher.student_led_proportion),
            end_value=0.0,
            start_agent_step=bc_steps,
            end_agent_step=teacher_total_steps,
        )
    )

    trainer_updates, env_updates = _trainer_and_env_overrides()
    tt.trainer = tt.trainer.model_copy(update=trainer_updates)
    tt.training_env = tt.training_env.model_copy(update=env_updates)

    if policy_architecture is None:
        cortex_pattern = list(cortex_pattern)
        if len(cortex_pattern) != cortex_num_layers:
            if cortex_num_layers % len(cortex_pattern) != 0:
                raise ValueError(
                    f"cortex_num_layers ({cortex_num_layers}) must be a multiple of len(cortex_pattern) "
                    f"({len(cortex_pattern)})"
                )
            cortex_pattern = cortex_pattern * (cortex_num_layers // len(cortex_pattern))

        stack_cfg = build_cortex_auto_config(
            d_hidden=cortex_d_hidden,
            num_layers=cortex_num_layers,
            pattern=cortex_pattern,
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
    evaluate_local: bool = False,
    ppo_during_bc: bool = True,
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
        evaluate_local=evaluate_local,
        ppo_during_bc=ppo_during_bc,
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
