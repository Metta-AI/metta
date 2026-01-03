"""Arena recipe with shaped rewards and a Cortex (~100M) policy."""

from pathlib import Path
from typing import Optional, Sequence

from cortex.stacks import build_cortex_auto_config

import metta.cogworks.curriculum as cc
import mettagrid.builder.envs as eb
from metta.agent.policies.cortex import CortexBaseConfig
from metta.agent.policy import PolicyArchitecture
from metta.cogworks.curriculum.curriculum import (
    CurriculumAlgorithmConfig,
    CurriculumConfig,
)
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
from metta.rl.nodes import default_nodes
from metta.rl.trainer_config import AdvantageConfig, OptimizerConfig, TorchProfilerConfig, TrainerConfig
from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig
from metta.rl.training.scheduler import RunGate, SchedulerConfig, ScheduleRule
from metta.rl.training.teacher import TeacherConfig, apply_teacher_phase
from metta.sim.simulation_config import SimulationConfig
from metta.sweep.core import Distribution as D
from metta.sweep.core import SweepParameters as SP
from metta.sweep.core import make_sweep
from metta.tools.eval import EvaluateTool
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.stub import StubTool
from metta.tools.sweep import SweepTool
from metta.tools.train import TrainTool
from mettagrid import MettaGridConfig


def _trainer_and_env_overrides() -> tuple[dict[str, object], dict[str, object]]:
    trainer_updates = {
        "compile": False,
        "batch_size": 4_194_304 // 2,
        "minibatch_size": 8192,
        "bptt_horizon": 256,
        "optimizer": OptimizerConfig(learning_rate=1e-4),
        "advantage": AdvantageConfig(gae_lambda=0.95, gamma=0.999),
    }

    env_updates = {
        "forward_pass_minibatch_target_size": 16384 // 2,
        "auto_workers": False,
        "num_workers": 1,
        "async_factor": 1,
    }

    return trainer_updates, env_updates


def mettagrid(num_agents: int = 24) -> MettaGridConfig:
    arena_env = eb.make_arena(num_agents=num_agents)

    arena_env.game.agent.rewards.inventory = {
        "heart": 1,
        "ore_red": 0.1,
        "battery_red": 0.8,
        "laser": 0.5,
        "armor": 0.5,
        "blueprint": 0.5,
    }
    arena_env.game.agent.rewards.inventory_max = {
        "heart": 100,
        "ore_red": 1,
        "battery_red": 1,
        "laser": 1,
        "armor": 1,
        "blueprint": 1,
    }

    return arena_env


def make_curriculum(
    arena_env: Optional[MettaGridConfig] = None,
    enable_detailed_slice_logging: bool = False,
    algorithm_config: Optional[CurriculumAlgorithmConfig] = None,
) -> CurriculumConfig:
    arena_env = arena_env or mettagrid()

    arena_tasks = cc.bucketed(arena_env)

    for item in ["ore_red", "battery_red", "laser", "armor"]:
        arena_tasks.add_bucket(f"game.agent.rewards.inventory.{item}", [0, 0.1, 0.5, 0.9, 1.0])
        arena_tasks.add_bucket(f"game.agent.rewards.inventory_max.{item}", [1, 2])

    # enable or disable attacks. we use cost instead of 'enabled'
    # to maintain action space consistency.
    arena_tasks.add_bucket("game.actions.attack.consumed_resources.laser", [1, 100])

    if algorithm_config is None:
        algorithm_config = LearningProgressConfig(
            use_bidirectional=True,  # Enable bidirectional learning progress by default
            ema_timescale=0.001,
            exploration_bonus=0.1,
            max_memory_tasks=1000,
            max_slice_axes=5,  # More slices for arena complexity
            enable_detailed_slice_logging=enable_detailed_slice_logging,
        )

    return arena_tasks.to_curriculum(algorithm_config=algorithm_config)


def simulations(env: Optional[MettaGridConfig] = None) -> list[SimulationConfig]:
    basic_env = env or mettagrid()
    basic_env.game.actions.attack.consumed_resources["laser"] = 100

    combat_env = basic_env.model_copy()
    combat_env.game.actions.attack.consumed_resources["laser"] = 1

    return [
        SimulationConfig(suite="arena", name="basic", env=basic_env),
        SimulationConfig(suite="arena", name="combat", env=combat_env),
    ]


def train(
    curriculum: Optional[CurriculumConfig] = None,
    enable_detailed_slice_logging: bool = False,
    dtype: str = "float32",
    policy_architecture: Optional[PolicyArchitecture] = None,
    teacher: Optional[TeacherConfig] = None,
) -> TrainTool:
    curriculum = curriculum or make_curriculum(enable_detailed_slice_logging=enable_detailed_slice_logging)

    eval_simulations = simulations()

    trainer_updates, env_updates = _trainer_and_env_overrides()

    if policy_architecture is None:
        stack_cfg = build_cortex_auto_config(
            d_hidden=256,
            num_layers=28,
            pattern=[
                "Ag,A",
                "Ag,A",
                "Ag,A",
                "Ag,A,S",
            ]
            * 7,
            post_norm=True,
            compile_blocks=True,
        )
        policy_architecture = CortexBaseConfig(stack_cfg=stack_cfg, dtype=dtype)

    nodes = default_nodes()
    default_teacher_steps = 600_000_000
    teacher = teacher or TeacherConfig(
        policy_uri="s3://softmax-public/policies/subho.abes.vit_baseline/subho.abes.vit_baseline:v2340.mpt",
        mode="sliced_kickstarter",
        steps=default_teacher_steps,
        teacher_led_proportion=0.2,
    )
    nodes["sliced_kickstarter"].action_loss_coef = 1.0
    nodes["sliced_kickstarter"].value_loss_coef = 0.0

    trainer_cfg = TrainerConfig(nodes=nodes)
    trainer_cfg = trainer_cfg.model_copy(update=trainer_updates)

    training_env = TrainingEnvironmentConfig(curriculum=curriculum)
    training_env = training_env.model_copy(update=env_updates)

    tt = TrainTool(
        trainer=trainer_cfg,
        training_env=training_env,
        evaluator=EvaluatorConfig(simulations=eval_simulations),
        policy_architecture=policy_architecture,
        torch_profiler=TorchProfilerConfig(),
    )
    scheduler_run_gates: list[RunGate] = []
    scheduler_rules: list[ScheduleRule] = []
    apply_teacher_phase(
        trainer_cfg=tt.trainer,
        training_env_cfg=tt.training_env,
        scheduler_rules=scheduler_rules,
        scheduler_run_gates=scheduler_run_gates,
        teacher_cfg=teacher,
        default_steps=default_teacher_steps,
    )
    tt.scheduler = SchedulerConfig(run_gates=scheduler_run_gates, rules=scheduler_rules)
    return tt


def evaluate(policy_uris: Optional[Sequence[str]] = None) -> EvaluateTool:
    """Evaluate policies on arena simulations."""
    return EvaluateTool(simulations=simulations(), policy_uris=policy_uris or [])


def evaluate_latest_in_dir(dir_path: Path) -> EvaluateTool:
    """Evaluate the latest policy on arena simulations."""
    checkpoints = dir_path.glob("*.mpt")
    policy_uri = [checkpoint.as_posix() for checkpoint in sorted(checkpoints, key=lambda x: x.stat().st_mtime)]
    if not policy_uri:
        raise ValueError(f"No policies found in {dir_path}")
    policy_uri = policy_uri[-1]
    sim = mettagrid(num_agents=6)
    return EvaluateTool(
        simulations=[SimulationConfig(suite="arena", name="very_basic", env=sim)], policy_uris=[policy_uri]
    )


def play(policy_uri: Optional[str] = None) -> PlayTool:
    """Interactive play with a policy."""
    return PlayTool(sim=simulations()[0], policy_uri=policy_uri)


def replay(policy_uri: Optional[str] = None) -> ReplayTool:
    """Generate replay from a policy."""
    return ReplayTool(sim=simulations()[0], policy_uri=policy_uri)


def evaluate_in_sweep(policy_uri: str) -> EvaluateTool:
    """Evaluation tool for sweep runs (single policy_uri, 4-minute sims)."""

    # Create sweep-optimized versions of the standard evaluations
    # Use a dedicated suite name to control the metric namespace in WandB
    basic_env = mettagrid()
    basic_env.game.actions.attack.consumed_resources["laser"] = 100

    combat_env = basic_env.model_copy()
    combat_env.game.actions.attack.consumed_resources["laser"] = 1

    simulations = [
        SimulationConfig(
            suite="sweep",
            name="basic",
            env=basic_env,
            num_episodes=1,  # Using 1 episode for evaluation
            max_time_s=240,  # 4 minutes max per simulation
        ),
        SimulationConfig(
            suite="sweep",
            name="combat",
            env=combat_env,
            num_episodes=1,
            max_time_s=240,
        ),
    ]

    return EvaluateTool(
        simulations=simulations,
        policy_uris=[policy_uri],
    )


def evaluate_stub(*args, **kwargs) -> StubTool:
    return StubTool()


def sweep(sweep_name: str) -> SweepTool:
    """Prototypical sweep entrypoint."""

    # Common parameters are accessible via SP (SweepParameters).
    parameters = [
        SP.LEARNING_RATE,
        SP.PPO_CLIP_COEF,
        SP.PPO_GAE_LAMBDA,
        SP.PPO_VF_COEF,
        SP.ADAM_EPS,
        SP.param(
            "trainer.total_timesteps",
            D.INT_UNIFORM,
            min=5e8,
            max=2e9,
            search_center=7.5e8,
        ),
    ]

    return make_sweep(
        name=sweep_name,
        recipe="recipes.prod.arena_basic_easy_shaped",
        train_entrypoint="train",
        # NB: You MUST use a specific sweep eval suite, different than those in training.
        # Besides this being a recommended practice, using the same eval suite in both
        # training and scoring will lead to key conflicts that will lock the sweep.
        eval_entrypoint="evaluate_stub",
        # Typically, "evaluator/eval_{suite}/score"
        objective="experience/rewards",
        parameters=parameters,
        max_trials=80,
        # Default value is 1. We don't recommend going higher than 4.
        # The faster each individual trial, the lower you should set this number.
        num_parallel_trials=4,
    )
