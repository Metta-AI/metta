"""Arena recipe with shaped rewards - STABLE
This recipe is automatically validated in CI and release processes.
"""

from typing import Optional

import metta.cogworks.curriculum as cc
import mettagrid.builder.envs as eb
from devops.stable.registry import ci_job, stable_job
from devops.stable.runner import AcceptanceCriterion
from metta.agent.policies.vit import ViTDefaultConfig
from metta.agent.policy import PolicyArchitecture
from metta.cogworks.curriculum.curriculum import (
    CurriculumAlgorithmConfig,
    CurriculumConfig,
)
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
from metta.common.wandb.context import WandbConfig
from metta.rl.trainer_config import TorchProfilerConfig, TrainerConfig
from metta.rl.training import CheckpointerConfig, EvaluatorConfig, TrainingEnvironmentConfig
from metta.rl.training.scheduler import LossRunGate, SchedulerConfig, ScheduleRule
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
    policy_architecture: Optional[PolicyArchitecture] = None,
    teacher: TeacherConfig | None = None,
) -> TrainTool:
    curriculum = curriculum or make_curriculum(enable_detailed_slice_logging=enable_detailed_slice_logging)

    eval_simulations = simulations()
    trainer_cfg = TrainerConfig()
    training_env_cfg = TrainingEnvironmentConfig(curriculum=curriculum)
    teacher = teacher or TeacherConfig()  # Disabled by default unless policy_uri is provided.

    if policy_architecture is None:
        policy_architecture = ViTDefaultConfig()

    # Enable optional teacher phases (e.g., sliced_cloner) when provided.
    scheduler_run_gates: list[LossRunGate] = []
    scheduler_rules: list[ScheduleRule] = []
    apply_teacher_phase(
        trainer_cfg=trainer_cfg,
        training_env_cfg=training_env_cfg,
        scheduler_rules=scheduler_rules,
        scheduler_run_gates=scheduler_run_gates,
        teacher_cfg=teacher,
        default_steps=teacher.steps or 1_000_000_000,
    )

    tt = TrainTool(
        trainer=trainer_cfg,
        training_env=training_env_cfg,
        evaluator=EvaluatorConfig(simulations=eval_simulations, epoch_interval=300),
        policy_architecture=policy_architecture,
        torch_profiler=TorchProfilerConfig(),
    )
    if scheduler_run_gates or scheduler_rules:
        tt.scheduler = SchedulerConfig(run_gates=scheduler_run_gates, rules=scheduler_rules)
    return tt


def evaluate(policy_uris: list[str] | str) -> EvaluateTool:
    """Evaluate policies on arena simulations."""
    return EvaluateTool(simulations=simulations(), policy_uris=policy_uris)


def play(policy_uri: Optional[str] = None) -> PlayTool:
    """Interactive play with a policy."""
    return PlayTool(sim=simulations()[0], policy_uri=policy_uri)


def replay(policy_uri: Optional[str] = None) -> ReplayTool:
    """Generate replay from a policy."""
    return ReplayTool(sim=simulations()[0], policy_uri=policy_uri)


def evaluate_in_sweep(policy_uri: str) -> EvaluateTool:
    """Evaluation tool for sweep runs.

    Uses 10 episodes per simulation with a 4-minute time limit to get
    reliable results quickly during hyperparameter sweeps.
    NB: Please note that this function takes a **single** policy_uri. This is the expected signature in our sweeps.
    Additional arguments are supported through eval_overrides.
    """

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
    """
    Prototypical sweep function.
    In your own recipe, you likely only every need this. You can override other SweepTool parameters in the CLI.

    Example usage:
        `uv run ./tools/run.py recipes.prod.arena_basic_easy_shaped.sweep \
            sweep_name="ak.baes.10081528" -- gpus=4 nodes=2 dispatcher_type=skypilot`

    We recommend running using local_test=True before running the sweep on the remote:
        `uv run ./tools/run.py recipes.prod.arena_basic_easy_shaped.sweep \
            sweep_name="ak.baes.10081528.local_test" -- local_test=True`

    This will run a quick local sweep and allow you to catch configuration bugs
    (NB: Unless those bugs are related to batch_size, minibatch_size, or hardware config).

    If this runs smoothly, you must launch the sweep on a remote sandbox
    (otherwise sweep progress will halt when you close your computer).

    Running on the remote:
        1 - Start a sweep controller sandbox: `./devops/skypilot/sandbox.py new --sweep-controller`, and ssh into it.
        2 - Clean git pollution: `git clean -df && git stash`
        3 - Ensure your sky credentials are present: `sky status` -- if not, follow the instructions on screen.
        4 - Install tmux on the sandbox `apt install tmux`
        5 - Launch tmux session: `tmux new -s sweep`
        6 - Launch the sweep:
            `uv run ./tools/run.py recipes.prod.arena_basic_easy_shaped.sweep \
                sweep_name="ak.baes.10081528" -- gpus=4 nodes=2 dispatcher_type=skypilot`
        7 - Detach when you want: CTRL+B then d
        8 - Attach to look at status/output: `tmux attach -t sweep`

    Please tag Axel (akerbec@softmax.com) on any bug report.
    """

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
        # Evaluations in sweeps are currently being fixed.
        eval_entrypoint="evaluate_stub",
        # Typically, "evaluator/eval_{suite}/score"
        metric_key="env_game/assembler.hearts.created",
        search_space=parameters,
        max_trials=80,
        # Default value is 1. We don't recommend going higher than 4.
        # The faster each individual trial, the lower you can set this number.
        num_parallel_trials=4,
    )


def train_ci() -> TrainTool:
    """Minimal train for CI smoke test."""
    return TrainTool(
        trainer=TrainerConfig(total_timesteps=100),
        training_env=TrainingEnvironmentConfig(
            curriculum=make_curriculum(),
            forward_pass_minibatch_target_size=96,
            vectorization="serial",
        ),
        evaluator=EvaluatorConfig(evaluate_local=False, evaluate_remote=False),
        checkpointer=CheckpointerConfig(epoch_interval=1),
        policy_architecture=ViTDefaultConfig(),
        wandb=WandbConfig.Off(),
    )


@ci_job(timeout_s=120)
def play_ci() -> PlayTool:
    """Play test with random policy."""
    return PlayTool(
        sim=simulations()[0],
        max_steps=10,
        render="log",
        open_browser_on_start=False,
    )


def evaluate_ci(policy_uri: str) -> EvaluateTool:
    """Evaluate the trained policy from train_ci."""
    sim = mettagrid(num_agents=6)
    sim.game.max_steps = 10
    return EvaluateTool(
        simulations=[SimulationConfig(suite="arena", name="very_basic", env=sim, max_time_s=60)],
        policy_uris=[policy_uri],
    )


@stable_job(
    remote_gpus=1,
    remote_nodes=1,
    timeout_s=7200,
    acceptance=[
        AcceptanceCriterion(metric="overview/sps", threshold=40000),
        AcceptanceCriterion(metric="env_agent/heart.gained", operator=">", threshold=0.1),
    ],
)
def train_100m() -> TrainTool:
    """Arena single GPU - 100M timesteps."""
    return TrainTool(
        trainer=TrainerConfig(total_timesteps=100_000_000),
        training_env=TrainingEnvironmentConfig(curriculum=make_curriculum()),
        policy_architecture=ViTDefaultConfig(),
    )


@stable_job(
    remote_gpus=4,
    remote_nodes=4,
    timeout_s=172800,
    acceptance=[
        AcceptanceCriterion(metric="overview/sps", threshold=80000),
        AcceptanceCriterion(metric="env_agent/heart.gained", operator=">", threshold=1.0),
    ],
)
def train_2b() -> TrainTool:
    """Arena multi GPU - 2B timesteps."""
    return TrainTool(
        trainer=TrainerConfig(total_timesteps=2_000_000_000),
        training_env=TrainingEnvironmentConfig(curriculum=make_curriculum()),
        policy_architecture=ViTDefaultConfig(),
    )
