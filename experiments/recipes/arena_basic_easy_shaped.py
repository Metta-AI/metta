from typing import Optional, Sequence
import math
import random

import metta.cogworks.curriculum as cc
import mettagrid.builder.envs as eb
from metta.agent.policies.vit import ViTDefaultConfig
from metta.agent.policy import PolicyArchitecture
from metta.cogworks.curriculum.curriculum import (
    CurriculumAlgorithmConfig,
    CurriculumConfig,
)
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
from metta.rl.loss import LossConfig
from metta.rl.trainer_config import OptimizerConfig, TorchProfilerConfig, TrainerConfig
from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig
from metta.sim.simulation_config import SimulationConfig
from metta.sweep.core import make_sweep, SweepParameters as SP, Distribution as D
from metta.sweep.core import grid_search as make_grid_search
from metta.tools.eval import EvaluateTool
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.sweep import SweepTool
from metta.tools.train import TrainTool
from mettagrid import MettaGridConfig
from mettagrid.config import ConverterConfig


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

    # Easy converter: 1 battery_red to 1 heart (instead of 3 to 1)
    altar = arena_env.game.objects.get("altar")
    if isinstance(altar, ConverterConfig) and hasattr(altar, "input_resources"):
        altar.input_resources["battery_red"] = 1

    return arena_env


def make_curriculum(
    arena_env: Optional[MettaGridConfig] = None,
    enable_detailed_slice_logging: bool = False,
    algorithm_config: Optional[CurriculumAlgorithmConfig] = None,
) -> CurriculumConfig:
    arena_env = arena_env or mettagrid()

    arena_tasks = cc.bucketed(arena_env)

    for item in ["ore_red", "battery_red", "laser", "armor"]:
        arena_tasks.add_bucket(
            f"game.agent.rewards.inventory.{item}", [0, 0.1, 0.5, 0.9, 1.0]
        )
        arena_tasks.add_bucket(f"game.agent.rewards.inventory_max.{item}", [1, 2])

    # enable or disable attacks. we use cost instead of 'enabled'
    # to maintain action space consistency.
    arena_tasks.add_bucket("game.actions.attack.consumed_resources.laser", [1, 100])

    # sometimes add initial_items to the buildings
    for obj in ["mine_red", "generator_red", "altar", "lasery", "armory"]:
        arena_tasks.add_bucket(f"game.objects.{obj}.initial_resource_count", [0, 1])

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
) -> TrainTool:
    curriculum = curriculum or make_curriculum(
        enable_detailed_slice_logging=enable_detailed_slice_logging
    )

    eval_simulations = simulations()
    trainer_cfg = TrainerConfig(
        losses=LossConfig(),
    )

    if policy_architecture is None:
        policy_architecture = ViTDefaultConfig()

    return TrainTool(
        trainer=trainer_cfg,
        training_env=TrainingEnvironmentConfig(curriculum=curriculum),
        evaluator=EvaluatorConfig(simulations=eval_simulations),
        policy_architecture=policy_architecture,
        torch_profiler=TorchProfilerConfig(),
    )


def evaluate(policy_uris: Optional[Sequence[str]] = None) -> EvaluateTool:
    """Evaluate policies on arena simulations."""
    return EvaluateTool(simulations=simulations(), policy_uris=policy_uris or [])


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
            num_episodes=10,  # 10 episodes for statistical reliability
            max_time_s=240,  # 4 minutes max per simulation
        ),
        SimulationConfig(
            suite="sweep",
            name="combat",
            env=combat_env,
            num_episodes=10,
            max_time_s=240,
        ),
    ]

    return EvaluateTool(
        simulations=simulations,
        policy_uris=[policy_uri],
    )


def sweep(sweep_name: str) -> SweepTool:
    """
    Prototypical sweep function.
    In your own recipe, you likely only every need this. You can override other SweepTool parameters in the CLI.

    Example usage:
        `uv run ./tools/run.py experiments.recipes.arena_basic_easy_shaped.sweep sweep_name="ak.baes.10081528" -- gpus=4 nodes=2`

    We recommend running using local_test=True before running the sweep on the remote:
        `uv run ./tools/run.py experiments.recipes.arena_basic_easy_shaped.sweep sweep_name="ak.baes.10081528.local_test" -- local_test=True`
    This will run a quick local sweep and allow you to catch configuration bugs (NB: Unless those bugs are related to batch_size, minibatch_size, or hardware configuration).
    If this runs smoothly, you must launch the sweep on a remote sandbox (otherwise sweep progress will halt when you close your computer).

    Running on the remote:
        1 - Start a sweep controller sandbox: `./devops/skypilot/sandbox.py --sweep-controller`, and ssh into it.
        2 - Clean git pollution: `git clean -df && git stash`
        3 - Ensure your sky credentials are present: `sky status` -- if not, follow the instructions on screen.
        4 - Install tmux on the sandbox `apt install tmux`
        5 - Launch tmux session: `tmux new -s sweep`
        6 - Launch the sweep: `uv run ./tools/run.py experiments.recipes.arena_basic_easy_shaped.sweep sweep_name="ak.baes.10081528" -- gpus=4 nodes=2`
        7 - Detach when you want: CTRL+B then d
        8 - Attach to look at status/output: `tmux attach -t sweep_configs`

    Please tag Axel (akerbec@softmax.ai) on any bug report.
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
        recipe="experiments.recipes.arena_basic_easy_shaped",
        train_entrypoint="train",
        # NB: You MUST use a specific sweep eval suite, different than those in training.
        # Besides this being a recommended practice, using the same eval suite in both
        # training and scoring will lead to key conflicts that will lock the sweep.
        eval_entrypoint="evaluate_in_sweep",
        # Typically, "evaluator/eval_{suite}/score"
        objective="evaluator/eval_sweep/score",
        parameters=parameters,
        max_trials=80,
        # Default value is 1. We don't recommend going higher than 4.
        # The faster each individual trial, the lower you should set this number.
        num_parallel_trials=18,
    )


def train_grid(
    *,
    batch_multiplier: int = 1,
    num_epochs: int = 1,
    learning_rate_scale_exponent: float = 0.5,
) -> TrainTool:
    """Train entry with batch/epoch/LR scaling knobs for grid search.

    - Batch size: default × {1, 2, 4, 8}
    - Update epochs: {1, 2, 4}
    - LR scaling: lr = lr_default * (batch_multiplier ** learning_rate_scale_exponent)
    - World-size scaling enabled to keep per-rank batch appropriate
    - Total timesteps fixed at 2e9
    """

    curriculum = make_curriculum()
    eval_simulations = simulations()

    # Compute derived values from defaults
    base_batch = TrainerConfig.model_fields["batch_size"].default
    default_lr = OptimizerConfig.model_fields["learning_rate"].default
    scaled_batch = int(base_batch) * int(batch_multiplier)
    scaled_lr = float(default_lr) * math.pow(
        float(batch_multiplier), float(learning_rate_scale_exponent)
    )

    trainer_cfg = TrainerConfig(
        losses=LossConfig(),
        batch_size=scaled_batch,
        scale_batches_by_world_size=True,
        update_epochs=int(num_epochs),
        optimizer=OptimizerConfig(learning_rate=scaled_lr),
        total_timesteps=2_000_000_000,
    )

    return TrainTool(
        trainer=trainer_cfg,
        training_env=TrainingEnvironmentConfig(curriculum=curriculum),
        evaluator=EvaluatorConfig(simulations=eval_simulations),
        policy_architecture=ViTDefaultConfig(),
        torch_profiler=TorchProfilerConfig(),
    )


def hardware_grid_sweep(sweep_name: str) -> SweepTool:
    """Grid search (hardware-focused) over batch size, update epochs, LR scaling exponent, and seeds.

    Dimensions:
      - batch_multiplier: [1, 2, 4, 8]
      - num_epochs: [1, 2, 4]
      - learning_rate_scale_exponent: [1.0, 0.5]
      - system.seed: three random integers drawn at runtime
    """

    # Random seed choices generated at runtime
    seed_choices = [random.randint(1, 1_000_000) for _ in range(3)]

    parameters = [
        SP.categorical("batch_multiplier", [1, 2, 4, 8]),
        SP.categorical("num_epochs", [1, 2, 4]),
        SP.categorical("learning_rate_scale_exponent", [1.0, 0.5]),
        SP.categorical("training_env.forward_pass_minibatch_target_size", [4096, 8192]),
        SP.categorical("system.seed", seed_choices),
    ]

    # 4 (batch) * 3 (epochs) * 2 (LR exponent) * 2 (env fp minibatch target) * 3 (seeds) = 144 trials total
    return make_grid_search(
        name=sweep_name,
        recipe="experiments.recipes.arena_basic_easy_shaped",
        train_entrypoint="train_grid",
        eval_entrypoint="evaluate_in_sweep",
        objective="evaluator/eval_sweep/score",
        parameters=parameters,
        max_trials=144,
        num_parallel_trials=18,
    )
