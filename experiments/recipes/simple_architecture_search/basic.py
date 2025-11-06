import typing

import metta.cogworks.curriculum as cc
import mettagrid.builder.envs as eb
import metta.agent.policies.agalite
import metta.agent.policies.fast
import metta.agent.policies.fast_dynamics
import metta.agent.policies.fast_lstm_reset
import metta.agent.policies.gtrxl
import metta.agent.policies.memory_free
import metta.agent.policies.puffer
import metta.agent.policies.transformer
import metta.agent.policies.trxl
import metta.agent.policies.trxl_nvidia
import metta.agent.policies.vit
import metta.agent.policies.vit_reset
import metta.agent.policies.vit_sliding_trans
import metta.cogworks.curriculum.curriculum
import metta.cogworks.curriculum.learning_progress_algorithm
import metta.rl.loss.loss_config
import metta.rl.trainer_config
import metta.rl.training
import metta.sim.simulation_config
import metta.sweep.core
import metta.tools.eval
import metta.tools.sweep
import metta.tools.train
import mettagrid

# Architecture configurations for benchmark testing
ARCHITECTURES = {
    "vit": metta.agent.policies.vit.ViTDefaultConfig(),
    "vit_sliding": metta.agent.policies.vit_sliding_trans.ViTSlidingTransConfig(),
    "vit_reset": metta.agent.policies.vit_reset.ViTResetConfig(),
    "transformer": metta.agent.policies.transformer.TransformerPolicyConfig(),
    "fast": metta.agent.policies.fast.FastConfig(),
    "fast_lstm_reset": metta.agent.policies.fast_lstm_reset.FastLSTMResetConfig(),
    "fast_dynamics": metta.agent.policies.fast_dynamics.FastDynamicsConfig(),
    "memory_free": metta.agent.policies.memory_free.MemoryFreeConfig(),
    "agalite": metta.agent.policies.agalite.AGaLiTeConfig(),
    "gtrxl": metta.agent.policies.gtrxl.gtrxl_policy_config(),
    "trxl": metta.agent.policies.trxl.trxl_policy_config(),
    "trxl_nvidia": metta.agent.policies.trxl_nvidia.trxl_nvidia_policy_config(),
    "puffer": metta.agent.policies.puffer.PufferPolicyConfig(),
}


def mettagrid(num_agents: int = 24) -> mettagrid.MettaGridConfig:
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
    arena_env: typing.Optional[mettagrid.MettaGridConfig] = None,
    enable_detailed_slice_logging: bool = False,
    algorithm_config: typing.Optional[
        metta.cogworks.curriculum.curriculum.CurriculumAlgorithmConfig
    ] = None,
) -> metta.cogworks.curriculum.curriculum.CurriculumConfig:
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

    if algorithm_config is None:
        algorithm_config = metta.cogworks.curriculum.learning_progress_algorithm.LearningProgressConfig(
            use_bidirectional=True,  # Enable bidirectional learning progress by default
            ema_timescale=0.001,
            exploration_bonus=0.1,
            max_memory_tasks=1000,
            max_slice_axes=5,  # More slices for arena complexity
            enable_detailed_slice_logging=enable_detailed_slice_logging,
        )

    return arena_tasks.to_curriculum(algorithm_config=algorithm_config)


def simulations(
    env: typing.Optional[mettagrid.MettaGridConfig] = None,
) -> list[metta.sim.simulation_config.SimulationConfig]:
    basic_env = env or mettagrid()
    basic_env.game.actions.attack.consumed_resources["laser"] = 100

    combat_env = basic_env.model_copy()
    combat_env.game.actions.attack.consumed_resources["laser"] = 1

    return [
        metta.sim.simulation_config.SimulationConfig(
            suite="arena", name="basic", env=basic_env
        ),
        metta.sim.simulation_config.SimulationConfig(
            suite="arena", name="combat", env=combat_env
        ),
    ]


def train(
    curriculum: typing.Optional[
        metta.cogworks.curriculum.curriculum.CurriculumConfig
    ] = None,
    enable_detailed_slice_logging: bool = False,
    arch_type: str = "fast",
) -> metta.tools.train.TrainTool:
    curriculum = curriculum or make_curriculum(
        enable_detailed_slice_logging=enable_detailed_slice_logging
    )

    eval_simulations = simulations()
    trainer_cfg = metta.rl.trainer_config.TrainerConfig(
        losses=metta.rl.loss.loss_config.LossConfig(),
    )

    policy_architecture = ARCHITECTURES[arch_type]

    return metta.tools.train.TrainTool(
        trainer=trainer_cfg,
        training_env=metta.rl.training.TrainingEnvironmentConfig(curriculum=curriculum),
        evaluator=metta.rl.training.EvaluatorConfig(simulations=eval_simulations),
        policy_architecture=policy_architecture,
        torch_profiler=metta.rl.trainer_config.TorchProfilerConfig(),
    )


def evaluate(
    policy_uris: typing.Optional[typing.Sequence[str]] = None,
) -> metta.tools.eval.EvaluateTool:
    """Evaluate policies on arena simulations."""
    return metta.tools.eval.EvaluateTool(
        simulations=simulations(), policy_uris=policy_uris or []
    )


def evaluate_in_sweep(policy_uri: str) -> metta.tools.eval.EvaluateTool:
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
        metta.sim.simulation_config.SimulationConfig(
            suite="sweep",
            name="basic",
            env=basic_env,
            num_episodes=10,  # 10 episodes for statistical reliability
            max_time_s=240,  # 4 minutes max per simulation
        ),
        metta.sim.simulation_config.SimulationConfig(
            suite="sweep",
            name="combat",
            env=combat_env,
            num_episodes=10,
            max_time_s=240,
        ),
    ]

    return metta.tools.eval.EvaluateTool(
        simulations=simulations,
        policy_uris=[policy_uri],
    )


def sweep_architecture(sweep_name: str) -> metta.tools.sweep.SweepTool:
    # NB: arch_type matches the corresponding input to "train", the train_entrypoint.
    architecture_parameter = metta.sweep.core.SweepParameters.categorical(
        "arch_type", list(ARCHITECTURES.keys())
    )
    return metta.sweep.core.grid_search(
        name=sweep_name,
        recipe="experiments.recipes.simple_architecture_search.basic",
        train_entrypoint="train",
        eval_entrypoint="evaluate_in_sweep",
        objective="evaluator/eval_sweep/score",
        parameters=[architecture_parameter],
        max_trials=200,
        num_parallel_trials=8,
    )
