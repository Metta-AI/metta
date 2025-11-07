import typing

import metta.cogworks.curriculum as cc
import mettagrid.builder.envs as eb
import metta.agent.policies.vit
import metta.agent.policy
import metta.cogworks.curriculum.curriculum
import metta.cogworks.curriculum.learning_progress_algorithm
import metta.rl.trainer_config
import metta.rl.training
import metta.sim.simulation_config
import metta.sweep.core
import metta.tools.eval
import metta.tools.play
import metta.tools.replay
import metta.tools.sweep
import metta.tools.train
import mettagrid


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
    policy_architecture: typing.Optional[metta.agent.policy.PolicyArchitecture] = None,
) -> metta.tools.train.TrainTool:
    curriculum = curriculum or make_curriculum(
        enable_detailed_slice_logging=enable_detailed_slice_logging
    )

    eval_simulations = simulations()
    trainer_cfg = metta.rl.trainer_config.TrainerConfig()

    if policy_architecture is None:
        policy_architecture = metta.agent.policies.vit.ViTDefaultConfig()

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


def play(policy_uri: typing.Optional[str] = None) -> metta.tools.play.PlayTool:
    """Interactive play with a policy."""
    return metta.tools.play.PlayTool(sim=simulations()[0], policy_uri=policy_uri)


def replay(policy_uri: typing.Optional[str] = None) -> metta.tools.replay.ReplayTool:
    """Generate replay from a policy."""
    return metta.tools.replay.ReplayTool(sim=simulations()[0], policy_uri=policy_uri)


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


def sweep(sweep_name: str) -> metta.tools.sweep.SweepTool:
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
        metta.sweep.core.SweepParameters.LEARNING_RATE,
        metta.sweep.core.SweepParameters.PPO_CLIP_COEF,
        metta.sweep.core.SweepParameters.PPO_GAE_LAMBDA,
        metta.sweep.core.SweepParameters.PPO_VF_COEF,
        metta.sweep.core.SweepParameters.ADAM_EPS,
        metta.sweep.core.SweepParameters.param(
            "trainer.total_timesteps",
            metta.sweep.core.Distribution.INT_UNIFORM,
            min=5e8,
            max=2e9,
            search_center=7.5e8,
        ),
    ]

    return metta.sweep.core.make_sweep(
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
        num_parallel_trials=4,
    )
