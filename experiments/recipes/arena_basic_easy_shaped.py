from typing import Optional, Sequence
from ray import tune
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
from metta.rl.trainer_config import TorchProfilerConfig, TrainerConfig
from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig
from metta.sim.simulation_config import SimulationConfig
from metta.sweep.core import ParameterSpec
from metta.sweep.core import SweepParameters as SP
from metta.sweep.ray.ray_controller import SweepConfig
from metta.tools.eval import EvaluateTool
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.ray_sweep import RaySweepTool
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
        arena_tasks.add_bucket(
            f"game.agent.rewards.inventory.{item}", [0, 0.1, 0.5, 0.9, 1.0]
        )
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


def sweep_full(sweep_name: str) -> RaySweepTool:
    """
    Comprehensive Ray sweep covering TrainerConfig and PPOConfig hyperparameters.
    """

    trainer_specs: list[ParameterSpec] = [
        SP.LEARNING_RATE,
        ParameterSpec("trainer.optimizer.beta1", tune.uniform(0.85, 0.99)),
        ParameterSpec("trainer.optimizer.beta2", tune.uniform(0.95, 0.9999)),
        ParameterSpec("trainer.optimizer.eps", tune.loguniform(1e-8, 1e-5)),
        ParameterSpec(
            "trainer.optimizer.weight_decay", tune.choice([0.0, 1e-6, 1e-5, 1e-4])
        ),
        ParameterSpec("trainer.optimizer.momentum", tune.uniform(0.8, 0.99)),
        # ParameterSpec("trainer.batch_size", tune.choice([131_072, 262_144, 524_288])),
        # ParameterSpec("trainer.minibatch_size", tune.choice([8_192, 16_384, 32_768])),
        # ParameterSpec("trainer.bptt_horizon", tune.choice([16, 32, 64, 128])),
        # ParameterSpec("trainer.update_epochs", tune.randint(1, 6)),
    ]

    ppo_specs: list[ParameterSpec] = [
        ParameterSpec(
            "trainer.losses.loss_configs.ppo.clip_coef", tune.uniform(0.005, 0.3)
        ),
        ParameterSpec(
            "trainer.losses.loss_configs.ppo.ent_coef", tune.loguniform(1e-4, 1e-1)
        ),
        ParameterSpec(
            "trainer.losses.loss_configs.ppo.gae_lambda", tune.uniform(0.8, 0.99)
        ),
        ParameterSpec(
            "trainer.losses.loss_configs.ppo.gamma", tune.uniform(0.95, 0.999)
        ),
        ParameterSpec(
            "trainer.losses.loss_configs.ppo.max_grad_norm", tune.uniform(0.1, 1.0)
        ),
        ParameterSpec(
            "trainer.losses.loss_configs.ppo.vf_clip_coef", tune.uniform(0.0, 0.5)
        ),
        ParameterSpec(
            "trainer.losses.loss_configs.ppo.vf_coef", tune.uniform(0.1, 1.0)
        ),
        ParameterSpec(
            "trainer.losses.loss_configs.ppo.l2_reg_loss_coef",
            tune.choice([0.0, 1e-6, 1e-5, 1e-4]),
        ),
        ParameterSpec(
            "trainer.losses.loss_configs.ppo.l2_init_loss_coef",
            tune.choice([0.0, 1e-6, 1e-5, 1e-4]),
        ),
        ParameterSpec(
            "trainer.losses.loss_configs.ppo.norm_adv", tune.choice([True, False])
        ),
        ParameterSpec(
            "trainer.losses.loss_configs.ppo.clip_vloss", tune.choice([True, False])
        ),
        ParameterSpec(
            "trainer.losses.loss_configs.ppo.target_kl",
            tune.choice([None, 0.01, 0.05, 0.1]),
        ),
        ParameterSpec(
            "trainer.losses.loss_configs.ppo.vtrace.rho_clip", tune.uniform(0.5, 2.0)
        ),
        ParameterSpec(
            "trainer.losses.loss_configs.ppo.vtrace.c_clip", tune.uniform(0.5, 2.0)
        ),
        ParameterSpec(
            "trainer.losses.loss_configs.ppo.prioritized_experience_replay.prio_alpha",
            tune.uniform(0.0, 1.0),
        ),
        ParameterSpec(
            "trainer.losses.loss_configs.ppo.prioritized_experience_replay.prio_beta0",
            tune.uniform(0.4, 1.0),
        ),
    ]

    search_space = {spec.path: spec.space for spec in (*trainer_specs, *ppo_specs)}
    search_space["trainer.total_timesteps"] = 2_000_000_000

    sweep_config = SweepConfig(
        sweep_id=sweep_name,
        recipe_module="experiments.recipes.arena_basic_easy_shaped",
        train_entrypoint="train",
        # No evals yet
        eval_entrypoint="evaluate_in_sweep",
        # No score key yet
        score_key="evaluator/eval_sweep/score",
        num_samples=100,
        gpus_per_trial=4,
        # Issues with concurrent trials computations
        # Issues with CPU compuations
        max_concurrent_trials=4,
    )

    return RaySweepTool(
        sweep_config=sweep_config,
        search_space=search_space,
    )
