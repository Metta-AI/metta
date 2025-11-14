from typing import Optional, Sequence

import metta.cogworks.curriculum as cc
import mettagrid.builder.envs as eb

# from metta.agent.policies.vit import ViTDefaultConfig
from metta.agent.policies.cortex import CortexBaseConfig
from metta.agent.policy import PolicyArchitecture
from metta.cogworks.curriculum.curriculum import (
    CurriculumAlgorithmConfig,
    CurriculumConfig,
)
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
from metta.rl.loss import ActionSupervisedConfig
from metta.rl.trainer_config import TorchProfilerConfig, TrainerConfig
from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig
from metta.sim.simulation_config import SimulationConfig
from metta.sweep.core import Distribution as D
from metta.sweep.core import SweepParameters as SP
from metta.sweep.core import make_sweep
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
        # "ore_red": 0.1,
        # "battery_red": 0.8,
        # "laser": 0.5,
        # "armor": 0.5,
        # "blueprint": 0.5,
    }
    arena_env.game.agent.rewards.inventory_max = {
        "heart": 100,
        # "ore_red": 1,
        # "battery_red": 1,
        # "laser": 1,
        # "armor": 1,
        # "blueprint": 1,
    }

    # Easy converter: 1 battery_red to 1 heart (instead of 3 to 1)
    altar = arena_env.game.objects.get("altar")
    if isinstance(altar, ConverterConfig) and hasattr(altar, "input_resources"):
        altar.input_resources["battery_red"] = 1

    arena_env.game.map_builder.width = 70
    arena_env.game.map_builder.height = 70
    arena_env.game.actions.attack.enabled = False

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
    curriculum = curriculum or make_curriculum(enable_detailed_slice_logging=enable_detailed_slice_logging)

    eval_simulations = simulations()
    trainer_cfg = TrainerConfig(
        losses={"action_supervised": ActionSupervisedConfig()},
        detect_anomaly=True,
    )

    if policy_architecture is None:
        # policy_architecture = ViTDefaultConfig()
        policy_architecture = CortexBaseConfig()

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
        num_trials=80,
        # Default value is 1. We don't recommend going higher than 4.
        # The faster each individual trial, the lower you should set this number.
        num_parallel_trials=4,
    )


# # This file is for local experimentation only. It is not checked in, and therefore won't be usable on skypilot
# # You can run these functions locally with e.g. `./tools/run.py experiments.recipes.scratchpad.alex.train`
# # The VSCode "Run and Debug" section supports options to run these functions.
# from typing import List, Optional

# import metta.cogworks.curriculum as cc
# import mettagrid.builder.envs as eb
# # from metta.agent.meta_cog.mc_vit_reset import MCViTResetConfig
# from metta.agent.policies.vit import ViTDefaultConfig
# from metta.agent.policy import PolicyArchitecture
# from metta.cogworks.curriculum.curriculum import (
#     CurriculumAlgorithmConfig,
#     CurriculumConfig,
# )
# from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
# from metta.common.wandb.context import WandbConfig
# from metta.rl.loss.loss_config import LossConfig
# # from metta.rl.loss.mc_ppo import MCPPOConfig
# from metta.rl.loss.ppo import PPOConfig
# from metta.rl.system_config import SystemConfig
# from metta.rl.trainer_config import TrainerConfig
# from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig
# from metta.sim.simulation_config import SimulationConfig
# from metta.tools.eval import EvaluateTool
# from metta.tools.play import PlayTool
# from metta.tools.replay import ReplayTool
# from metta.tools.train import TrainTool
# from mettagrid import MettaGridConfig
# from mettagrid.config import ConverterConfig

# from experiments.recipes import arena


# def make_mettagrid(num_agents: int = 24) -> MettaGridConfig:
#     arena_env = eb.make_arena(num_agents=num_agents)

#     arena_env.game.agent.rewards.inventory = {
#         "heart": 1,
#         "ore_red": 0.1,
#         "battery_red": 0.8,
#         "laser": 0.5,
#         "armor": 0.5,
#         "blueprint": 0.5,
#     }
#     arena_env.game.agent.rewards.inventory_max = {
#         "heart": 100,
#         "ore_red": 1,
#         "battery_red": 1,
#         "laser": 1,
#         "armor": 1,
#         "blueprint": 1,
#     }

#     # Easy converter: 1 battery_red to 1 heart (instead of 3 to 1)
#     altar = arena_env.game.objects.get("altar")
#     if isinstance(altar, ConverterConfig) and hasattr(altar, "input_resources"):
#         altar.input_resources["battery_red"] = 1

#     return arena_env


# def make_curriculum(
#     arena_env: Optional[MettaGridConfig] = None,
#     enable_detailed_slice_logging: bool = False,
#     algorithm_config: Optional[CurriculumAlgorithmConfig] = None,
# ) -> CurriculumConfig:
#     arena_env = arena_env or make_mettagrid()

#     arena_tasks = cc.bucketed(arena_env)

#     for item in ["ore_red", "battery_red", "laser", "armor"]:
#         arena_tasks.add_bucket(
#             f"game.agent.rewards.inventory.{item}", [0, 0.1, 0.5, 0.9, 1.0]
#         )
#         arena_tasks.add_bucket(f"game.agent.rewards.inventory_max.{item}", [1, 2])

#     # enable or disable attacks. we use cost instead of 'enabled'
#     # to maintain action space consistency.
#     arena_tasks.add_bucket("game.actions.attack.consumed_resources.laser", [1, 100])

#     # sometimes add initial_items to the buildings
#     for obj in ["mine_red", "generator_red", "altar", "lasery", "armory"]:
#         arena_tasks.add_bucket(f"game.objects.{obj}.initial_resource_count", [0, 1])

#     if algorithm_config is None:
#         algorithm_config = LearningProgressConfig(
#             use_bidirectional=True,  # Enable bidirectional learning progress by default
#             ema_timescale=0.001,
#             exploration_bonus=0.1,
#             max_memory_tasks=1000,
#             max_slice_axes=5,  # More slices for arena complexity
#             enable_detailed_slice_logging=enable_detailed_slice_logging,
#         )

#     return arena_tasks.to_curriculum(algorithm_config=algorithm_config)


# def make_evals(env: Optional[MettaGridConfig] = None) -> List[SimulationConfig]:
#     basic_env = env or make_mettagrid()
#     basic_env.game.actions.attack.consumed_resources["laser"] = 100

#     combat_env = basic_env.model_copy()
#     combat_env.game.actions.attack.consumed_resources["laser"] = 1

#     return [
#         SimulationConfig(suite="arena", name="basic", env=basic_env),
#         SimulationConfig(suite="arena", name="combat", env=combat_env),
#     ]


# def train(
#     curriculum: Optional[CurriculumConfig] = None,
#     enable_detailed_slice_logging: bool = False,
#     policy_architecture: Optional[PolicyArchitecture] = None,
# ) -> TrainTool:
#     curriculum = curriculum or make_curriculum(
#         enable_detailed_slice_logging=enable_detailed_slice_logging
#     )

#     # eval_simulations = make_evals()
#     # evaluator = EvaluatorConfig(simulations=eval_simulations)
#     evaluator = EvaluatorConfig(evaluate_remote=False)


#     trainer_cfg = TrainerConfig(
#         # losses = LossConfig(loss_configs={"muesli": MuesliConfig()}),
#         losses=LossConfig(loss_configs={"ppo": PPOConfig()}),
#         # losses=LossConfig(loss_configs={"mc_ppo": MCPPOConfig()}),
#         )
#     # policy_config = FastDynamicsConfig()
#     # policy_config = FastLSTMResetConfig()
#     # policy_config = FastConfig()
#     # policy_config = ViTSmallConfig()
#     # policy_config = ViTSlidingTransConfig()
#     # policy_config = ViTLatentCrossAttentionConfig()
#     # policy_config = ViTSlidingTransWithGRUConfig()
#     # policy_config = ViTSlidingEmbCriticConfig()
#     # policy_config = ViTCrossAttentionTransConfig()
#     policy_config = ViTDefaultConfig()
#     # policy_config = MCViTResetConfig()
#     training_env = TrainingEnvironmentConfig(curriculum=curriculum)

#     print(policy_config)

#     return TrainTool(
#         trainer=trainer_cfg,
#         training_env=training_env,
#         evaluator=evaluator,
#         policy_architecture=policy_config,
#         wandb=WandbConfig.Off(),
#         stats_server_uri=None,
#         system=SystemConfig(local_only=True),
#         # optional hard-off for uploads even if W&B is off:
#         # uploader=UploaderConfig(epoch_interval=0),
#     )


# def play() -> PlayTool:
#     env = arena.make_evals()[0].env
#     env.game.max_steps = 100
#     cfg = arena.play(env)
#     return cfg


# def replay() -> ReplayTool:
#     env = arena.make_mettagrid()
#     env.game.max_steps = 100
#     cfg = arena.replay(env)
#     # cfg.policy_uri = "wandb://run/daveey.combat.lpsm.8x4"
#     return cfg


# def evaluate(run: str = "local.alex.1") -> EvaluateTool:
#     cfg = arena.evaluate(policy_uri=f"wandb://run/{run}")

#     # If your run doesn't exist, try this:
#     # cfg = arena.evaluate(policy_uri="wandb://run/daveey.combat.lpsm.8x4")
#     return cfg
