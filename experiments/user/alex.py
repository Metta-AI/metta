# This file is for local experimentation only. It is not checked in, and therefore won't be usable on skypilot
# You can run these functions locally with e.g. `./tools/run.py experiments.recipes.scratchpad.alex.train`
# The VSCode "Run and Debug" section supports options to run these functions.
from typing import List, Optional

import metta.cogworks.curriculum as cc
import mettagrid.builder.envs as eb
from experiments.recipes import arena
from metta.agent.policies.fast_dynamics import FastDynamicsConfig
from metta.agent.policy import PolicyArchitecture
from metta.cogworks.curriculum.curriculum import (
    CurriculumAlgorithmConfig,
    CurriculumConfig,
)
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
from metta.rl.loss.loss_config import LossConfig
from metta.rl.loss.ppo import PPOConfig
from metta.rl.loss.dynamics import DynamicsConfig
from metta.rl.trainer_config import TrainerConfig
from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.sim import SimTool
from metta.tools.train import TrainTool
from mettagrid import MettaGridConfig
from mettagrid.config import ConverterConfig


def make_mettagrid(num_agents: int = 24) -> MettaGridConfig:
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
    arena_env = arena_env or make_mettagrid()

    arena_tasks = cc.bucketed(arena_env)

    for item in ["ore_red", "battery_red", "laser", "armor"]:
        arena_tasks.add_bucket(
            f"game.agent.rewards.inventory.{item}", [0, 0.1, 0.5, 0.9, 1.0]
        )
        arena_tasks.add_bucket(f"game.agent.rewards.inventory.{item}_max", [1, 2])

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


def make_evals(env: Optional[MettaGridConfig] = None) -> List[SimulationConfig]:
    basic_env = env or make_mettagrid()
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

    eval_simulations = make_evals()
    trainer_cfg = TrainerConfig(
        losses=LossConfig(
            loss_configs={"ppo": PPOConfig(), "dynamics": DynamicsConfig()}
        ),
    )
    policy_config = FastDynamicsConfig()
    # policy_config = FastLSTMResetConfig()
    # policy_config = FastConfig()
    # policy_config = ViTSmallConfig()
    # policy_config = ViTSlidingTransConfig()
    training_env = TrainingEnvironmentConfig(curriculum=curriculum)
    evaluator = EvaluatorConfig(simulations=eval_simulations)

    return TrainTool(
        trainer=trainer_cfg,
        training_env=training_env,
        evaluator=evaluator,
        policy_architecture=policy_config,
    )


def play() -> PlayTool:
    env = arena.make_evals()[0].env
    env.game.max_steps = 100
    cfg = arena.play(env)
    return cfg


def replay() -> ReplayTool:
    env = arena.make_mettagrid()
    env.game.max_steps = 100
    cfg = arena.replay(env)
    # cfg.policy_uri = "wandb://run/daveey.combat.lpsm.8x4"
    return cfg


def evaluate(run: str = "local.alex.1") -> SimTool:
    cfg = arena.evaluate(policy_uri=f"wandb://run/{run}")

    # If your run doesn't exist, try this:
    # cfg = arena.evaluate(policy_uri="wandb://run/daveey.combat.lpsm.8x4")
    return cfg


# # This file is for local experimentation only. It is not checked in, and therefore won't be usable on skypilot
# # You can run these functions locally with e.g. `./tools/run.py experiments.recipes.scratchpad.alex.train`
# # The VSCode "Run and Debug" section supports options to run these functions.
# from typing import List, Optional

# import metta.cogworks.curriculum as cc
# import metta.mettagrid.config.envs as eb
# from experiments.recipes import arena
# from metta.agent.agent_config import AgentConfig
# from metta.agent.pytorch_td.vit_lstm import ViTLSTMConfig
# from metta.agent.pytorch_td.fast_td import FastConfig
# from metta.cogworks.curriculum.curriculum import CurriculumConfig
# from metta.mettagrid.mettagrid_config import MettaGridConfig
# from metta.rl.loss.loss_config import LossConfig
# from metta.rl.trainer_config import (
#     CheckpointConfig,
#     EvaluationConfig,
#     TrainerConfig,
# )
# from metta.sim.simulation_config import SimulationConfig
# from metta.tools.play import PlayTool
# from metta.tools.replay import ReplayTool
# from metta.tools.sim import SimTool
# from metta.tools.train import TrainTool


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
#     arena_env.game.objects["altar"].input_resources = {"battery_red": 1}

#     return arena_env


# def make_curriculum(arena_env: Optional[MettaGridConfig] = None) -> CurriculumConfig:
#     arena_env = arena_env or make_mettagrid()

#     # make a set of training tasks for the arena
#     arena_tasks = cc.bucketed(arena_env)

#     for item in ["ore_red", "battery_red", "laser", "armor"]:
#         arena_tasks.add_bucket(
#             f"game.agent.rewards.inventory.{item}", [0, 0.1, 0.5, 0.9, 1.0]
#         )
#         arena_tasks.add_bucket(f"game.agent.rewards.inventory.{item}_max", [1, 2])

#     # enable or disable attacks. we use cost instead of 'enabled'
#     # to maintain action space consistency.
#     arena_tasks.add_bucket("game.actions.attack.consumed_resources.laser", [1, 100])

#     # sometimes add initial_items to the buildings
#     for obj in ["mine_red", "generator_red", "altar", "lasery", "armory"]:
#         arena_tasks.add_bucket(f"game.objects.{obj}.initial_resource_count", [0, 1])

#     return CurriculumConfig(task_generator=arena_tasks)


# def make_evals(env: Optional[MettaGridConfig] = None) -> List[SimulationConfig]:
#     basic_env = env or make_mettagrid()
#     basic_env.game.actions.attack.consumed_resources["laser"] = 100

#     combat_env = basic_env.model_copy()
#     combat_env.game.actions.attack.consumed_resources["laser"] = 1

#     return [
#         SimulationConfig(name="arena/basic", env=basic_env),
#         SimulationConfig(name="arena/combat", env=combat_env),
#     ]


# def train(curriculum: Optional[CurriculumConfig] = None) -> TrainTool:
#     trainer_cfg = TrainerConfig(
#         losses=LossConfig(),
#         curriculum=curriculum or make_curriculum(),
#         checkpoint=CheckpointConfig(
#             checkpoint_interval=50,  # 50 instead of default 5
#             wandb_checkpoint_interval=50,
#         ),
#         evaluation=EvaluationConfig(
#             simulations=[
#                 SimulationConfig(
#                     name="arena/basic", env=eb.make_arena(num_agents=24, combat=False)
#                 ),
#                 SimulationConfig(
#                     name="arena/combat", env=eb.make_arena(num_agents=24, combat=True)
#                 ),
#             ],
#             evaluate_remote=True,  # True instead of default False
#             evaluate_local=False,  # False instead of default True
#             skip_git_check=True,
#         ),
#     )
#     policy_config = FastConfig()
#     policy_config.lstm_config.hidden_size = 128
#     # policy_config = ViTLSTMConfig()
#     my_agent_config = AgentConfig(policy_config=policy_config)

#     return TrainTool(trainer=trainer_cfg, policy_architecture=my_agent_config)


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


# def evaluate(run: str = "local.alex.1") -> SimTool:
#     cfg = arena.evaluate(policy_uri=f"wandb://run/{run}")

#     # If your run doesn't exist, try this:
#     # cfg = arena.evaluate(policy_uri="wandb://run/daveey.combat.lpsm.8x4")
#     return cfg
