from typing import List, Optional, Sequence

import metta.cogworks.curriculum as cc
import mettagrid.builder.envs as eb
from metta.cogworks.curriculum.curriculum import (
    CurriculumAlgorithmConfig,
    CurriculumConfig,
)
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
from metta.rl.loss.grpo import GRPOConfig
from metta.rl.loss.loss_config import LossConfig
from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig
from metta.rl.trainer_config import TrainerConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.eval_remote import EvalRemoteTool
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.sim import SimTool
from metta.tools.train import TrainTool
from mettagrid import MettaGridConfig
from mettagrid.config import ConverterConfig

# TODO(dehydration): make sure this trains as well as main on arena
# it's possible the maps are now different


def make_mettagrid(num_agents: int = 24) -> MettaGridConfig:
    arena_env = eb.make_arena(num_agents=num_agents)
    return arena_env


def make_curriculum(
    arena_env: Optional[MettaGridConfig] = None,
    enable_detailed_slice_logging: bool = False,
    algorithm_config: Optional[CurriculumAlgorithmConfig] = None,
) -> CurriculumConfig:
    arena_env = arena_env or make_mettagrid()

    arena_tasks = cc.bucketed(arena_env)

    # arena_tasks.add_bucket("game.map_builder.root.params.agents", [1, 2, 3, 4, 6])
    # arena_tasks.add_bucket("game.map_builder.width", [10, 20, 30, 40])
    # arena_tasks.add_bucket("game.map_builder.height", [10, 20, 30, 40])
    # arena_tasks.add_bucket("game.map_builder.instance_border_width", [0, 6])

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
            use_bidirectional=True,  # Default: bidirectional learning progress
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
) -> TrainTool:
    curriculum = curriculum or make_curriculum(
        enable_detailed_slice_logging=enable_detailed_slice_logging
    )

    return TrainTool(
        training_env=TrainingEnvironmentConfig(curriculum=curriculum),
        evaluator=EvaluatorConfig(simulations=make_evals()),
    )


def train_grpo(
    curriculum: Optional[CurriculumConfig] = None,
    enable_detailed_slice_logging: bool = False,
) -> TrainTool:
    """Train with GRPO loss instead of PPO."""
    curriculum = curriculum or make_curriculum(
        enable_detailed_slice_logging=enable_detailed_slice_logging
    )

    return TrainTool(
        training_env=TrainingEnvironmentConfig(curriculum=curriculum),
        evaluator=EvaluatorConfig(simulations=make_evals()),
        trainer=TrainerConfig(losses=LossConfig(loss_configs={"grpo": GRPOConfig()})),
    )


def train_shaped(rewards: bool = True, converters: bool = True) -> TrainTool:
    env_cfg = make_mettagrid()
    env_cfg.game.agent.rewards.inventory["heart"] = 1
    env_cfg.game.agent.rewards.inventory_max["heart"] = 100

    if rewards:
        env_cfg.game.agent.rewards.inventory.update(
            {
                "ore_red": 0.1,
                "battery_red": 0.8,
                "laser": 0.5,
                "armor": 0.5,
                "blueprint": 0.5,
            }
        )
        env_cfg.game.agent.rewards.inventory_max.update(
            {
                "ore_red": 1,
                "battery_red": 1,
                "laser": 1,
                "armor": 1,
                "blueprint": 1,
            }
        )

    if converters:
        altar = env_cfg.game.objects.get("altar")
        if isinstance(altar, ConverterConfig) and hasattr(altar, "input_resources"):
            altar.input_resources["battery_red"] = 1

    return TrainTool(
        training_env=TrainingEnvironmentConfig(curriculum=cc.env_curriculum(env_cfg)),
        evaluator=EvaluatorConfig(simulations=make_evals()),
    )


def play(env: Optional[MettaGridConfig] = None) -> PlayTool:
    eval_env = env or make_mettagrid()
    return PlayTool(sim=SimulationConfig(suite="arena", env=eval_env, name="eval"))


def replay(env: Optional[MettaGridConfig] = None) -> ReplayTool:
    eval_env = env or make_mettagrid()
    return ReplayTool(sim=SimulationConfig(suite="arena", env=eval_env, name="eval"))


def evaluate(
    policy_uri: str, simulations: Optional[Sequence[SimulationConfig]] = None
) -> SimTool:
    simulations = simulations or make_evals()
    return SimTool(
        simulations=simulations,
        policy_uris=[policy_uri],
    )


def evaluate_remote(
    policy_uri: str, simulations: Optional[Sequence[SimulationConfig]] = None
) -> EvalRemoteTool:
    simulations = simulations or make_evals()
    return EvalRemoteTool(
        simulations=simulations,
        policy_uri=policy_uri,
    )
