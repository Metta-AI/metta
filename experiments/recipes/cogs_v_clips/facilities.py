"""A Cogs vs Clips version of the arena recipe.

This is meant as a basic testbed for CvC buildings / mechanics, not as a full-fledged recipe.
"""

from typing import Optional, Sequence

import metta.cogworks.curriculum as cc
from metta.cogworks.curriculum.curriculum import (
    CurriculumAlgorithmConfig,
    CurriculumConfig,
)
import random
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
from metta.rl.loss import LossConfig
from metta.rl.trainer_config import TrainerConfig
from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.sim import SimTool
from metta.tools.train import TrainTool
from mettagrid.config import AssemblerConfig, MettaGridConfig
from cogames.cogs_vs_clips.scenarios import make_game
from mettagrid.mapgen.mapgen import MapGen


facility_env = {
    "num_cogs": 24,
    "width": 10,
    "height": 10,
    "num_assemblers": 1,
    "num_chargers": 0,
    "num_carbon_extractors": 0,
    "num_oxygen_extractors": 0,
    "num_germanium_extractors": 0,
    "num_silicon_extractors": 0,
    "num_chests": 0,
    "num_instances": 24,
    "border_width": 6,
    "instance_border_width": 3,
    "ascii_map": "training_facility_open_1.map",
}


def make_mettagrid(
    num_cogs: int = 24,
    width: int = 10,
    height: int = 10,
    num_assemblers: int = 1,
    num_chargers: int = 0,
    num_carbon_extractors: int = 0,
    num_oxygen_extractors: int = 0,
    num_germanium_extractors: int = 0,
    num_silicon_extractors: int = 0,
    num_chests: int = 0,
    dir="packages/cogames/src/cogames/maps/",
) -> MettaGridConfig:
    env = make_game(
        num_cogs,
        width,
        height,
        num_assemblers,
        num_chargers,
        num_carbon_extractors,
        num_oxygen_extractors,
        num_germanium_extractors,
        num_silicon_extractors,
        num_chests,
    )
    num_instances = 24 // num_cogs
    map_file = random.choice(
        [
            "training_facility_open_1.map",
            "training_facility_open_2.map",
            "training_facility_open_3.map",
            "training_facility_tight_4.map",
            "training_facility_tight_5.map",
        ]
    )
    env.game.map_builder = MapGen.Config(
        instances=num_instances,
        instance=MapGen.Config.with_ascii_uri(f"{dir}/{map_file}"),
    )

    return env


def make_curriculum(
    facility_env: Optional[MettaGridConfig] = None,
    enable_detailed_slice_logging: bool = False,
    algorithm_config: Optional[CurriculumAlgorithmConfig] = None,
) -> CurriculumConfig:
    facility_env = facility_env or make_mettagrid()

    facility_tasks = cc.bucketed(facility_env)

    # TODO add buckets for whatever you want to bucket over, eg: rewards, recipes, regen amount
    facility_tasks.add_bucket("game.agent.rewards.inventory.heart", [0, 10])

    facility_tasks.add_bucket("game.env.map_builder.", [0, 10])

    if algorithm_config is None:
        algorithm_config = LearningProgressConfig(
            use_bidirectional=True,  # Default: bidirectional learning progress
            ema_timescale=0.001,
            exploration_bonus=0.1,
            max_memory_tasks=1000,
            max_slice_axes=5,  # More slices for arena complexity
            enable_detailed_slice_logging=enable_detailed_slice_logging,
        )

    return facility_tasks.to_curriculum(algorithm_config=algorithm_config)


# def make_evals(env: Optional[MettaGridConfig] = None) -> List[SimulationConfig]:
#     basic_env = env or make_mettagrid()
#     basic_env.game.actions.attack.consumed_resources["laser"] = 100

#     combat_env = basic_env.model_copy()
#     combat_env.game.actions.attack.consumed_resources["laser"] = 1

#     return [
#         SimulationConfig(suite="cvc_arena", name="basic", env=basic_env),
#         SimulationConfig(suite="cvc_arena", name="combat", env=combat_env),
#     ]


def train(
    curriculum: Optional[CurriculumConfig] = None,
    enable_detailed_slice_logging: bool = False,
) -> TrainTool:
    resolved_curriculum = curriculum or make_curriculum(
        enable_detailed_slice_logging=enable_detailed_slice_logging
    )

    trainer_cfg = TrainerConfig(
        losses=LossConfig(),
    )

    evaluator_cfg = EvaluatorConfig(
        simulations=[],
    )

    return TrainTool(
        trainer=trainer_cfg,
        training_env=TrainingEnvironmentConfig(curriculum=resolved_curriculum),
        evaluator=evaluator_cfg,
    )


def train_shaped(rewards: bool = True, assemblers: bool = True) -> TrainTool:
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

    if assemblers:
        # Update altar recipe to require battery_red input
        altar_config = env_cfg.game.objects["altar"]
        assert isinstance(altar_config, AssemblerConfig)
        altar_config.recipes[0][1].input_resources["battery_red"] = 1

    trainer_cfg = TrainerConfig(
        losses=LossConfig(),
    )

    curriculum = cc.env_curriculum(env_cfg)

    return TrainTool(
        trainer=trainer_cfg,
        training_env=TrainingEnvironmentConfig(curriculum=curriculum),
        evaluator=EvaluatorConfig(simulations=make_evals(env_cfg)),
    )


def make_evals(env: Optional[MettaGridConfig] = None) -> list[SimulationConfig]:
    """Create evaluation simulations."""
    eval_env = env or make_mettagrid()
    return [SimulationConfig(suite="facility_env", env=eval_env, name="eval")]


def play(env: Optional[MettaGridConfig] = None) -> PlayTool:
    eval_env = env or make_mettagrid()
    return PlayTool(
        sim=SimulationConfig(suite="facility_env", env=eval_env, name="eval")
    )


def replay(env: Optional[MettaGridConfig] = None) -> ReplayTool:
    eval_env = env or make_mettagrid()
    return ReplayTool(
        sim=SimulationConfig(suite="cvc_arena", env=eval_env, name="eval")
    )


def evaluate(
    policy_uri: str, simulations: Optional[Sequence[SimulationConfig]] = None
) -> SimTool:
    simulations = simulations or make_evals()
    return SimTool(
        simulations=simulations,
        policy_uris=[policy_uri],
    )
