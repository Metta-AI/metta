"""A Cogs vs Clips version of the arena recipe.

This is meant as a basic testbed for CvC buildings / mechanics, not as a full-fledged recipe.
"""

from typing import Optional

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
from metta.tools.train import TrainTool
from mettagrid.config import MettaGridConfig
from cogames.cogs_vs_clips.scenarios import make_game
from mettagrid.mapgen.mapgen import MapGen


def make_mettagrid(
    num_cogs: int = 4,
    width: int = 10,
    height: int = 10,
    num_assemblers: int = 1,
    num_chargers: int = 1,
    num_carbon_extractors: int = 1,
    num_oxygen_extractors: int = 1,
    num_germanium_extractors: int = 1,
    num_silicon_extractors: int = 1,
    num_chests: int = 1,
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
    num_instances = 6
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
    env.game.num_agents = 24

    return env


def make_task_generator(facility_env: Optional[MettaGridConfig] = None):
    facility_env = facility_env or make_mettagrid()
    facility_tasks = cc.bucketed(facility_env)

    # Add buckets for whatever you want to bucket over, eg: rewards, recipes, regen amount
    facility_tasks.add_bucket("game.agent.rewards.inventory.heart", [0, 1, 2])
    # TODO The below gives an error, for some reason
    # facility_tasks.add_bucket("game.agent.rewards.stats.chest.heart.amount", [3, 5, 10])
    facility_tasks.add_bucket("game.agent.rewards.inventory.carbon", [0, 0.5, 1])
    facility_tasks.add_bucket("game.agent.rewards.inventory.oxygen", [0, 0.5, 1])
    facility_tasks.add_bucket("game.agent.rewards.inventory.germanium", [0, 0.5, 1])
    facility_tasks.add_bucket("game.agent.rewards.inventory.silicon", [0, 0.5, 1])
    facility_tasks.add_bucket("game.max_steps", [250, 500, 1000, 1500])

    return facility_tasks


def make_curriculum(
    facility_env: Optional[MettaGridConfig] = None,
    enable_detailed_slice_logging: bool = False,
    algorithm_config: Optional[CurriculumAlgorithmConfig] = None,
) -> CurriculumConfig:
    facility_tasks = make_task_generator(facility_env)

    if algorithm_config is None:
        algorithm_config = LearningProgressConfig(
            ema_timescale=0.001,
            exploration_bonus=0.15,
            max_memory_tasks=1000,
            max_slice_axes=3,
            progress_smoothing=0.15,
            num_active_tasks=1000,
            rand_task_rate=0.25,
        )

    return facility_tasks.to_curriculum(algorithm_config=algorithm_config)


def train(
    curriculum: Optional[CurriculumConfig] = None,
    enable_detailed_slice_logging: bool = False,
) -> TrainTool:
    from experiments.evals.cogs_v_clips import make_cogs_v_clips_ascii_evals

    resolved_curriculum = curriculum or make_curriculum(
        enable_detailed_slice_logging=enable_detailed_slice_logging
    )

    trainer_cfg = TrainerConfig(
        losses=LossConfig(),
    )

    evaluator_cfg = EvaluatorConfig(
        simulations=make_cogs_v_clips_ascii_evals(),
    )

    return TrainTool(
        trainer=trainer_cfg,
        training_env=TrainingEnvironmentConfig(curriculum=resolved_curriculum),
        evaluator=evaluator_cfg,
    )


def play(env: Optional[MettaGridConfig] = None) -> PlayTool:
    eval_env = env or make_mettagrid()
    return PlayTool(
        sim=SimulationConfig(suite="facility_env", env=eval_env, name="eval")
    )


def replay(env: Optional[MettaGridConfig] = None) -> ReplayTool:
    eval_env = env or make_mettagrid()
    policy_uri = "s3://softmax-public/policies/cogs_v_clips.level_1.eval_local.multi_agent_pairs_bases_vit_reset.2025-10-02/:latest"

    return ReplayTool(
        policy_uri=policy_uri,
        sim=SimulationConfig(suite="cogs_v_clips", env=eval_env, name="eval"),
    )


def experiment():
    import subprocess
    import time

    subprocess.run(
        [
            "./devops/skypilot/launch.py",
            "experiments.recipes.cogs_v_clips.facilities.train",
            f"run=cogs_v_clips.facilities.{random.randint(0, 10000)}.{time.strftime('%Y-%m-%d')}",
            "--gpus=4",
            "--heartbeat-timeout=3600",
            "--skip-git-check",
        ]
    )
