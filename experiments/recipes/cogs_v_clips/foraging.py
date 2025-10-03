import random
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from metta.cogworks.curriculum.curriculum import CurriculumConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.train import TrainTool
from mettagrid.builder.envs import make_icl_assembler
from metta.cogworks.curriculum.task_generator import TaskGenerator, TaskGeneratorConfig
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
from metta.rl.loss import LossConfig
from metta.rl.trainer_config import TrainerConfig
from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig
from cogames.cogs_vs_clips.stations import (
    assembler as assembler_config,
    carbon_extractor,
    oxygen_extractor,
    germanium_extractor,
    silicon_extractor,
    chest as chest_cfg,
)
from metta.agent.policies.vit_reset import ViTResetConfig
from mettagrid.config.mettagrid_config import (
    MettaGridConfig,
    Position,
)
from mettagrid.config.mettagrid_config import RecipeConfig
from experiments.recipes.cogs_v_clips.config import (
    foraging_curriculum_args,
    size_ranges,
)


@dataclass
class _BuildCfg:
    game_objects: Dict[str, Any] = field(default_factory=dict)
    map_builder_objects: Dict[str, int] = field(default_factory=dict)


class ForagingTaskGenerator(TaskGenerator):
    """Pure foraging, no energy or chargers"""

    class Config(TaskGeneratorConfig["ForagingTaskGenerator"]):
        num_cogs: list[int]
        num_assemblers: list[int]
        num_extractors: list[int]
        num_chests: list[int]
        room_size: list[str]
        positions: list[list[Position]]

    def __init__(self, config: "ForagingTaskGenerator.Config"):
        super().__init__(config)
        self.config = config
        self.used_resources = set()
        self.extractors = {
            "carbon": carbon_extractor(),
            "silicon": silicon_extractor(),
            "germanium": germanium_extractor(),
            "oxygen": oxygen_extractor(),
        }

    def _get_width_and_height(self, room_size: str, rng: random.Random):
        lo, hi = size_ranges[room_size]
        return rng.randint(lo, hi), rng.randint(lo, hi)

    def _set_width_and_height(self, room_size, num_agents, num_objects, rng):
        """Set the width and height of the environment to be at least the minimum area required for the number of agents, altars, and generators."""
        width, height = self._get_width_and_height(room_size, rng)
        area = width * height
        minimum_area = num_agents + num_objects * 2
        if area < minimum_area:
            width, height = minimum_area // 2, minimum_area // 2
        return width, height

    def _calculate_max_steps(self, num_objects: int, width: int, height: int) -> int:
        area = width * height
        max_steps = max(150, area * num_objects * 2)
        return min(max_steps, 1800)

    def _make_extractors(self, num_extractors, cfg, rng: random.Random):
        """Make generators that input nothing and output resources for the altar"""
        for _ in range(num_extractors):
            resource = rng.choice(list(self.extractors.keys()))
            self.used_resources.add(resource)
            extractor = self.extractors[resource]
            extractor_name = extractor.name
            extractor.recipes = [
                (
                    ["Any"],
                    RecipeConfig(
                        input_resources={},
                        output_resources={resource: rng.choice([1, 5, 10])},
                        cooldown=1,
                    ),
                )
            ]
            cfg.game_objects[extractor_name] = extractor
            if extractor_name not in cfg.map_builder_objects:
                cfg.map_builder_objects[extractor.name] = 1
            else:
                cfg.map_builder_objects[extractor.name] += 1

    def _make_chests(self, num_chests, cfg):
        chest = chest_cfg()
        chest.position_deltas = [("N", 1), ("S", 1), ("E", 1), ("W", 1)]
        cfg.game_objects["chest"] = chest
        cfg.map_builder_objects["chest"] = num_chests

    def _make_assemblers(
        self,
        num_assemblers,
        cfg,
        position,
        num_extractors,
        rng: random.Random,
        max_input_resources=3,
    ):
        input_resources = {}
        if num_extractors > 0:
            input_resources.update(
                {
                    resource: rng.choice([1, 3, 5])
                    for resource in rng.sample(
                        list(self.used_resources),
                        rng.randint(
                            1, min(len(self.used_resources), max_input_resources)
                        ),
                    )
                }
            )

        assembler = assembler_config()
        assembler.recipes = [
            (
                position,
                RecipeConfig(
                    input_resources=input_resources,
                    output_resources={"heart": 1},
                    cooldown=1,
                ),
            )
        ]
        cfg.game_objects["assembler"] = assembler
        cfg.map_builder_objects["assembler"] = num_assemblers

    def _make_env_cfg(
        self,
        num_agents,
        num_instances,
        num_assemblers,
        num_extractors,
        width,
        height,
        recipe_position,
        num_chests,
        max_steps,
        rng: random.Random = random.Random(),
    ) -> MettaGridConfig:
        cfg = _BuildCfg()

        self._make_extractors(num_extractors, cfg, rng)

        self._make_assemblers(num_assemblers, cfg, recipe_position, num_extractors, rng)

        if num_chests > 0:
            # if using chests, then we get reward from hearts in chest
            self._make_chests(num_chests, cfg)
            inventory_rewards = {"heart": 0}
            stat_rewards = {"chest.heart.amount": 5}
            resource_limits = {"heart": 1}
        else:
            # otherwise, we get reward from heart in inventory
            inventory_rewards = {"heart": 1}
            stat_rewards = {}
            resource_limits = {"heart": 20}

        return make_icl_assembler(
            num_agents=num_agents,
            num_instances=num_instances,
            max_steps=max_steps,
            game_objects=cfg.game_objects,
            map_builder_objects=cfg.map_builder_objects,
            width=width,
            height=height,
            resources=list(self.used_resources) + ["heart"],
            inventory_rewards=inventory_rewards,
            resource_limits=resource_limits,
            stat_rewards=stat_rewards,
            terrain=rng.choice(["sparse", "balanced", "dense", "no-terrain"]),
        )

    def _generate_task(
        self, task_id: int, rng: random.Random
    ) -> MettaGridConfig:
        num_cogs = rng.choice(self.config.num_cogs)
        num_assemblers = rng.choice(self.config.num_assemblers)
        num_extractors = rng.choice(self.config.num_extractors)
        num_chests = rng.choice(self.config.num_chests)
        room_size = rng.choice(self.config.room_size)
        recipe_position = rng.choice(self.config.positions)

        num_objects = num_assemblers + num_extractors + num_chests

        width, height = self._set_width_and_height(
            room_size, num_cogs, num_objects, rng
        )
        max_steps = self._calculate_max_steps(num_objects, width, height)

        icl_env = self._make_env_cfg(
            num_agents=num_cogs,
            num_instances=24 // num_cogs,
            num_assemblers=num_assemblers,
            num_extractors=num_extractors,
            width=width,
            height=height,
            recipe_position=recipe_position,
            num_chests=num_chests,
            max_steps=max_steps,
            rng=rng,
        )

        icl_env.label = f"{room_size}_{num_objects}_objects"
        return icl_env

    def generate_task(
        self,
        task_id: int,
        rng: random.Random,
        num_instances: Optional[int] = None,
    ) -> MettaGridConfig:
        return self._generate_task(task_id, rng, num_instances)


def train(curriculum_style: str = "all") -> TrainTool:
    from experiments.evals.foraging import make_foraging_eval_suite

    task_generator_cfg = ForagingTaskGenerator.Config(
        **foraging_curriculum_args[curriculum_style]
    )
    curriculum = CurriculumConfig(
        task_generator=task_generator_cfg,
        algorithm_config=LearningProgressConfig(
            num_active_tasks=1000,
        ),
    )

    return TrainTool(
        trainer=TrainerConfig(
            losses=LossConfig(),
        ),
        training_env=TrainingEnvironmentConfig(curriculum=curriculum),
        policy_architecture=ViTResetConfig(),
        evaluator=EvaluatorConfig(
            simulations=make_foraging_eval_suite(),
            evaluate_remote=False,
            evaluate_local=True,
        ),
        stats_server_uri="https://api.observatory.softmax-research.net",
    )


def make_env(
    num_cogs=1,
    position=["Any"],
    num_assemblers=3,
    num_chests=1,
    num_extractors=1,
    sizes="small",
):
    task_generator = ForagingTaskGenerator(
        config=ForagingTaskGenerator.Config(
            num_cogs=[num_cogs],
            positions=[position],
            room_size=[sizes],
            num_assemblers=[num_assemblers],
            num_chests=[num_chests],
            num_extractors=[num_extractors],
        )
    )
    return task_generator.get_task(random.randint(0, 1000000))


def replay() -> ReplayTool:
    eval_env = make_env()
    policy_uri = "s3://softmax-public/policies/cogs_v_clips.level_1.eval_local.multi_agent_pairs_bases_vit_reset.2025-10-02/:latest"

    return ReplayTool(
        policy_uri=policy_uri,
        sim=SimulationConfig(suite="cogs_v_clips", env=eval_env, name="eval"),
    )


def experiment():
    import subprocess
    import time

    for curriculum_style in foraging_curriculum_args:
        subprocess.run(
            [
                "./devops/skypilot/launch.py",
                "experiments.recipes.cogs_v_clips.foraging.train",
                f"run=cogs_v_clips.foraging_{curriculum_style}.{random.randint(0, 10000)}.{time.strftime('%Y-%m-%d')}",
                f"curriculum_style={curriculum_style}",
                "--gpus=4",
                "--heartbeat-timeout=3600",
                "--skip-git-check",
            ]
        )
        time.sleep(1)


def make_mettagrid(task_generator) -> MettaGridConfig:
    return task_generator.get_task(random.randint(0, 1000000))


def play(curriculum_style: str = "pairs") -> PlayTool:
    task_generator = ForagingTaskGenerator(
        config=ForagingTaskGenerator.Config(
            **foraging_curriculum_args[curriculum_style]
        )
    )
    return PlayTool(
        sim=SimulationConfig(
            env=make_mettagrid(task_generator), suite="cogs_vs_clippies", name="play"
        )
    )


if __name__ == "__main__":
    experiment()
