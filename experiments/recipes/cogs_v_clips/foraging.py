import random

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
from metta.agent.policies.fast_lstm_reset import FastLSTMResetConfig
from metta.agent.policies.vit_reset import ViTResetConfig

from mettagrid.config.mettagrid_config import (
    MettaGridConfig,
    Position,
)
from experiments.recipes.cogs_v_clips.utils import (
    foraging_curriculum_args,
    size_ranges,
    RESOURCES,
    make_assembler,
    make_extractor,
    make_chest,
    BuildCfg,
    make_agent,
)


class ForagingTaskGenerator(TaskGenerator):
    """Pure foraging, no energy or chargers"""

    class Config(TaskGeneratorConfig["ForagingTaskGenerator"]):
        num_cogs: list[int]
        num_assemblers: list[int]
        num_chests: list[int]
        size: list[int]
        assembler_positions: list[list[Position]]
        num_extractors: list[int] = [0]
        num_extractor_types: list[int] = [0]
        extractor_positions: list[list[Position]] = [["Any"]]

    def __init__(self, config: "ForagingTaskGenerator.Config"):
        super().__init__(config)
        self.config = config
        self.used_resources = set()

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

    def _calculate_max_steps(self, num_objects: int, size: int) -> int:
        area = size * size
        max_steps = max(150, area * num_objects * 3)
        max_steps = min(max_steps, 1500)
        return min(max_steps, 1000)

    def _make_extractors(
        self,
        num_extractors,
        num_extractor_types,
        cfg,
        extractor_position,
        rng: random.Random,
    ):
        """Make generators that input nothing and output resources for the altar"""
        resource_types = rng.sample(RESOURCES, num_extractor_types)

        # need to make sure there will be exactly num extractors
        extractor_counts = {
            r: num_extractors // num_extractor_types for r in resource_types
        }
        remaining = num_extractors - sum(list(extractor_counts.values()))
        if remaining > 0:
            extractor_counts[resource_types[0]] += remaining

        for resource, count in extractor_counts.items():
            if count > 0:
                self.used_resources.add(resource)
                extractor = make_extractor(
                    resource,
                    inputs={},
                    outputs={resource: 1},
                    position=extractor_position,
                )
                cfg.game_objects[extractor.name] = extractor
                cfg.map_builder_objects[extractor.name] = count

    def _make_chests(self, num_chests, cfg):
        chest = make_chest(position_deltas=[("N", 1), ("S", 1), ("E", 1), ("W", 1)])
        cfg.game_objects["chest"] = chest
        cfg.map_builder_objects["chest"] = num_chests

    def _make_assemblers(
        self,
        num_assemblers,
        cfg,
        position,
        num_extractors,
    ):
        input_resources = {}
        if num_extractors > 0:
            input_resources.update({resource: 1 for resource in self.used_resources})

        cooldown = max(20, num_assemblers * 3)

        assembler = make_assembler(
            input_resources, {"heart": 1}, position, cooldown=cooldown
        )
        cfg.game_objects["assembler"] = assembler
        cfg.map_builder_objects["assembler"] = num_assemblers

    def _make_env_cfg(
        self,
        num_agents,
        num_instances,
        num_assemblers,
        num_extractors,
        num_extractor_types,
        size,
        assembler_position,
        extractor_position,
        num_chests,
        max_steps,
        rng: random.Random = random.Random(),
    ) -> MettaGridConfig:
        cfg = BuildCfg()

        self._make_extractors(
            num_extractors,
            num_extractor_types,
            cfg,
            extractor_position,
            rng,
        )

        self._make_assemblers(num_assemblers, cfg, assembler_position, num_extractors)

        if num_chests > 0:
            # if using chests, then we get reward from hearts in chest
            self._make_chests(num_chests, cfg)
            inventory_rewards = {"heart": 0}
            stat_rewards = {"chest.heart.amount": 1}
            resource_limits = {"heart": 1}
        else:
            # otherwise, we get reward from heart in inventory
            inventory_rewards = {"heart": 1}
            stat_rewards = {}
            resource_limits = {"heart": 20}

        agent = make_agent(
            stat_rewards=stat_rewards,
            inventory_rewards=inventory_rewards,
            resource_limits=resource_limits,
        )

        if num_chests == 0:
            perimeter_object_names = ["assembler"] + [
                f"{resource}_extractor"
                for resource in ["carbon", "oxygen", "germanium", "silicon"]
            ]
            center_object_names = []
        elif num_chests <= 4:
            perimeter_object_names = ["assembler"] + [
                f"{resource}_extractor"
                for resource in ["carbon", "oxygen", "germanium", "silicon"]
            ]
            center_object_names = ["chest"]
        else:
            perimeter_object_names = list(cfg.map_builder_objects.keys())
            center_object_names = []

        perimeter_objects = {
            name: object
            for name, object in cfg.map_builder_objects.items()
            if name in perimeter_object_names
        }
        center_objects = {
            name: object
            for name, object in cfg.map_builder_objects.items()
            if name in center_object_names
        }

        return make_icl_assembler(
            num_agents=num_agents,
            num_instances=num_instances,
            max_steps=max_steps,
            game_objects=cfg.game_objects,
            perimeter_objects=perimeter_objects,
            center_objects=center_objects,
            resources=list(self.used_resources) + ["heart", "energy"],
            agent=agent,
            size=size,
            random_scatter=True,
        )

    def _generate_task(self, task_id: int, rng: random.Random) -> MettaGridConfig:
        num_cogs = rng.choice(self.config.num_cogs)
        num_assemblers = rng.choice(self.config.num_assemblers)
        num_extractors = rng.choice(self.config.num_extractors)
        num_chests = rng.choice(self.config.num_chests)
        size = rng.choice(self.config.size)
        assembler_position = rng.choice(self.config.assembler_positions)
        extractor_position = rng.choice(self.config.extractor_positions)
        if extractor_position != ["Any"]:
            # for now in situations where we have extractor positions, they should be the same as assembler positions
            extractor_position = assembler_position
        num_objects = num_assemblers + num_extractors + num_chests
        num_extractor_types = rng.choice(self.config.num_extractor_types)

        max_steps = self._calculate_max_steps(num_objects, size)

        icl_env = self._make_env_cfg(
            num_agents=num_cogs,
            num_instances=1,
            num_assemblers=num_assemblers,
            num_extractors=num_extractors,
            num_extractor_types=num_extractor_types,
            size=size,
            assembler_position=assembler_position,
            extractor_position=extractor_position,
            num_chests=num_chests,
            max_steps=max_steps,
            rng=rng,
        )

        icl_env.label = f"size_{size}_{num_objects}_objects"

        return icl_env


def train(
    curriculum_style: str = "assembly_lines_chests_pairs",
    architecture: str = "lstm_reset",
) -> TrainTool:
    from experiments.evals.cogs_v_clips.foraging import make_foraging_eval_suite

    task_generator_cfg = ForagingTaskGenerator.Config(
        **foraging_curriculum_args[curriculum_style]
    )
    curriculum = CurriculumConfig(
        task_generator=task_generator_cfg,
        algorithm_config=LearningProgressConfig(
            num_active_tasks=1000,
        ),
    )

    if architecture == "lstm_reset":
        policy_architecture = FastLSTMResetConfig()
    elif architecture == "vit_reset":
        policy_architecture = ViTResetConfig()

    return TrainTool(
        trainer=TrainerConfig(
            losses=LossConfig(),
        ),
        training_env=TrainingEnvironmentConfig(curriculum=curriculum),
        policy_architecture=policy_architecture,
        evaluator=EvaluatorConfig(
            simulations=make_foraging_eval_suite(),
            evaluate_remote=False,
            evaluate_local=True,
        ),
        stats_server_uri="https://api.observatory.softmax-research.net",
    )


def make_env(
    num_cogs=1,
    assembler_position=["Any"],
    extractor_position=["Any"],
    num_assemblers=3,
    num_chests=1,
    num_extractors=1,
    num_extractor_types=1,
    size=7,
):
    task_generator = ForagingTaskGenerator(
        config=ForagingTaskGenerator.Config(
            num_cogs=[num_cogs],
            assembler_positions=[assembler_position],
            extractor_positions=[extractor_position],
            size=[size],
            num_assemblers=[num_assemblers],
            num_chests=[num_chests],
            num_extractors=[num_extractors],
            num_extractor_types=[num_extractor_types],
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

    for architecture in ["lstm_reset", "vit_reset", "vit_default"]:
        for curriculum_style in foraging_curriculum_args:
            subprocess.run(
                [
                    "./devops/skypilot/launch.py",
                    "experiments.recipes.cogs_v_clips.foraging.train",
                    f"run=cogs_v_clips.foraging_{curriculum_style}_{architecture}.{random.randint(0, 10000)}.{time.strftime('%Y-%m-%d')}",
                    f"curriculum_style={curriculum_style}",
                    f"architecture={architecture}",
                    "--gpus=4",
                    "--heartbeat-timeout=3600",
                    "--skip-git-check",
                ]
            )
            time.sleep(1)


def make_mettagrid(task_generator) -> MettaGridConfig:
    return task_generator.get_task(random.randint(0, 1000000))


def play(curriculum_style: str = "assembly_lines_chests_pairs") -> PlayTool:
    task_generator = ForagingTaskGenerator(
        config=ForagingTaskGenerator.Config(
            **foraging_curriculum_args[curriculum_style]
        )
    )

    env = make_mettagrid(task_generator)

    # #single chest in middle
    # env = make_env(num_cogs=4, num_assemblers=10, num_extractors=0, num_chests=1, size=20, position=["N", "S"])

    # #assemblers and chests around
    # env = make_env(num_cogs=4, num_assemblers=10, num_extractors=0, num_chests=4, size=20, position=["N", "S"])

    # with extractors
    # env = make_env(
    #     num_cogs=4,
    #     num_assemblers=10,
    #     num_extractors=10,
    #     num_extractor_types=2,
    #     num_chests=4,
    #     size=20,
    #     assembler_position=["N", "S"],
    #     extractor_position=["Any"],
    # )

    return PlayTool(
        sim=SimulationConfig(env=env, suite="cogs_vs_clippies", name="play")
    )


if __name__ == "__main__":
    experiment()
