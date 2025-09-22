"""

Here we want to experiment on whether the agents can in-context learn how to use assemblers with
arbitrary positions and recipes.


Options:

- only an altar, no input resource, only positions

- assembler converter, that has input resources and positions

- single agent versus multiagent

"""

"""
curriculum 1: single agent, two altars in cooldown, different positions — all the way from any, to adjacent, to a particular square.
curriculum 2: single agent, converter and altar, different positions, different recipes
curriculum 3: single agent, 2 converters and altar, different positions, different recipes - two convertors either both relevant or only 1. For instance altar either takes resources from both convertors to give heart, or from only one convertor to give heart.
curriculum 4: multiagent, two altars in cooldown, different positions. Both agents need to configure the pattern on both altars.
curriculum 5: multiagent, converter and altar, different positions, different recipes.
curriculum 6: multiagent, 2 convertors and altar, agents need to learn in context which is the right convertor.
"""

from metta.cogworks.curriculum.task_generator import TaskGenerator, TaskGeneratorConfig
from mettagrid.config.mettagrid_config import Position
from pydantic import Field
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import random
from mettagrid.builder import building
from mettagrid.builder.envs import make_icl_assembler
from mettagrid.config.mettagrid_config import MettaGridConfig, RecipeConfig
from metta.rl.trainer_config import TrainerConfig, LossConfig
from metta.rl.training.training_environment import TrainingEnvironmentConfig
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.sim import SimTool
from metta.cogworks.curriculum.curriculum import CurriculumConfig
from metta.tools.train import TrainTool
import subprocess
import time
from metta.sim.simulation_config import SimulationConfig
import numpy as np
CONVERTER_TYPES = {
    "generator_red": building.assembler_generator_red,
    "generator_blue": building.assembler_generator_blue,
    "generator_green": building.assembler_generator_green,
    "mine_red": building.assembler_mine_red,
    "mine_blue": building.assembler_mine_blue,
    "mine_green": building.assembler_mine_green,
}

RESOURCE_TYPES = [
    "ore_red",
    "ore_blue",
    "ore_green",
    "battery_red",
    "battery_blue",
    "battery_green",
    "laser",
    "blueprint",
    "armor",
]

# curriculum_args = {
#     # 1) Single agent, only altars; positions vary (Any, W+E, N+S)
#     "single_agent_only_altars": {
#         "num_agents": [1],
#         "num_altars": [2],
#         "num_converters": [0],
#         "widths": [6, 10, 12],
#         "heights": [6, 10, 12],
#         "generator_positions": [["Any"]],
#         "altar_positions": [["Any"], ["W"], ["E"], ["N"], ["S"]],
#     },
#
#     # 2) Single agent, 1 converter + 1 altar; positions Any or single-side (N/S/E/W) TODO: cur
#     "single_agent_converter_and_altar": {
#         "num_agents": [1],
#         "num_altars": [1],
#         "num_converters": [1],
#         "widths": [6, 10, 12],
#         "heights": [6, 10, 12],
#         "generator_positions": [["Any"], ["N"], ["S"], ["E"], ["W"]],
#         "altar_positions": [["Any"], ["N"], ["S"], ["E"], ["W"]],
#         "altar_inputs": ["one"],                          # one converter available
#     },
#
#     # 3) Single agent, 2 converters + 1 altar; only one converter required
#     #    Positions: either Any for both, or both constrained to N+S or E+W error: unknown object type generator_green
#     "single_agent_two_converters_one_active": {
#         "num_agents": [1],
#         "num_altars": [1],
#         "num_converters": [2],
#         "widths": [6, 10, 12],
#         "heights": [6, 10, 12],
#         "generator_positions": [["Any"], ["N"], ["S"], ["E"], ["W"]],
#         "altar_positions": [["Any"]],
#         "altar_inputs": ["one"],                          # only one converter’s output needed
#     },
#
#     # 4) Multi-agent (up to 2 agents), 2 altars; Any positions ISSUE: cooldown needs to be longer or we'll get degenerate strategies. is there another way to enforce alternate usage?
#     "multi_agent_any": {
#         "num_agents": [1, 2],
#         "num_altars": [2],
#         "num_converters": [0],
#         "widths": [4, 6, 8, 10],
#         "heights": [4, 6, 8, 10],
#         "generator_positions": [["Any"]],          # no converters, ignored
#         "altar_positions": [["Any"]],
#         "altar_inputs": ["one"],
#     },
#
#     # 5) Multi-agent (2 agents), altars positioned N+S or W+E
#     "multi_agent_altars": {
#         "num_agents": [2],
#         "num_altars": [2],
#         "num_converters": [0],
#         "widths": [4, 6, 8, 10],
#         "heights": [4, 6, 8, 10],
#         "generator_positions": [["Any", "Any"]],          # no converters, ignored
#         "altar_positions": [["N", "S"], ["W", "E"]],
#         "altar_inputs": ["one"],
#     },
#     "multi_agent_both": {
#         "num_agents": [2],
#         "num_altars": [1],
#         "num_converters": [2],
#         "widths": [4, 6, 8, 10],
#         "heights": [4, 6, 8, 10],
#         "generator_positions": [["Any", "Any"], ["N", "S"], ["E", "W"]],
#         "altar_positions": [["Any"]],
#         "altar_inputs": ["both"],
#     },
#     "multi_agent_one_converter_one_altar": {
#         "num_agents": [2],
#         "num_altars": [1],
#         "num_converters": [1],
#         "widths": [4, 6, 8, 10],
#         "heights": [4, 6, 8, 10],
#         "generator_positions": [["Any"], ["N"], ["S"], ["E"], ["W"]],
#         "altar_positions": [["Any"], ["N"], ["S"], ["E"], ["W"]],
#         "altar_inputs": ["one"],
#     },
# }

curriculum_args = {
    "single_agent_two_altars": {
        "num_agents": [1],
        "num_altars": [2],
        "num_converters": [0],
        "widths": [5, 6, 7, 8],
        "heights": [5, 6, 7, 8],
        "generator_positions": [["Any"]],
        "altar_positions": [
            ["Any"],
            ["N", "S"], ["E", "W"],
            ["N", "E"], ["N", "W"], ["S", "E"], ["S", "W"],
            ["N"], ["S"], ["E"], ["W"],
        ],
    },
    "two_agent_two_altars_pattern": {
        "num_agents": [2],
        "num_altars": [2],
        "num_converters": [0],
        "widths": [5, 6, 7, 8],
        "heights": [5, 6, 7, 8],
        "generator_positions": [["Any"]],
        "altar_positions": [
            ["Any"],
            ["N", "S"], ["E", "W"],
            ["N", "E"], ["N", "W"], ["S", "E"], ["S", "W"],
        ],
    },
    "two_agent_two_altars_any": {
        "num_agents": [2],
        "num_altars": [2],
        "num_converters": [0],
        "widths": [5, 6, 7, 8],
        "heights": [5, 6, 7, 8],
        "generator_positions": [["Any"]],
        "altar_positions": [["Any"]],
    },
    # "three_agents_two_altars": {
    #     "num_agents": [3],
    #     "num_altars": [2],
    #     "num_converters": [0],
    #     "widths": [4, 6, 8, 10],
    #     "heights": [4, 6, 8, 10],
    #     "generator_positions": [["Any"]],
    #     "altar_positions": [
    #         ["Any"],
    #         ["N", "S"], ["E", "W"],
    #         ["N", "E"], ["N", "W"], ["S", "E"], ["S", "W"],
    #         ["N"], ["S"], ["E"], ["W"],
    #     ],
    # },
}

@dataclass
class _BuildCfg:
    game_objects: Dict[str, Any] = field(default_factory=dict)
    map_builder_objects: Dict[str, int] = field(default_factory=dict)


class AssemblerTaskGenerator(TaskGenerator):
    class Config(TaskGeneratorConfig["AssemblerTaskGenerator"]):
        num_agents: list[int] = Field(default=[1])
        max_steps: int = 512
        num_altars: list[int] = Field(default=[2])
        num_converters: list[int] = Field(default=[0])
        generator_positions: list[list[Position]] = Field(default=[["Any"]])
        altar_positions: list[list[Position]] = Field(default=[["Any"]])
        altar_inputs: list[str] = Field(default=["one", "both"])
        widths: list[int] = Field(default=[6])
        heights: list[int] = Field(default=[6])

    def __init__(self, config: "AssemblerTaskGenerator.Config"):
        super().__init__(config)
        self.config = config
        self.converter_types = CONVERTER_TYPES.copy()
        self.resource_types = RESOURCE_TYPES.copy()

    def make_env_cfg(
        self,
        num_agents,
        num_instances,
        num_altars,
        num_converters,
        altar_input: str,
        width,
        height,
        converter_positions: list[Position],
        altar_positions: list[Position],
        max_steps: int,
        rng: random.Random,
    ) -> MettaGridConfig:
        cfg = _BuildCfg()


        # ensure the positions are the same length as the number of agents and altars
        if len(converter_positions) > num_agents:
            converter_positions = converter_positions[:num_agents]
        if len(altar_positions) > num_agents:
            altar_positions = altar_positions[:num_agents]

        if len(converter_positions) < num_agents:
            converter_positions = converter_positions + [converter_positions[0]] * (num_agents - len(converter_positions))
        if len(altar_positions) < num_agents:
            altar_positions = altar_positions + [altar_positions[0]] * (num_agents - len(altar_positions))

        print(f"converter_positions: {converter_positions}")
        print(f"altar_positions: {altar_positions}")

        # sample num_converters converters - TODO i want this with replacement
        converter_names = rng.sample(list(self.converter_types.keys()), num_converters)
        resources = rng.sample(self.resource_types, num_converters)
        for converter_name in converter_names:
            cfg.map_builder_objects[converter_name] = 1
        cfg.map_builder_objects["altar"] = num_altars

        for i in range(num_converters):
            # create a generator red, that outputs a battery red, and inputs nothing
            converter = self.converter_types[converter_names[i]]
            # no input resources
            recipe = (
                converter_positions,
                RecipeConfig(
                    input_resources={}, output_resources={resources[i]: 1}, cooldown=20
                ),
            )
            converter.recipes = [recipe]
            cfg.game_objects[converter_names[i]] = converter

        for _ in range(num_altars):
            altar = building.assembler_altar
            if num_converters == 0:
                input_resources = {}
            elif altar_input == "both":
                input_resources = {c: 1 for c in resources}
            elif altar_input == "one":
                input_resources = {rng.sample(resources, 1)[0]: 1}
            recipe = (
                altar_positions,
                RecipeConfig(
                    input_resources=input_resources,
                    output_resources={"heart": 1},
                    cooldown=20,
                ),
            )
            altar.recipes = [recipe]

            cfg.game_objects["altar"] = altar

        return make_icl_assembler(
            num_agents=num_agents,
            num_instances=num_instances,
            max_steps=max_steps,
            game_objects=cfg.game_objects,
            map_builder_objects=cfg.map_builder_objects,
            width=width,
            height=height,
        )

    def _generate_task(self, task_id: int, rng: random.Random) -> MettaGridConfig:
        altar_position = rng.choice(self.config.altar_positions)
        generator_position = rng.choice(self.config.generator_positions)
        num_agents = rng.choice(self.config.num_agents)
        num_altars = rng.choice(self.config.num_altars)
        num_converters = rng.choice(self.config.num_converters)
        width = rng.choice(self.config.widths)
        height = rng.choice(self.config.heights)
        max_steps = self.config.max_steps
        altar_input = rng.choice(self.config.altar_inputs)

        if num_agents == 1:
            num_instances = 4
        elif num_agents == 2:
            num_instances = 2
        elif num_agents == 4:
            num_instances = 1
        else:
            raise ValueError(f"Invalid number of agents: {num_agents}")

        return self.make_env_cfg(
            num_agents,
            num_instances,
            num_altars,
            num_converters,
            altar_input,
            width,
            height,
            generator_position,
            altar_position,
            max_steps,
            rng,
        )


def make_mettagrid(curriculum_style: str = "single_agent_two_altars") -> MettaGridConfig:
    task_generator_cfg = AssemblerTaskGenerator.Config( **curriculum_args[curriculum_style])
    task_generator = AssemblerTaskGenerator(task_generator_cfg)
    return task_generator.get_task(np.random.randint(0, 1000000))


def make_curriculum(
    num_agents: list[int] = [1, 2],
    num_altars: list[int] = [2],
    num_converters: list[int] = [0, 1, 2],
    widths: list[int] = [4, 6, 8, 10],
    heights: list[int] = [4, 6, 8, 10],
    generator_positions: list[list[Position]] = [["Any"], ["Any", "Any"]],
    altar_positions: list[list[Position]] = [["Any"], ["Any", "Any"]],
    altar_inputs: list[str] = ["one", "both"],
) -> CurriculumConfig:
    task_generator_cfg = AssemblerTaskGenerator.Config(
        num_agents=num_agents,
        num_altars=num_altars,
        num_converters=num_converters,
        widths=widths,
        heights=heights,
        generator_positions=generator_positions,
        altar_positions=altar_positions,
        altar_inputs=altar_inputs,
    )
    task_generator = AssemblerTaskGenerator(task_generator_cfg)
    return CurriculumConfig(task_generator=task_generator_cfg)


def train(curriculum_style: str = "single_agent_two_altars") -> TrainTool:
    curriculum = make_curriculum(**curriculum_args[curriculum_style])
    trainer_cfg = TrainerConfig(
        losses=LossConfig(),
    )
    trainer_cfg.batch_size = 4128768
    trainer_cfg.bptt_horizon = 512
    return TrainTool(trainer=trainer_cfg, training_env=TrainingEnvironmentConfig(curriculum=curriculum))


def play(curriculum_style: str = "single_agent_two_altars") -> PlayTool:
    eval_env = make_mettagrid(curriculum_style)
    return PlayTool(
        sim=SimulationConfig(
            env=eval_env,
            name="in_context_assemblers",
        ),
    )


def replay(curriculum_style: str = "single_agent_two_altars") -> ReplayTool:
    eval_env = make_mettagrid(curriculum_style)
    # Default to the research policy if none specified
    default_policy_uri = (
        "s3://your-bucket/checkpoints/georgedeane.operant_conditioning.in_context_learning.all.0.1.08-19/"
        "georgedeane.operant_conditioning.in_context_learning.all.0.1.08-19:v50.pt"
    )
    return ReplayTool(
        sim=SimulationConfig(
            env=eval_env,
            name="in_context_assemblers",
        ),
        policy_uri=default_policy_uri,
    )

def experiment():
    curriculum_styles = [
        "single_agent_two_altars",
        "two_agent_two_altars_pattern",
        "two_agent_two_altars_any",
    ]

    for curriculum_style in curriculum_styles:
        subprocess.run(
            [
                "./devops/skypilot/launch.py",
                "experiments.recipes.in_context_learning.assemblers.train",
                f"run=icl_assemblers2_{curriculum_style}.{time.strftime('%Y-%m-%d')}",
                f"curriculum_style={curriculum_style}",
                "--gpus=4",
                "--heartbeat-timeout=3600",
                "--skip-git-check",
            ]
        )
        time.sleep(1)


if __name__ == "__main__":
    experiment()
