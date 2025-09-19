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
from metta.mettagrid.mettagrid_config import Position
from pydantic import Field
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import random
from metta.mettagrid.builder import building
from metta.mettagrid.builder.envs import make_icl_assembler
from metta.mettagrid.mettagrid_config import MettaGridConfig, RecipeConfig
from metta.rl.trainer_config import TrainerConfig
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.sim import SimTool
from metta.cogworks.curriculum.curriculum import CurriculumConfig
from metta.tools.train import TrainTool
from metta.sim.simulation_config import SimulationConfig

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

curriculum_args = {
    # 1) Single agent, only altars; positions vary (Any, W+E, N+S)
    "single_agent_only_altars": {
        "num_agents": [1],
        "num_altars": [2],
        "num_converters": [0],
        "widths": [6, 10, 12],
        "heights": [6, 10, 12],
        "generator_positions": [["Any"]],
        "altar_positions": [["Any"], ["W"], ["E"], ["N"], ["S"]],
    },

    # 2) Single agent, 1 converter + 1 altar; positions Any or single-side (N/S/E/W)
    "single_agent_converter_and_altar": {
        "num_agents": [1],
        "num_altars": [1],
        "num_converters": [1],
        "widths": [6, 10, 12],
        "heights": [6, 10, 12],
        "generator_positions": [["Any"], ["N"], ["S"], ["E"], ["W"]],
        "altar_positions": [["Any"], ["N"], ["S"], ["E"], ["W"]],
        "altar_inputs": ["one"],                          # one converter available
    },

    # 3) Single agent, 2 converters + 1 altar; only one converter required
    #    Positions: either Any for both, or both constrained to N+S or E+W
    "single_agent_two_converters_one_active": {
        "num_agents": [1],
        "num_altars": [1],
        "num_converters": [2],
        "widths": [6, 10, 12],
        "heights": [6, 10, 12],
        "generator_positions": [["Any"], ["N"], ["S"], ["E"], ["W"]],
        "altar_positions": [["Any"]],
        "altar_inputs": ["one"],                          # only one converter’s output needed
    },

    # 4) Multi-agent (up to 2 agents), 2 altars; Any positions
    "multi_agent_any": {
        "num_agents": [1, 2],
        "num_altars": [2],
        "num_converters": [0],
        "widths": [4, 6, 8, 10],
        "heights": [4, 6, 8, 10],
        "generator_positions": [["Any"]],          # no converters, ignored
        "altar_positions": [["Any"]],
        "altar_inputs": ["one"],
    },

    # 5) Multi-agent (2 agents), altars positioned N+S or W+E
    "multi_agent_altars": {
        "num_agents": [2],
        "num_altars": [2],
        "num_converters": [0],
        "widths": [4, 6, 8, 10],
        "heights": [4, 6, 8, 10],
        "generator_positions": [["Any", "Any"]],          # no converters, ignored
        "altar_positions": [["N", "S"], ["W", "E"]],
        "altar_inputs": ["one"],
    },
    "multi_agent_both": {
        "num_agents": [2],
        "num_altars": [1],
        "num_converters": [2],
        "widths": [4, 6, 8, 10],
        "heights": [4, 6, 8, 10],
        "generator_positions": [["Any", "Any"], ["N", "S"], ["E", "W"]],
        "altar_positions": [["Any"]],
        "altar_inputs": ["both"],
    },
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
        if len(altar_positions) > num_altars:
            altar_positions = altar_positions[:num_altars]

        if len(converter_positions) < num_agents:
            converter_positions = converter_positions + [converter_positions[0]] * (num_agents - len(converter_positions))
        if len(altar_positions) < num_altars:
            altar_positions = altar_positions + [altar_positions[0]] * (num_altars - len(altar_positions))

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
                    input_resources={}, output_resources={resources[i]: 1}, cooldown=10
                ),
            )
            converter.recipes = [recipe]
            cfg.game_objects[converter_name] = converter

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
                    cooldown=10,
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


def make_mettagrid(curriculum_style: str = "single_agent_only_altars") -> MettaGridConfig:
    task_generator_cfg = AssemblerTaskGenerator.Config( **curriculum_args[curriculum_style])
    task_generator = AssemblerTaskGenerator(task_generator_cfg)
    return task_generator.get_task(0)


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


def train(curriculum_style: str = "single_agent_only_altars") -> TrainTool:
    curriculum = make_curriculum(**curriculum_args[curriculum_style])
    trainer_cfg = TrainerConfig(
        curriculum=curriculum,
    )
    return TrainTool(trainer=trainer_cfg)

def play(curriculum_style: str = "single_agent_only_altars") -> PlayTool:
    eval_env = make_mettagrid(curriculum_style)
    return PlayTool(
        sim=SimulationConfig(
            env=eval_env,
            name="in_context_assemblers",
        ),
    )


def replay(curriculum_style: str = "single_agent_only_altars") -> ReplayTool:
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
