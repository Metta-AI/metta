"""

Here we want to experiment on whether the agents can in-context learn how to use assemblers with
arbitrary positions and recipes.


Options:

- only an altar, no input resource, only positions

- assembler converter, that has input resources and positions

- single agent versus multiagent

"""

"""
curriculum 1: single agent, two altars in cooldown, different positions
curriculum 2: single agent, converter and altar, different positions, same recipe
curriculum 3: single agent, converter and altar, different positions, different recipes

curriculum 4: multiagent, two altars in cooldown, different positions
curriculum 5: multiagent, converter and altar, different positions, same recipe
curriculum 6: multiagent, converter and altar, different positions, different recipes
"""

from metta.cogworks.curriculum.task_generator import TaskGenerator, TaskGeneratorConfig
from metta.mettagrid.mettagrid_config import Position
from pydantic import Field
from dataclasses import dataclass, field
from typing import Dict, Any
import random
from metta.mettagrid.builder import building
from metta.mettagrid.builder.envs import make_icl_assembler
from metta.mettagrid.mettagrid_config import MettaGridConfig, RecipeConfig
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.sim import SimTool
from metta.tools.train import TrainTool
from metta.sim.simulation_config import SimulationConfig

CONVERTER_TYPES = {
    "generator_red": building.assembler_generator_red,
    "altar": building.assembler_altar,
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

@dataclass
class _BuildCfg:
    game_objects: Dict[str, Any] = field(default_factory=dict)
    map_builder_objects: Dict[str, int] = field(default_factory=dict)


class AssemblerTaskGenerator(TaskGenerator):
    class Config(TaskGeneratorConfig["AssemblerTaskGenerator"]):

        num_agents: list[int] = Field(default = [1])
        max_steps: int = 512
        num_altars: list[int] = Field(default = [2])
        num_converters: list[int] = Field(default = [0])
        generator_positions: list[list[Position]] = Field(default = [Position(["Any"])])
        altar_positions: list[list[Position]] = Field(default = [Position(["Any"])])
        widths: list[int] = Field(default = [6])
        heights: list[int] = Field(default = [6])

    def __init__(self, config: "AssemblerTaskGenerator.Config"):
        super().__init__(config)
        self.config = config

    def make_env_cfg(self, num_agents, num_instances, num_altars, num_converters, width, height, generator_position: Position, altar_position: list[Position], max_steps: int) -> MettaGridConfig:
        cfg = _BuildCfg()
        cfg.map_builder_objects["generator_red"] = num_converters
        cfg.map_builder_objects["altar"] = num_altars

        for _ in range(num_converters):
            # create a generator red, that outputs a battery red, and inputs nothing
            generator_red = building.assembler_generator_red
            generator_red.recipes = []
            # no input resources
            recipe = ([generator_position], RecipeConfig(input_resources={}, output_resources={"battery_red": 1}, cooldown=10))
            generator_red.recipes.append(recipe)
            cfg.game_objects["generator_red"] = generator_red


        for _ in range(num_altars):
            altar = building.assembler_altar
            altar.recipes = []
            if num_converters == 0:
                # create a altar, that outputs a heart, and inputs nothing
                for position in altar_position:
                    recipe = ([position], RecipeConfig(input_resources={}, output_resources={"heart": 1}, cooldown=10))
                    altar.recipes.append(recipe)
            else:
                # create a altar, that outputs a heart, and inputs one battery red
                for position in altar_position:
                    recipe = ([position], RecipeConfig(input_resources={"battery_red": 1}, output_resources={"heart": 1}, cooldown=10))
                    altar.recipes.append(recipe)

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

        if num_agents == 1:
            num_instances = 4
        elif num_agents == 2:
            num_instances = 2
        elif num_agents == 4:
            num_instances = 1
        else:
            raise ValueError(f"Invalid number of agents: {num_agents}")

        return self.make_env_cfg(num_agents, num_instances, num_altars, num_converters, width, height, generator_position, altar_position, max_steps)


# def make_mettagrid() -> MettaGridConfig:
#     return AssemblerTaskGenerator.Config().make_env_cfg(num_agents, num_instances, num_altars, num_converters, width, height, generator_position, altar_position, max_steps)



def play(env: Optional[MettaGridConfig] = None) -> PlayTool:
    eval_env = env or make_mettagrid()
    return PlayTool(
        sim=SimulationConfig(
            env=eval_env,
            name="in_context_resource_chain",
        ),
    )


def replay(env: Optional[MettaGridConfig] = None) -> ReplayTool:
    eval_env = env or make_mettagrid()
    # Default to the research policy if none specified
    default_policy_uri = (
        "s3://your-bucket/checkpoints/georgedeane.operant_conditioning.in_context_learning.all.0.1.08-19/"
        "georgedeane.operant_conditioning.in_context_learning.all.0.1.08-19:v50.pt"
    )
    return ReplayTool(
        sim=SimulationConfig(
            env=eval_env,
            name="in_context_resource_chain",
        ),
        policy_uri=default_policy_uri,
    )
