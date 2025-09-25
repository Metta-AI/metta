"""

Here we want to experiment on whether the agents can in-context learn how to use assemblers with
arbitrary positions and recipes.


Options:

- only an altar, no input resource, only positions

- assembler converter, that has input resources and positions

- single agent versus multiagent

"""

from dis import Positions
import random
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from metta.cogworks.curriculum.curriculum import CurriculumConfig
from metta.cogworks.curriculum.task_generator import (
    TaskGenerator,
    TaskGeneratorConfig,
)
from metta.rl.trainer_config import LossConfig, TrainerConfig
from metta.rl.training.training_environment import TrainingEnvironmentConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.train import TrainTool
from mettagrid.builder import building
from mettagrid.builder.envs import make_icl_assembler
from mettagrid.config.mettagrid_config import (
    MettaGridConfig,
    Position,
    RecipeConfig,
)
from metta.agent.policies.fast_lstm_reset import FastLSTMResetConfig
from pydantic import Field

"""
curriculum 1: single agent, two altars in cooldown, different positions — all the way from any, to adjacent, to a particular square.
curriculum 2: single agent, converter and altar, different positions, different recipes
curriculum 3: single agent, 2 converters and altar, different positions, different recipes - two convertors either both relevant or only 1. For instance altar either takes resources from both convertors to give heart, or from only one convertor to give heart.
curriculum 4: multiagent, two altars in cooldown, different positions. Both agents need to configure the pattern on both altars.
curriculum 5: multiagent, converter and altar, different positions, different recipes.
curriculum 6: multiagent, 2 convertors and altar, agents need to learn in context which is the right convertor.
"""

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

num_agents_to_positions = {
    1: [["N"], ["S"], ["E"], ["W"]],
    2: [
        ["N", "S"],
        ["E", "W"],
        ["N", "E"],  # one agent must be north, the other agent must be east
        ["N", "W"],  # one agent must be north, the other agent must be west
        ["S", "E"],
        ["S", "W"],
    ],
    3: [
        ["N", "S", "E"],
        ["E", "W", "N"],
        ["W", "E", "S"],
        ["N", "S", "W"],
        ["S", "N", "E"],
    ],
    4: [
        ["N", "S", "E", "W"],
        ["E", "W", "N", "S"],
        ["W", "E", "S", "N"],
        ["N", "S", "W", "E"],
        ["S", "N", "E", "W"],
    ],
}


def make_curriculum_args(
    num_agents: list[int],
    num_altars: list[int],
    num_converters: list[int],
    widths: list[int],
    heights: list[int],
    include_Any=False,
) -> dict:
    positions = []  # currently altar and converter positions are the same
    for n in num_agents:
        positions.extend(num_agents_to_positions[n])
    if include_Any:
        positions.extend((["Any"] * n for n in num_agents))
    generator_positions = positions if len(num_converters) > 0 else None
    return {
        "num_agents": num_agents,
        "num_altars": num_altars,
        "num_converters": num_converters,
        "widths": widths,
        "heights": heights,
        "generator_positions": generator_positions,
        "altar_positions": positions,
    }


def calculate_max_steps(num_objects: int, width: int, height: int) -> int:
    area = width * height
    max_steps = max(150, area * num_objects * 2)
    return min(max_steps, 1800)


# TODO set max inventory to 1


curriculum_args = {
    "single_agent_two_altars": {
        "num_agents": [1],
        "num_altars": [2],
        "num_converters": [],
        "widths": [5, 6, 7, 8],
        "heights": [5, 6, 7, 8],
    },
    "single_agent_many_altars": {
        "num_agents": [1],
        "num_altars": list(range(4, 16, 2)),
        "num_converters": [],
        "widths": list(range(7, 16, 2)),
        "heights": list(range(7, 16, 2)),
    },
    "single_agent_many_altars_with_any": {
        "num_agents": [1],
        "num_altars": list(range(4, 16, 2)),
        "num_converters": [],
        "widths": list(range(7, 16, 2)),
        "heights": list(range(7, 16, 2)),
        "include_Any": True,
    },
    "two_agent_5_altars_pattern": {
        "num_agents": [2],
        "num_altars": [5],
        "num_converters": [],
        "widths": list(range(7, 16, 2)),
        "heights": list(range(7, 16, 2)),
    },
    "two_agent_two_altars_pattern": {
        "num_agents": [2],
        "num_altars": [2],
        "num_converters": [],
        "widths": list(range(7, 14, 2)),
        "heights": list(range(7, 14, 2)),
    },
    "three_agents_2_4_altars": {
        "num_agents": [3],
        "num_altars": [2, 4],
        "num_converters": [],
        "widths": list(range(7, 16, 2)),
        "heights": list(range(7, 16, 2)),
        "include_Any": False,
    },
    "three_agent_many_altars_with_any": {
        "num_agents": [3],
        "num_altars": list(range(4, 16, 2)),
        "num_converters": [],
        "widths": list(range(7, 16, 2)),
        "heights": list(range(7, 16, 2)),
        "include_Any": True,
    },
    "multi_agent_multi_altars": {
        "num_agents": [1, 2, 3],
        "num_altars": list(range(4, 16, 2)),
        "num_converters": [],
        "widths": list(range(7, 16, 2)),
        "heights": list(range(7, 16, 2)),
    },
    "two_agents_1g_1a": {
        "num_agents": [2],
        "num_altars": [1],
        "num_converters": [1],
        "widths": list(range(5, 10)),
        "heights": list(range(5, 10)),
    },
    "multi_agents_1g_1a": {
        "num_agents": [1, 2, 3],
        "num_altars": [1],
        "num_converters": [1],
        "widths": list(range(5, 16, 2)),
        "heights": list(range(5, 16, 2)),
    },
    "multi_agents_1g_1a_with_any": {
        "num_agents": [1, 2, 3],
        "num_altars": [1],
        "num_converters": [1],
        "widths": list(range(5, 16, 2)),
        "heights": list(range(5, 16, 2)),
        "include_Any": True,
    },
    "three_agents_1g_1a": {
        "num_agents": [3],
        "num_altars": [1],
        "num_converters": [1],
        "widths": list(range(5, 16, 2)),
        "heights": list(range(5, 16, 2)),
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
        generator_positions: list[list[Position]] | None = Field(default=None)
        altar_positions: list[list[Position]] | Any = Field(default=None)
        altar_inputs: list[str] | None = Field(default=None)
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
        width,
        height,
        converter_positions: list[Position],
        altar_positions: list[Position],
        max_steps: int,
        altar_input: str | None = None,
        rng: random.Random = random.Random(),
    ) -> MettaGridConfig:
        cfg = _BuildCfg()
        # sample num_converters converters - TODO i want this with replacement
        converter_names = rng.sample(list(self.converter_types.keys()), num_converters)
        resources = rng.sample(self.resource_types, num_converters)
        for i, converter_name in enumerate(converter_names):
            cfg.map_builder_objects[converter_name] = 1
            converter = self.converter_types[converter_name].copy()
            converter.recipes = []
            # TODO allow this to accomadate multiple options
            recipe = (
                converter_positions,
                RecipeConfig(
                    input_resources={}, output_resources={resources[i]: 1}, cooldown=20
                ),
            )
            converter.recipes.append(recipe)
            cfg.game_objects[converter_name] = converter
        cfg.map_builder_objects["altar"] = num_altars

        if num_converters == 0:
            altar_cooldown = 25 + num_altars * 10
        else:
            altar_cooldown = 1

        altar = building.assembler_altar.copy()
        if num_converters == 0:
            input_resources = {}
        elif altar_input == "all":
            input_resources = {c: 1 for c in resources}
        else:
            input_resources = {rng.sample(resources, 1)[0]: 1}

        altar.recipes = []
        # TODO allow this to accomadate multiple options
        recipe = (
            altar_positions,
            RecipeConfig(
                input_resources=input_resources,
                output_resources={"heart": 1},
                cooldown=altar_cooldown,
            ),
        )
        altar.recipes.append(recipe)
        cfg.game_objects["altar"] = altar

        # print(f"Generating a map with {num_agents} agents, {num_altars} altars, {num_converters} converters, {width}x{height}, altar positions: {altar_positions}, converter positions: {converter_positions}.")

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
        num_agents = rng.choice(self.config.num_agents)

        # altar positions must be the same length as the number of agents
        altar_position = rng.choice(
            [p for p in self.config.altar_positions if len(p) == num_agents]
        )

        # generator positions must be the same length as the number of agents
        if self.config.generator_positions and len(self.config.generator_positions) > 0:
            # currently generator and altar positions are the same
            generator_position = altar_position
            num_converters = rng.choice(self.config.num_converters)

        else:
            generator_position = None
            num_converters = 0
        num_altars = rng.choice(self.config.num_altars)
        width = rng.choice(self.config.widths)
        height = rng.choice(self.config.heights)
        altar_input = (
            rng.choice(self.config.altar_inputs)
            if self.config.altar_inputs and num_converters > 2
            else None
        )

        # ensure the area is large enough
        area = width * height
        minimum_area = (num_agents + num_altars + num_converters) * 2
        if area < minimum_area:
            width, height = minimum_area // 2, minimum_area // 2
        if width * height < num_agents + num_altars + num_converters:
            raise ValueError(
                f"Width ({width}) * height ({height}) must be greater than or equal to {num_agents + num_altars + num_converters}."
            )
        max_steps = calculate_max_steps(
            num_agents + num_altars + num_converters, width, height
        )

        if 24 % num_agents != 0:
            raise ValueError(
                f"Number of agents ({num_agents}) must be a divisor of 24."
            )
        num_instances = 24 // num_agents

        return self.make_env_cfg(
            num_agents,
            num_instances,
            num_altars,
            num_converters,
            width,
            height,
            generator_position,
            altar_position,
            max_steps,
            altar_input,
            rng,
        )


def make_mettagrid(
    curriculum_style: str = "single_agent_two_altars",
) -> MettaGridConfig:
    task_generator_cfg = AssemblerTaskGenerator.Config(
        **make_curriculum_args(**curriculum_args[curriculum_style])
    )
    task_generator = AssemblerTaskGenerator(task_generator_cfg)
    return task_generator.get_task(random.randint(0, 1000000))


def make_assembler_env(
    num_agents: int,
    max_steps: int,
    num_altars: int,
    num_converters: int,
    width: int,
    height: int,
    generator_position: list[Position] = ["Any"],
    altar_position: list[Position] = ["Any"],
    altar_input: str = "one",
) -> MettaGridConfig:
    task_generator_cfg = AssemblerTaskGenerator.Config(
        num_agents=[num_agents],
        max_steps=max_steps,
        num_altars=[num_altars],
        num_converters=[num_converters],
        generator_positions=[generator_position],
        altar_positions=[altar_position],
        altar_inputs=[altar_input],
        widths=[width],
        heights=[height],
    )
    task_generator = AssemblerTaskGenerator(task_generator_cfg)
    return task_generator.get_task(random.randint(0, 1000000))


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
    return CurriculumConfig(task_generator=task_generator_cfg)


def train(curriculum_style: str = "single_agent_two_altars") -> TrainTool:
    curriculum = make_curriculum(
        **make_curriculum_args(**curriculum_args[curriculum_style])
    )
    policy_config = FastLSTMResetConfig()
    trainer_cfg = TrainerConfig(
        losses=LossConfig(),
    )
    return TrainTool(
        trainer=trainer_cfg,
        training_env=TrainingEnvironmentConfig(curriculum=curriculum),
        policy_architecture=policy_config,
        stats_server_uri="https://api.observatory.softmax-research.net",
    )


def play_eval() -> PlayTool:
    env = make_assembler_env(
        num_agents=1,
        max_steps=512,
        num_altars=2,
        num_converters=0,
        width=6,
        height=6,
        altar_position=["W"],
        altar_input="one",
    )

    return PlayTool(
        sim=SimulationConfig(
            env=env,
            name="in_context_assemblers",
            suite="in_context_learning",
        ),
    )


def replay(curriculum_style: str = "single_agent_two_altars") -> ReplayTool:
    eval_env = make_mettagrid(curriculum_style)
    # Default to the research policy if none specified
    default_policy_uri = (
        "s3://softmax-public/policies/icl_assemblers3_two_agent_two_altars_pattern.2025-09-22/"
        "icl_assemblers3_two_agent_two_altars_pattern.2025-09-22:v500.pt"
    )
    default_policy_uri = "s3://softmax-public/policies/icl_assemblers3_two_agent_two_altars_pattern.2025-09-22/icl_assemblers3_two_agent_two_altars_pattern.2025-09-22:v500.pt"

    return ReplayTool(
        sim=SimulationConfig(
            env=eval_env,
            suite="in_context_learning",
            name="in_context_assemblers",
        ),
        policy_uri=default_policy_uri,
    )


def experiment():
    for curriculum_style in curriculum_args:
        subprocess.run(
            [
                "./devops/skypilot/launch.py",
                "experiments.recipes.in_context_learning.assemblers.train",
                f"run=icl_assemblers3_{curriculum_style}.{time.strftime('%Y-%m-%d')}",
                f"curriculum_style={curriculum_style}",
                "--gpus=4",
                "--heartbeat-timeout=3600",
                "--skip-git-check",
            ]
        )
        time.sleep(1)


def play(
    env: Optional[MettaGridConfig] = None,
    curriculum_style: str = "three_agents_2_4_altars",
) -> PlayTool:
    eval_env = env or make_mettagrid(curriculum_style)
    return PlayTool(
        sim=SimulationConfig(
            env=eval_env,
            suite="in_context_learning",
            name="eval",
        ),
    )


if __name__ == "__main__":
    experiment()
