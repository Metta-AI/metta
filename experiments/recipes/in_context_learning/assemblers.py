import random
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

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
from metta.tools.sim import SimTool
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
    num_generators: list[int],
    widths: list[int],
    heights: list[int],
    include_Any=False,
) -> dict:
    positions = []  # currently altar and converter positions are the same
    for n in num_agents:
        positions.extend(num_agents_to_positions[n])
    if include_Any:
        positions.extend((["Any"] * n for n in num_agents))
    return {
        "num_agents": num_agents,
        "num_altars": num_altars,
        "num_generators": num_generators,
        "widths": widths,
        "heights": heights,
        "positions": positions,
    }


def calculate_max_steps(num_objects: int, width: int, height: int) -> int:
    area = width * height
    max_steps = max(150, area * num_objects * 2)
    return min(max_steps, 1800)


curriculum_args = {
    "single_agent_two_altars": {
        "num_agents": [1],
        "num_altars": [2],
        "num_generators": [0],
        "widths": [5, 6, 7, 8],
        "heights": [5, 6, 7, 8],
    },
    "single_agent_many_altars": {
        "num_agents": [1],
        "num_altars": list(range(4, 16, 2)),
        "num_generators": [0],
        "widths": list(range(7, 16, 2)),
        "heights": list(range(7, 16, 2)),
    },
    "single_agent_many_altars_with_any": {
        "num_agents": [1],
        "num_altars": list(range(4, 16, 2)),
        "num_generators": [0],
        "widths": list(range(7, 16, 2)),
        "heights": list(range(7, 16, 2)),
        "include_Any": True,
    },
    "two_agent_5_altars_pattern": {
        "num_agents": [2],
        "num_altars": [5],
        "num_generators": [0],
        "widths": list(range(7, 16, 2)),
        "heights": list(range(7, 16, 2)),
    },
    "two_agent_two_altars_pattern": {
        "num_agents": [2],
        "num_altars": [2],
        "num_generators": [0],
        "widths": list(range(7, 14, 2)),
        "heights": list(range(7, 14, 2)),
    },
        "two_agent_two_altars_progressive_pattern": {
        "num_agents": [2],
        "num_altars": [2],
        "num_converters": [0],
        "widths": [5, 6, 7, 8],
        "heights": [5, 6, 7, 8],
        "generator_positions": [["Any"]],
        "altar_positions": [
            ["N", "S", "E"], ["N", "W", "E"], ["S", "E", "W"], ["S", "W", "E"],
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
    "three_agents_two_altars": {
        "num_agents": [3],
        "num_altars": [2, 4],
        "num_generators": [0],
        "widths": list(range(7, 16, 2)),
        "heights": list(range(7, 16, 2)),
        "include_Any": False,
    },
    "three_agent_many_altars_with_any": {
        "num_agents": [3],
        "num_altars": list(range(4, 16, 2)),
        "num_generators": [0],
        "widths": list(range(7, 16, 2)),
        "heights": list(range(7, 16, 2)),
        "include_Any": True,
    },
    "multi_agent_multi_altars": {
        "num_agents": [1, 2, 3],
        "num_altars": list(range(4, 16, 2)),
        "num_generators": [0],
        "widths": list(range(7, 16, 2)),
        "heights": list(range(7, 16, 2)),
    },
    "two_agents_1g_1a": {
        "num_agents": [2],
        "num_altars": [1],
        "num_generators": [1],
        "widths": list(range(5, 10)),
        "heights": list(range(5, 10)),
    },
    "multi_agents_1g_1a": {
        "num_agents": [1, 2, 3],
        "num_altars": [1],
        "num_generators": [1],
        "widths": list(range(5, 16, 2)),
        "heights": list(range(5, 16, 2)),
    },
    "multi_agents_1g_1a_with_any": {
        "num_agents": [1, 2, 3],
        "num_altars": [1],
        "num_generators": [1],
        "widths": list(range(5, 16, 2)),
        "heights": list(range(5, 16, 2)),
        "include_Any": True,
    },
    "three_agents_1g_1a": {
        "num_agents": [3],
        "num_altars": [1],
        "num_generators": [1],
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
        num_altars: list[int] = Field(default=[2])
        num_generators: list[int] = Field(default=[0])
        positions: list[list[Position]]
        widths: list[int] = Field(default=[6])
        heights: list[int] = Field(default=[6])

    def __init__(self, config: "AssemblerTaskGenerator.Config"):
        super().__init__(config)
        self.config = config
        self.converter_types = CONVERTER_TYPES.copy()
        self.resource_types = RESOURCE_TYPES.copy()

    def _make_generators(self, num_generators, cfg, position, rng: random.Random):
        """Make generators that input nothing and output resources for the altar"""
        generator_names = rng.sample(list(self.converter_types.keys()), num_generators)
        resources = rng.sample(self.resource_types, num_generators)
        for i, generator_name in enumerate(generator_names):
            cfg.map_builder_objects[generator_name] = 1
            generator = self.converter_types[generator_name].copy()
            recipe = (
                position,
                RecipeConfig(
                    input_resources={}, output_resources={resources[i]: 1}, cooldown=20
                ),
            )
            generator.recipes = [recipe]
            cfg.game_objects[generator_name] = generator

    def _make_altars(
        self, num_altars, cfg, position, num_generators, rng: random.Random
    ):
        cfg.map_builder_objects["altar"] = num_altars

        altar_cooldown = 25 + num_altars * 10 if num_generators == 0 else 1

        altar = building.assembler_altar.copy()
        # input recipe will either be nothing (if no generator) or some subset of generator resources
        input_resources = (
            {}
            if num_generators == 0
            else {
                resource: 1
                for resource in rng.sample(
                    self.resource_types, rng.randint(1, len(self.resource_types))
                )
            }
        )

        recipe = (
            position,
            RecipeConfig(
                input_resources=input_resources,
                output_resources={"heart": 1},
                cooldown=altar_cooldown,
            ),
        )
        altar.recipes = [recipe]
        cfg.game_objects["altar"] = altar

    def make_env_cfg(
        self,
        num_agents,
        num_instances,
        num_altars,
        num_generators,
        width,
        height,
        recipe_position,
        max_steps,
        rng: random.Random = random.Random(),
    ) -> MettaGridConfig:
        cfg = _BuildCfg()
        self._make_generators(num_generators, cfg, recipe_position, rng)

        self._make_altars(num_altars, cfg, recipe_position, num_generators, rng)

        return make_icl_assembler(
            num_agents=num_agents,
            num_instances=num_instances,
            max_steps=max_steps,
            game_objects=cfg.game_objects,
            map_builder_objects=cfg.map_builder_objects,
            width=width,
            height=height,
        )

    def _get_width_and_height(self, num_agents, num_altars, num_generators, rng):
        width = rng.choice(self.config.widths)
        height = rng.choice(self.config.heights)
        area = width * height
        minimum_area = (num_agents + num_altars + num_generators) * 2
        if area < minimum_area:
            width, height = minimum_area // 2, minimum_area // 2
        return width, height

    def _generate_task(self, task_id: int, rng: random.Random) -> MettaGridConfig:
        num_agents = rng.choice(self.config.num_agents)

        # positions must be the same length as the number of agents
        recipe_position = rng.choice(
            [p for p in self.config.positions if len(p) <= num_agents]
        )

        num_altars = rng.choice(self.config.num_altars)
        num_generators = rng.choice(self.config.num_generators)
        width, height = self._get_width_and_height(
            num_agents, num_altars, num_generators, rng
        )
        max_steps = calculate_max_steps(
            num_agents + num_altars + num_generators, width, height
        )

        if num_agents == 1:
            num_instances = 24
        elif num_agents == 2:
            num_instances = 12
        elif num_agents == 4:
            num_instances = 6
        elif num_agents == 12:
            num_instances = 2
        elif num_agents == 24:
            num_instances = 1
        else:
            raise ValueError(f"Invalid number of agents: {num_agents}")

        return self.make_env_cfg(
            num_agents,
            num_instances,
            num_altars,
            num_generators,
            width,
            height,
            recipe_position,
            max_steps,
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
    num_altars: int,
    num_generators: int,
    width: int,
    height: int,
    position: list[Position] = ["Any"],
) -> MettaGridConfig:
    task_generator_cfg = AssemblerTaskGenerator.Config(
        num_agents=[num_agents],
        num_altars=[num_altars],
        num_generators=[num_generators],
        positions=[position],
        widths=[width],
        heights=[height],
    )
    task_generator = AssemblerTaskGenerator(task_generator_cfg)
    return task_generator.get_task(random.randint(0, 1000000))


def make_curriculum(
    num_agents: list[int] = [1, 2],
    num_altars: list[int] = [2],
    num_generators: list[int] = [0, 1, 2],
    widths: list[int] = [4, 6, 8, 10],
    heights: list[int] = [4, 6, 8, 10],
    positions: list[list[Position]] = [["Any"], ["Any", "Any"]],
) -> CurriculumConfig:
    task_generator_cfg = AssemblerTaskGenerator.Config(
        num_agents=num_agents,
        num_altars=num_altars,
        num_generators=num_generators,
        widths=widths,
        heights=heights,
        positions=positions,
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

def evaluate(
    policy_uri: str, simulations: Optional[Sequence[SimulationConfig]] = None
) -> SimTool:
    # Local import to avoid circular import at module load time
    from experiments.evals.in_context_learning.assemblers import (
        make_assembler_eval_suite,
    )

    policy_uris = []
    for curriculum_style in curriculum_args:
        policy_uris.append(
            f"s3://softmax-public/policies/george.icl_assemblers_{curriculum_style}.2025-09-25/george.icl_assemblers_{curriculum_style}.2025-09-25:latest.pt"
        )

    simulations = simulations or make_assembler_eval_suite()
    return SimTool(
        simulations=simulations,
        policy_uris=policy_uris,
        stats_server_uri="https://api.observatory.softmax-research.net",
    )

# command tor un evalution: ./tools/run.py experiments.recipes.in_context_learning.assemblers.evaluate


def play_eval() -> PlayTool:
    num_agents = 12
    env = make_assembler_env(
        num_agents=num_agents,
        num_altars=30,
        width=30,
        height=30,
        num_generators=0,
        position=["Any", "Any", "Any"],
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
                f"run=george.icl_assemblers_{curriculum_style}.{time.strftime('%Y-%m-%d')}",
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


# ./tools/run.py experiments.recipes.in_context_learning.assemblers.replay curriculum_style=two_agent_two_altars_pattern policy_uri=s3://softmax-public/policies/icl_assemblers4_single_agent_two_altars.2025-09-23/icl_assemblers4_single_agent_two_altars.2025-09-23:v100.pt
