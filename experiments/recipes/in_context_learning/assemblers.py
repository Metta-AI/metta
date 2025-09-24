"""

Here we want to experiment on whether the agents can in-context learn how to use assemblers with
arbitrary positions and recipes.


Options:

- only an altar, no input resource, only positions

- assembler converter, that has input resources and positions

- single agent versus multiagent

"""

import random
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
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
from mettagrid.builder import building, empty_converters
from mettagrid.builder.envs import make_icl_assembler
from mettagrid.config.mettagrid_config import (
    ActionConfig,
    ActionsConfig,
    AgentConfig,
    AgentRewards,
    ChangeGlyphActionConfig,
    GameConfig,
    MettaGridConfig,
    Position,
    RecipeConfig,
)
from mettagrid.map_builder.map_builder import GameMap, MapBuilder, MapBuilderConfig
from mettagrid.map_builder.utils import draw_border
from mettagrid.mapgen.mapgen import MapGen
from pydantic import Field

from experiments.recipes.in_context_learning.icl_resource_chain import ICLTaskGenerator

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
            ["N", "S"],
            ["E", "W"],
            ["N", "E"],
            ["N", "W"],
            ["S", "E"],
            ["S", "W"],
            ["N"],
            ["S"],
            ["E"],
            ["W"],
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
            ["N", "S"],
            ["E", "W"],
            ["N", "E"],
            ["N", "W"],
            ["S", "E"],
            ["S", "W"],
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

        # # ensure the positions are the same length as the number of agents and altars
        # if len(converter_positions) > num_agents:
        #     converter_positions = converter_positions[:num_agents]
        # if len(altar_positions) > num_agents:
        #     altar_positions = altar_positions[:num_agents]

        # if len(converter_positions) < num_agents:
        #     converter_positions = converter_positions + [converter_positions[0]] * (num_agents - len(converter_positions))
        # if len(altar_positions) < num_agents:
        #     altar_positions = altar_positions + [altar_positions[0]] * (num_agents - len(altar_positions))

        # print(f"converter_positions: {converter_positions}")
        # print(f"altar_positions: {altar_positions}")

        # sample num_converters converters - TODO i want this with replacement
        converter_names = rng.sample(list(self.converter_types.keys()), num_converters)
        resources = rng.sample(self.resource_types, num_converters)
        for i, converter_name in enumerate(converter_names):
            cfg.map_builder_objects[converter_name] = 1
            # create a generator red, that outputs a battery red, and inputs nothing
            converter = self.converter_types[converter_name].copy()
            # no input resources
            recipe = (
                converter_positions,
                RecipeConfig(
                    input_resources={}, output_resources={resources[i]: 1}, cooldown=20
                ),
            )
            converter.recipes = [recipe]
            cfg.game_objects[converter_name] = converter

        # NOTE: This is a hack to support multiple altars with different recipes.
        # The map builder expects unique names for game objects with different properties.
        # Here we create unique altar names and configurations.
        cfg.map_builder_objects["altar"] = num_altars

        altar = building.assembler_altar.copy()
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
            altar_input,
            width,
            height,
            generator_position,
            altar_position,
            max_steps,
            rng,
        )


def make_mettagrid(
    curriculum_style: str = "single_agent_two_altars",
) -> MettaGridConfig:
    task_generator_cfg = AssemblerTaskGenerator.Config(
        **curriculum_args[curriculum_style]
    )
    task_generator = AssemblerTaskGenerator(task_generator_cfg)
    return task_generator.get_task(np.random.randint(0, 1000000))


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
    return CurriculumConfig(task_generator=task_generator_cfg)


def train(curriculum_style: str = "single_agent_two_altars") -> TrainTool:
    curriculum = make_curriculum(**curriculum_args[curriculum_style])
    trainer_cfg = TrainerConfig(
        losses=LossConfig(),
    )
    trainer_cfg.batch_size = 4177920
    trainer_cfg.bptt_horizon = 512
    return TrainTool(
        trainer=trainer_cfg,
        training_env=TrainingEnvironmentConfig(curriculum=curriculum),
    )


# def play(curriculum_style: str = "single_agent_two_altars") -> PlayTool:
#     eval_env = make_mettagrid(curriculum_style)
#     return PlayTool(
#         sim=SimulationConfig(
#             env=eval_env,
#             name="in_context_assemblers",
#         ),
#     )


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
    return ReplayTool(
        sim=SimulationConfig(
            env=eval_env,
            suite="in_context_learning",
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
    curriculum_style: str = "single_agent_two_altars",
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


class ForagingMapBuilder(MapBuilder):
    class Config(MapBuilderConfig["ForagingMapBuilder"]):
        seed: Optional[int] = None
        width: int = 12
        height: int = 12
        agents: int | dict[str, int] = 1
        border_width: int = 0
        border_object: str = "wall"
        # Number of assemblers to place (all use key name "altar")
        num_assemblers: int = 1
        # Cluster specifications: list of dicts with keys
        #   name: object key to place (e.g., "mine_red")
        #   count: number of objects in this cluster
        #   distance: ring distance from assembler center
        #   radius: local cluster radius (default 1)
        clusters: list[dict] = []

    def __init__(self, config: "ForagingMapBuilder.Config"):
        # Mirror pattern used by other builders (e.g., AssemblerMapBuilder)
        self._config: ForagingMapBuilder.Config = config
        import numpy as _np

        self._rng = _np.random.default_rng(self._config.seed)

    def build(self) -> GameMap:
        import numpy as _np

        # Reset RNG if seed is provided to keep builds deterministic across calls
        if getattr(self._config, "seed", None) is not None:
            self._rng = _np.random.default_rng(self._config.seed)
        rng = self._rng

        h, w = self._config.height, self._config.width
        grid = _np.full((h, w), "empty", dtype="<U50")

        if self._config.border_width > 0:
            draw_border(grid, self._config.border_width, self._config.border_object)

        def in_bounds(i: int, j: int) -> bool:
            return 0 <= i < h and 0 <= j < w

        reserved = _np.zeros((h, w), dtype=bool)

        def reserve_3x3(i: int, j: int):
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    ii, jj = i + di, j + dj
                    if in_bounds(ii, jj):
                        reserved[ii, jj] = True

        # Place assemblers (altars). Start from center and then spread on a coarse grid.
        centers: list[tuple[int, int]] = []
        cx, cy = h // 2, w // 2
        centers.append((cx, cy))
        grid[cx, cy] = "altar"
        reserve_3x3(cx, cy)

        # Additional assemblers, if requested
        extra = self._config.num_assemblers - 1
        if extra > 0:
            # Sample positions around the central one with minimum spacing
            attempts = 0
            while extra > 0 and attempts < 1000:
                attempts += 1
                ang = rng.uniform(0, 2 * _np.pi)
                dist = rng.integers(low=3, high=max(4, min(h, w) // 3))
                i = int(cx + dist * _np.sin(ang))
                j = int(cy + dist * _np.cos(ang))
                if not in_bounds(i, j):
                    continue
                if reserved[
                    max(0, i - 1) : min(h, i + 2), max(0, j - 1) : min(w, j + 2)
                ].any():
                    continue
                grid[i, j] = "altar"
                reserve_3x3(i, j)
                centers.append((i, j))
                extra -= 1

        # Place clusters around each assembler center.
        clusters = list(self._config.clusters)
        if clusters:
            # Assign clusters evenly to assembler centers
            for idx, cluster in enumerate(clusters):
                name = str(cluster.get("name"))
                count = int(cluster.get("count", 2))
                distance = int(cluster.get("distance", 4))
                radius = int(cluster.get("radius", 1))

                base_i, base_j = centers[idx % len(centers)]
                # Choose an angle for this cluster around its base assembler
                theta = rng.uniform(0, 2 * _np.pi)
                ci = int(base_i + distance * _np.sin(theta))
                cj = int(base_j + distance * _np.cos(theta))

                # Clamp within bounds (keeping 1 cell margin)
                ci = max(1, min(h - 2, ci))
                cj = max(1, min(w - 2, cj))

                # Try to place 'count' objects near (ci, cj)
                placed = 0
                for _ in range(count * 6):  # a few attempts per item
                    if placed >= count:
                        break
                    oi = ci + rng.integers(-radius, radius + 1)
                    oj = cj + rng.integers(-radius, radius + 1)
                    if not in_bounds(oi, oj):
                        continue
                    if reserved[
                        max(0, oi - 1) : min(h, oi + 2), max(0, oj - 1) : min(w, oj + 2)
                    ].any():
                        continue
                    grid[oi, oj] = name
                    reserve_3x3(oi, oj)
                    placed += 1

        # Place agents randomly in remaining cells
        if isinstance(self._config.agents, int):
            num_agents = self._config.agents
        else:
            num_agents = sum(self._config.agents.values())

        empties = _np.argwhere((grid == "empty") & (~reserved))
        rng.shuffle(empties)
        for k in range(min(num_agents, len(empties))):
            i, j = map(int, empties[k])
            grid[i, j] = "agent.agent"

        return GameMap(grid)


def make_foraging_assembler_env(
    num_agents: int = 1,
    num_assemblers: int = 1,
    num_unique_resources: int = 2,
    cluster_distance: int = 4,
    cluster_size: int = 3,
    recipe_complexity: int = 2,
    width: int = 12,
    height: int = 12,
    cooldown: int = 60,
    agent_pattern: list[Position] | None = None,
) -> MettaGridConfig:
    """Create a foraging-style assembler environment with clustered resources.

    - Clusters of resource converters are placed at a fixed distance from the assembler(s).
    - The assembler recipe requires a configurable number of unique resources.
    - Agents have an inventory limit of 2 per item to encourage trips.
    """

    rng = random.Random(0)

    # Choose unique resources needed for clusters/recipes
    # For foraging, require directly-gatherable items only (no intermediate generators).
    # Keep recipes restricted to ores so clusters can be single-step gather → assemble.
    available_resources = [
        "ore_red",
        "ore_blue",
        "ore_green",
    ]
    chosen_resources = rng.sample(available_resources, k=max(1, num_unique_resources))

    # Map each resource to a converter object name. Use mines for foraging flavor.
    color_map = {"red": "red", "blue": "blue", "green": "green"}

    def resource_to_mine(resource: str) -> str:
        # resource like "ore_red" or "battery_blue"
        color = resource.split("_")[-1]
        return f"mine_{color_map[color]}"

    cluster_specs: list[dict] = []
    for res in chosen_resources:
        cluster_specs.append(
            {
                "name": resource_to_mine(res),
                "count": int(cluster_size),
                "distance": int(cluster_distance),
                "radius": 1,
            }
        )

    # Build game_objects (types) and their recipes
    game_objects: Dict[str, Any] = {"wall": empty_converters.wall}

    # Converters for resources: direct mines (ConverterConfig). Faster cooldown for gameplay feel.
    for res in chosen_resources:
        mine_key = resource_to_mine(res)
        mine_cfg = getattr(building, mine_key).model_copy(deep=True)
        mine_cfg.cooldown = 10
        game_objects[mine_key] = mine_cfg

    # Assembler (altar) with a long cooldown requiring 'recipe_complexity' unique inputs
    # Use ConverterConfig altar to avoid positional/glyph semantics; deposit ores then get heart.
    altar = building.altar.model_copy(deep=True)
    required = chosen_resources[: max(1, min(recipe_complexity, len(chosen_resources)))]
    input_resources = {r: 1 for r in required}
    altar.input_resources = input_resources
    altar.cooldown = cooldown
    game_objects["altar"] = altar

    # Construct the env config directly (similar to make_icl_assembler but with our builder)
    if 24 % num_agents != 0:
        raise ValueError(f"Number of agents ({num_agents}) must be a divisor of 24.")
    num_instances = 24 // num_agents

    cfg = MettaGridConfig(
        game=GameConfig(
            max_steps=512,
            num_agents=num_agents * num_instances,
            objects=game_objects,
            map_builder=MapGen.Config(
                instances=num_instances,
                instance_map=ForagingMapBuilder.Config(
                    agents=num_agents,
                    width=width,
                    height=height,
                    num_assemblers=num_assemblers,
                    clusters=cluster_specs,
                ),
            ),
            actions=ActionsConfig(
                move=ActionConfig(),
                rotate=ActionConfig(enabled=False),
                get_items=ActionConfig(),
                put_items=ActionConfig(),
            ),
            agent=AgentConfig(
                rewards=AgentRewards(inventory={"heart": 1}),
                # Limit each inventory item to 2 (encourages foraging trips)
                default_resource_limit=2,
                resource_limits={"heart": 15},
            ),
        )
    )
    return cfg


def play_foraging() -> PlayTool:
    env = make_foraging_mettagrid()
    return PlayTool(
        sim=SimulationConfig(
            env=env,
            name="icl_foraging_assemblers",
            suite="in_context_learning",
        ),
    )


class ForagingTaskGenerator(ICLTaskGenerator):
    class Config(ICLTaskGenerator.Config):
        num_agents: list[int] = Field(default=[1])
        widths: list[int] = Field(default=[12])
        heights: list[int] = Field(default=[12])
        num_assemblers: list[int] = Field(default=[1])
        cluster_distances: list[int] = Field(default=[4])
        cluster_sizes: list[int] = Field(default=[3])
        altar_cooldowns: list[int] = Field(default=[60])
        altar_patterns: list[list[Position]] = Field(
            default=[
                ["Any"],
                ["N"],
                ["S"],
                ["E"],
                ["W"],
                ["N", "S"],
                ["E", "W"],
            ]
        )
        directional_recipes: bool = Field(default=True)
        direction_recipes: list[list[Position]] = Field(
            default=[["N"], ["S"], ["E"], ["W"]]
        )

    def __init__(self, config: "ForagingTaskGenerator.Config"):
        super().__init__(config)
        self.config = config

    def _generate_task(self, task_id: int, rng: random.Random) -> MettaGridConfig:
        # Sample knobs
        num_agents = rng.choice(self.config.num_agents)
        if 24 % num_agents != 0:
            raise ValueError(
                f"Number of agents ({num_agents}) must be a divisor of 24."
            )
        num_instances = 24 // num_agents

        width = rng.choice(self.config.widths)
        height = rng.choice(self.config.heights)
        num_assemblers = rng.choice(self.config.num_assemblers)
        cluster_distance = rng.choice(self.config.cluster_distances)
        cluster_size = rng.choice(self.config.cluster_sizes)
        cooldown = rng.choice(self.config.altar_cooldowns)

        # Sampling: unique resource types (ores only)
        available_resources = ["ore_red", "ore_blue", "ore_green"]
        # Use num_resources from ICLTaskGenerator.Config
        n_res = max(
            1, rng.choice(self.config.num_resources) if self.config.num_resources else 2
        )
        chosen_resources = rng.sample(
            available_resources, k=min(n_res, len(available_resources))
        )

        # Complexity: number of unique inputs required at altar
        if self.config.max_recipe_inputs:
            recipe_complexity = rng.choice(self.config.max_recipe_inputs)
        else:
            recipe_complexity = min(2, len(chosen_resources))

        color_map = {"red": "red", "blue": "blue", "green": "green"}

        def resource_to_mine(resource: str) -> str:
            color = resource.split("_")[-1]
            return f"mine_{color_map[color]}"

        cluster_specs: list[dict] = []
        for res in chosen_resources:
            cluster_specs.append(
                {
                    "name": resource_to_mine(res),
                    "count": int(cluster_size),
                    "distance": int(cluster_distance),
                    "radius": 1,
                }
            )

        # Objects: use AssemblerConfig semantics (spatial recipes)
        game_objects: Dict[str, Any] = {"wall": empty_converters.wall}
        for res in chosen_resources:
            mine_key = resource_to_mine(res)
            mine_cfg = CONVERTER_TYPES[mine_key].model_copy(deep=True)
            # Keep default spatial recipe from template (outputs ore); optional cooldown tuning could be done by
            # replacing the recipe's cooldown if needed.
            game_objects[mine_key] = mine_cfg

        altar = building.assembler_altar.model_copy(deep=True)
        if self.config.directional_recipes:
            recipes: list[tuple[list[Position], RecipeConfig]] = []
            for patt in self.config.direction_recipes:
                k = max(1, min(recipe_complexity, len(chosen_resources)))
                inputs = rng.sample(chosen_resources, k=k)
                recipes.append(
                    (
                        patt,
                        RecipeConfig(
                            input_resources={r: 1 for r in inputs},
                            output_resources={"heart": 1},
                            cooldown=cooldown,
                        ),
                    )
                )
            altar.recipes = recipes
        else:
            required = chosen_resources[
                : max(1, min(recipe_complexity, len(chosen_resources)))
            ]
            positions: list[Position] = rng.choice(self.config.altar_patterns)
            altar.recipes = [
                (
                    positions,
                    RecipeConfig(
                        input_resources={r: 1 for r in required},
                        output_resources={"heart": 1},
                        cooldown=cooldown,
                    ),
                )
            ]
        game_objects["altar"] = altar

        env_cfg = MettaGridConfig(
            game=GameConfig(
                max_steps=self.config.max_steps,
                num_agents=num_agents * num_instances,
                objects=game_objects,
                map_builder=MapGen.Config(
                    instances=num_instances,
                    instance_map=ForagingMapBuilder.Config(
                        agents=num_agents,
                        width=width,
                        height=height,
                        num_assemblers=num_assemblers,
                        clusters=cluster_specs,
                    ),
                ),
                actions=ActionsConfig(
                    move=ActionConfig(),
                    rotate=ActionConfig(enabled=False),
                    get_items=ActionConfig(enabled=False),
                    put_items=ActionConfig(enabled=False),
                    change_glyph=ChangeGlyphActionConfig(number_of_glyphs=16),
                ),
                agent=AgentConfig(
                    rewards=AgentRewards(inventory={"heart": 1}),
                    default_resource_limit=2,
                    resource_limits={"heart": 15},
                ),
            )
        )

        env_cfg.label = f"foraging_{len(chosen_resources)}res_{num_assemblers}assemblers_{width}x{height}"
        return env_cfg


def make_foraging_mettagrid() -> MettaGridConfig:
    cfg = ForagingTaskGenerator.Config(
        num_resources=[2, 3],
        max_recipe_inputs=[1, 2, 3],
        num_agents=[1],
        widths=[12],
        heights=[12],
        num_assemblers=[1],
        cluster_distances=[4],
        cluster_sizes=[3],
        altar_cooldowns=[60],
    )
    gen = ForagingTaskGenerator(cfg)
    return gen.get_task(0)


# -------- Directional multi-recipe maps --------


def make_foraging_directional_env(
    num_agents: int = 1,
    width: int = 12,
    height: int = 12,
    num_assemblers: int = 1,
    cluster_distance: int = 4,
    cluster_size: int = 3,
    low_cooldown: int = 10,
    high_cooldown: int = 80,
) -> MettaGridConfig:
    """Create a map where the altar has different recipes on N/S/E/W.

    - N: sink recipe (consume item, produce nothing)
    - S: expensive, low cooldown -> heart
    - E: cheap, high cooldown -> heart
    - W: expensive, high cooldown -> heart
    """

    if 24 % num_agents != 0:
        raise ValueError(f"Number of agents ({num_agents}) must be a divisor of 24.")
    num_instances = 24 // num_agents

    chosen_resources = ["ore_red", "ore_blue", "ore_green"]

    def resource_to_mine(resource: str) -> str:
        color = resource.split("_")[-1]
        return f"mine_{color}"

    # Mines with assembler semantics
    game_objects: Dict[str, Any] = {"wall": empty_converters.wall}
    for res in chosen_resources:
        mine_key = resource_to_mine(res)
        mine_cfg = CONVERTER_TYPES[mine_key].model_copy(deep=True)
        game_objects[mine_key] = mine_cfg

    # Clusters for each resource
    cluster_specs: list[dict] = []
    for res in chosen_resources:
        cluster_specs.append(
            {
                "name": resource_to_mine(res),
                "count": int(cluster_size),
                "distance": int(cluster_distance),
                "radius": 1,
            }
        )

    # Altar with four directional recipes
    altar = building.assembler_altar.model_copy(deep=True)
    # Pick concrete inputs per category
    sink_input = {chosen_resources[0]: 1}
    cheap_inputs = {chosen_resources[1]: 1}
    expensive_inputs_low = {chosen_resources[0]: 2, chosen_resources[1]: 2}
    expensive_inputs_high = {chosen_resources[0]: 2, chosen_resources[2]: 2}

    altar.recipes = [
        (  # North: sink (no output)
            ["N"],
            RecipeConfig(
                input_resources=sink_input, output_resources={}, cooldown=low_cooldown
            ),
        ),
        (  # South: expensive, low cooldown
            ["S"],
            RecipeConfig(
                input_resources=expensive_inputs_low,
                output_resources={"heart": 1},
                cooldown=low_cooldown,
            ),
        ),
        (  # East: cheap, high cooldown
            ["E"],
            RecipeConfig(
                input_resources=cheap_inputs,
                output_resources={"heart": 1},
                cooldown=high_cooldown,
            ),
        ),
        (  # West: expensive, high cooldown
            ["W"],
            RecipeConfig(
                input_resources=expensive_inputs_high,
                output_resources={"heart": 1},
                cooldown=high_cooldown,
            ),
        ),
    ]
    game_objects["altar"] = altar

    return MettaGridConfig(
        game=GameConfig(
            max_steps=512,
            num_agents=num_agents * num_instances,
            objects=game_objects,
            map_builder=MapGen.Config(
                instances=num_instances,
                instance_map=ForagingMapBuilder.Config(
                    agents=num_agents,
                    width=width,
                    height=height,
                    num_assemblers=num_assemblers,
                    clusters=cluster_specs,
                ),
            ),
            actions=ActionsConfig(
                move=ActionConfig(),
                rotate=ActionConfig(enabled=False),
                get_items=ActionConfig(enabled=False),
                put_items=ActionConfig(enabled=False),
                change_glyph=ChangeGlyphActionConfig(number_of_glyphs=16),
            ),
            agent=AgentConfig(
                rewards=AgentRewards(inventory={"heart": 1}),
                default_resource_limit=2,
                resource_limits={"heart": 15},
            ),
        )
    )


def make_foraging_directional_suite() -> list[SimulationConfig]:
    return [
        SimulationConfig(
            env=make_foraging_directional_env(
                width=10, height=10, low_cooldown=8, high_cooldown=80
            ),
            name="foraging_directional_small",
            suite="in_context_learning",
        ),
        SimulationConfig(
            env=make_foraging_directional_env(
                width=12, height=12, low_cooldown=10, high_cooldown=100
            ),
            name="foraging_directional_medium",
            suite="in_context_learning",
        ),
    ]


def play_foraging_directional() -> PlayTool:
    env = make_foraging_directional_env()
    return PlayTool(
        sim=SimulationConfig(
            env=env,
            name="icl_foraging_directional",
            suite="in_context_learning",
        ),
    )
