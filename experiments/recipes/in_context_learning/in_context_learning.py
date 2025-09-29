import random
from dataclasses import dataclass, field
from typing import Any, Dict, List

from metta.cogworks.curriculum.task_generator import TaskGenerator, TaskGeneratorConfig
from pydantic import Field
from mettagrid.builder import building, empty_converters
from mettagrid.config.mettagrid_config import (
    Position,
    RecipeConfig,
    MettaGridConfig,
)
from metta.cogworks.curriculum.curriculum import CurriculumConfig
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
from metta.rl.loss import LossConfig
from metta.rl.trainer_config import TrainerConfig
from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig
from metta.tools.train import TrainTool
from metta.agent.policies.fast_lstm_reset import FastLSTMResetConfig
from typing import Callable
from metta.sim.simulation_config import SimulationConfig
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from mettagrid.builder.envs import make_icl_with_numpy

CONVERTER_TYPES = {
    "mine_red": empty_converters.mine_red,
    "mine_blue": empty_converters.mine_blue,
    "mine_green": empty_converters.mine_green,
    "generator_red": empty_converters.generator_red,
    "generator_blue": empty_converters.generator_blue,
    "generator_green": empty_converters.generator_green,
    "altar": empty_converters.altar,
    "lab": empty_converters.lab,
    "lasery": empty_converters.lasery,
    "factory": empty_converters.factory,
    "temple": empty_converters.temple,
    "armory": empty_converters.armory,
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

ASSEMBLER_TYPES = {
    "generator_red": building.assembler_generator_red,
    "generator_blue": building.assembler_generator_blue,
    "generator_green": building.assembler_generator_green,
    "mine_red": building.assembler_mine_red,
    "mine_blue": building.assembler_mine_blue,
    "mine_green": building.assembler_mine_green,
    "altar": building.assembler_altar,
}

size_ranges = {
    "tiny": (5, 10),  # 2 objects 2 agents max for assemblers
    "small": (10, 20),  # 9 objects, 5 agents max
    "medium": (20, 30),
    "large": (30, 40),
    "xlarge": (40, 50),
}

num_agents_to_positions = {
    1: [["N"], ["S"], ["E"], ["W"], ["Any"]],
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

room_size_templates = {
    "tiny": {
        "room_size": ["tiny"],
        "num_agents": [2],
        "num_objects": [2],
        "terrain": ["no-terrain"],
    },
    "small": {
        "room_size": ["small"],
        "num_agents": [12],
        "num_objects": [5, 10],
        "terrain": ["no-terrain", "sparse"],
    },
    "medium": {
        "room_size": ["medium"],
        "num_agents": [12],
        "num_objects": [10, 20, 30],
        "terrain": ["sparse"],
    },
    "large": {
        "room_size": ["large"],
        "num_agents": [12],
        "num_objects": [10, 20, 30, 50],
        "terrain": ["no-terrain", "sparse", "balanced"],
    },
    "xlarge": {
        "room_size": ["xlarge"],
        "num_agents": [2, 6, 12],
        "num_objects": [10, 20, 30, 50],
        "terrain": ["no-terrain", "sparse", "balanced", "dense"],
    },
}


@dataclass
class _BuildCfg:
    used_objects: List[str] = field(default_factory=list)
    all_input_resources: set[str] = field(default_factory=set)
    all_output_resources: set[str] = field(default_factory=set)
    converters: List[str] = field(default_factory=list)
    game_objects: Dict[str, Any] = field(default_factory=dict)
    map_builder_objects: Dict[str, int] = field(default_factory=dict)


def calculate_avg_hop(room_size: str) -> float:
    return (size_ranges[room_size][0] + size_ranges[room_size][1]) / 2


class ICLTaskGenerator(TaskGenerator):
    """
    Shared superclass for Ordered/Unordered chain generators.
    Subclasses should implement `_make_env_cfg(...)`.
    """

    class Config(TaskGeneratorConfig["ICLTaskGenerator"]):
        # Common knobs
        num_agents: list[int] = Field(
            default=[1],
            description="Number of agents to include.",
        )

        num_resources: list[int] = Field(
            default_factory=list,
            description="Number of base/intermediate resources to include.",
        )
        num_converters: list[int] = Field(
            default_factory=list,
            description="Ordered: number of sinks; Unordered: number of heart-producing recipe converters.",
        )
        room_sizes: list[str] = Field(
            default=["small"], description="Room sizes to sample from."
        )
        map_dir: str | None = Field(
            default=None,
            description="Directory for pre-generated maps (None to build procedurally).",
        )

        # Unordered-only (ignored by Ordered subclasses)
        max_recipe_inputs: list[int] = Field(
            default=[1],
            description="Max inputs per recipe converter (sampled per env).",
        )
        # assembler specific
        positions: list[list[Position]] = Field(
            default=[["Any"]],
            description="Positions for assemblers.",
        )

    def __init__(self, config: "ICLTaskGenerator.Config"):
        super().__init__(config)
        self.resource_types = RESOURCE_TYPES.copy()
        self.converter_types = CONVERTER_TYPES.copy()
        self.assembler_types = ASSEMBLER_TYPES.copy()
        self.config = config

    # -------- helpers shared by ordered/unordered --------

    def _choose_converter_name(
        self, pool: Dict[str, Any], used: set[str], rng: random.Random
    ) -> str:
        """Pick an unused converter prefab name from the pool."""
        choices = [name for name in pool.keys() if name not in used]
        if not choices:
            raise ValueError("No available converter names left to choose from.")
        return str(rng.choice(choices))

    def _add_converter(
        self,
        input_resources: dict[str, int],
        output_resources: dict[str, int],
        cfg: _BuildCfg,
        rng: random.Random,
        cooldown: int = 10,
    ):
        converter_name = self._choose_converter_name(
            self.converter_types, set(cfg.used_objects), rng
        )
        cfg.used_objects.append(converter_name)

        converter = self.converter_types[converter_name].copy()

        converter.output_resources = output_resources
        converter.input_resources = input_resources
        converter.cooldown = int(cooldown)
        cfg.all_output_resources.update(output_resources)
        cfg.all_input_resources.update(input_resources)

        cfg.game_objects[converter_name] = converter
        cfg.map_builder_objects[converter_name] = 1

        return converter_name

    def _add_assembler(
        self,
        input_resources: dict[str, int],
        output_resources: dict[str, int],
        position,
        cfg: _BuildCfg,
        rng: random.Random,
        cooldown: int = 10,
        assembler_name: str | None = None,
        replacement=False,
    ):
        if replacement:
            assembler_name = (
                rng.choice(list(self.assembler_types.keys()))
                if assembler_name is None
                else assembler_name
            )
        else:
            assembler_name = (
                self._choose_converter_name(
                    self.assembler_types, set(cfg.used_objects), rng
                )
                if assembler_name is None
                else assembler_name
            )
        cfg.used_objects.append(assembler_name)
        assembler = self.assembler_types[assembler_name].copy()

        recipe = (
            position,
            RecipeConfig(
                input_resources=input_resources,
                output_resources=output_resources,
                cooldown=int(cooldown),
            ),
        )

        assembler.recipes = [recipe]
        cfg.game_objects[assembler_name] = assembler
        if assembler_name in cfg.map_builder_objects:
            cfg.map_builder_objects[assembler_name] += 1
        else:
            cfg.map_builder_objects[assembler_name] = 1

    def _add_chest(self,
        position,
        cfg: _BuildCfg,
        rng: random.Random,
        cooldown: int = 10,
        chest_name: str | None = None,
    ):
        chest = building.chest_heart

        chest.deposit_positions = position
        chest.withdrawal_positions = None
        cfg.game_objects["chest"] = chest
        if chest_name in cfg.map_builder_objects:
            cfg.map_builder_objects[chest_name] += 1
        else:
            cfg.map_builder_objects[chest_name] = 1


    def _get_width_and_height(self, room_size: str, rng: random.Random):
        lo, hi = size_ranges[room_size]
        width = rng.randint(lo, hi)
        height = rng.randint(lo, hi)
        return width, height

    def _set_width_and_height(self, room_size, num_agents, num_objects, rng):
        """Set the width and height of the environment to be at least the minimum area required for the number of agents, altars, and generators."""
        width, height = self._get_width_and_height(room_size, rng)
        area = width * height
        minimum_area = (num_agents + num_objects) * 2
        if area < minimum_area:
            width, height = minimum_area // 2, minimum_area // 2
        return width, height

    def calculate_max_steps(
        self, num_resources: int, num_converters: int, width: int, height: int
    ) -> int:
        raise NotImplementedError("Subclasses must implement calculate_max_steps(...)")

    def _setup_task(self, rng: random.Random):
        """
        Sample the high-level task spec that both Ordered and Unordered builders use.
        Returns:
            resources: List[str]
            num_converters: int
            room_size: str
            width: int
            height: int
            max_recipe_inputs: int
        """
        cfg = self.config

        # counts
        num_agents = rng.choice(cfg.num_agents)
        num_resources = rng.choice(cfg.num_resources)
        num_converters = rng.choice(cfg.num_converters)

        if num_resources == 0:
            resources = []
        else:
            num_resources = max(1, min(num_resources, len(self.resource_types)))
            resources = rng.sample(self.resource_types, num_resources)

        room_size = rng.choice(cfg.room_sizes)
        terrain_density = (
            rng.choice(room_size_templates[room_size]["terrain"]) or "no-terrain"
        )

        width, height = self._set_width_and_height(
            room_size, num_agents, num_resources + num_converters, rng
        )

        recipe_position = rng.choice(
            [p for p in self.config.positions if len(p) <= num_agents]
        )
        max_steps = self.calculate_max_steps(
            num_resources, num_converters, width, height
        )
        return (
            num_agents,
            resources,
            num_converters,
            room_size,
            terrain_density,
            width,
            height,
            max_steps,
            recipe_position,
        )

    def load_from_numpy(
        self,
        num_agents,
        max_steps,
        game_objects,
        map_builder_objects,
        dir,
        rng,
        num_instances=24,
    ) -> MettaGridConfig:
        from metta.map.terrain_from_numpy import InContextLearningFromNumpy

        env = make_icl_with_numpy(
            num_agents=num_agents,
            num_instances=num_instances,
            max_steps=max_steps,
            game_objects=game_objects,
            instance_map=InContextLearningFromNumpy.Config(
                agents=num_agents,
                dir=dir,
                objects=map_builder_objects,
                rng=rng,
            ),
        )
        return env

    # Subclasses must implement this to actually build MettaGridConfig:
    def _make_env_cfg(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement _make_env_cfg(...)")


class LPParams:
    def __init__(
        self,
        ema_timescale: float = 0.001,
        exploration_bonus: float = 0.15,
        max_memory_tasks: int = 1000,
        max_slice_axes: int = 3,
        progress_smoothing: float = 0.15,
        enable_detailed_slice_logging: bool = False,
        num_active_tasks: int = 1000,
        rand_task_rate: float = 0.25,
    ):
        self.ema_timescale = ema_timescale
        self.exploration_bonus = exploration_bonus
        self.max_memory_tasks = max_memory_tasks
        self.max_slice_axes = max_slice_axes
        self.progress_smoothing = progress_smoothing
        self.enable_detailed_slice_logging = enable_detailed_slice_logging
        self.num_active_tasks = num_active_tasks
        self.rand_task_rate = rand_task_rate


def setup_curriculum(task_generator_cfg, lp_params: LPParams) -> CurriculumConfig:
    algorithm_config = LearningProgressConfig(**lp_params.__dict__)
    return CurriculumConfig(
        task_generator=task_generator_cfg,
        algorithm_config=algorithm_config,
    )


def train_icl(
    task_generator_cfg,
    evaluator_fn: Callable[[], list[SimulationConfig]],
    lp_params: LPParams = LPParams(),
) -> TrainTool:
    curriculum = setup_curriculum(task_generator_cfg, lp_params)
    trainer_cfg = TrainerConfig(
        losses=LossConfig(),
    )
    policy_config = FastLSTMResetConfig()
    # TODO change to VIT DEFAULT LSTM CONFIG
    return TrainTool(
        trainer=trainer_cfg,
        training_env=TrainingEnvironmentConfig(curriculum=curriculum),
        policy_architecture=policy_config,
        evaluator=EvaluatorConfig(
            simulations=evaluator_fn(),
            evaluate_remote=False,
            evaluate_local=True,
        ),
        stats_server_uri="https://api.observatory.softmax-research.net",
    )


def make_mettagrid(task_generator) -> MettaGridConfig:
    return task_generator.get_task(random.randint(0, 1000000))


def play_icl(task_generator) -> PlayTool:
    return PlayTool(
        sim=SimulationConfig(
            env=make_mettagrid(task_generator),
            suite="in_context_learning",
            name="eval",
        ),
    )


def replay_icl(task_generator, policy_uri: str) -> ReplayTool:
    return ReplayTool(
        sim=SimulationConfig(
            env=make_mettagrid(task_generator),
            suite="in_context_learning",
            name="eval",
        ),
        policy_uri=policy_uri,
    )
