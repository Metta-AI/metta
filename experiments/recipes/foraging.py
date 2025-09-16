from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, cast

from metta.cogworks.curriculum.curriculum import (
    CurriculumConfig,
    DiscreteRandomConfig,
)
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
from metta.cogworks.curriculum.task_generator import (
    AnyTaskGeneratorConfig,
    TaskGenerator,
    TaskGeneratorConfig,
    TaskGeneratorSet,
)
from metta.mettagrid.builder import empty_converters
from metta.mettagrid.map_builder.foraging import ForagingMapBuilder
from metta.mettagrid.mapgen.mapgen import MapGen
from metta.mettagrid.mettagrid_config import GameConfig, MettaGridConfig
from metta.rl.loss.loss_config import LossConfig
from metta.rl.trainer_config import EvaluationConfig, TrainerConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.play import PlayTool
from metta.tools.sim import SimTool
from metta.tools.train import TrainTool
from pydantic import Field

# Resource and converter catalogs
RESOURCE_TYPES = [
    "ore_red",
    "ore_blue",
    "ore_green",
    "battery_red",
    "battery_blue",
    "battery_green",
]

CONVERTER_TYPES = {
    "altar": empty_converters.altar,
    "lab": empty_converters.lab,
    "lasery": empty_converters.lasery,
    "factory": empty_converters.factory,
    "temple": empty_converters.temple,
}

MINE_TYPES = {
    "mine_red": empty_converters.mine_red,
    "mine_blue": empty_converters.mine_blue,
    "mine_green": empty_converters.mine_green,
}


@dataclass
class _BuildCfg:
    used_objects: List[str] = field(default_factory=list)
    hub_objects: Dict[str, Any] = field(default_factory=dict)
    outside_objects: Dict[str, int] = field(default_factory=dict)


class ForagingTaskGenerator(TaskGenerator):
    class Config(TaskGeneratorConfig["ForagingTaskGenerator"]):
        """Configuration for ForagingTaskGenerator.

        Produces a hub of heart-creating converters and outside resource deposits.
        """

        # Agents per environment (supports multi-agent via MapGen multi-instances)
        num_agents: int = Field(default=24)

        # Map size per instance
        width: int = Field(default=13)
        height: int = Field(default=13)

        # Hub composition
        num_converters: int = Field(default=3, description="Number of hub converters")
        converter_pool: list[str] = Field(
            default_factory=lambda: list(CONVERTER_TYPES.keys())
        )
        # Each converter consumes exactly one input resource by default
        input_resource_pool: list[str] = Field(
            default_factory=lambda: ["ore_red", "ore_blue", "ore_green"]
        )

        # Cooldowns: either fixed or sampled per converter
        cooldowns: list[int] = Field(default_factory=lambda: [0, 5, 10])

        # Outside deposits
        # Number of deposits per resource (keys must be in RESOURCE_TYPES)
        deposits_per_resource: dict[str, int] = Field(
            default_factory=lambda: {"ore_red": 4, "ore_blue": 4, "ore_green": 4}
        )
        # One-shot vs regenerating sources
        deposit_initial_count: int = Field(default=1)
        deposit_max_conversions: int = Field(
            default=0, description="0 for preload only; -1 for infinite"
        )
        deposit_cooldown: int = Field(default=0)

        # Agent inventory limits
        default_resource_limit: int = Field(
            default=1, description="Carry capacity for non-overridden items"
        )
        heart_limit: int = Field(default=255)
        # Per-item caps; for class-like behavior on ores, set all ore_* to 1
        resource_limits: dict[str, int] = Field(
            default_factory=lambda: {"ore_red": 1, "ore_blue": 1, "ore_green": 1}
        )

        # Map builder options
        hub_layout: str = Field(default="grid")
        hub_box_radius: int = Field(default=2)
        outside_min_radius: int = Field(default=3)
        outside_max_radius: int = Field(default=5)
        border_width: int = Field(default=0)

        # Episode length
        max_steps: int = Field(default=256)

    def __init__(self, config: "ForagingTaskGenerator.Config"):
        super().__init__(config)
        self.config = config

    def _make_hub_converter(self, name: str, input_item: str, cooldown: int) -> Any:
        # Copy prototype and set recipe → heart
        proto = CONVERTER_TYPES[name].model_copy()
        proto.input_resources = {input_item: 1}
        proto.output_resources = {"heart": 1}
        proto.cooldown = int(cooldown)
        proto.max_output = 1  # prevent stacking
        return proto

    def _make_deposit(self, resource_name: str) -> tuple[str, Any]:
        # Use mine prototypes; override output to the requested resource if needed
        # For ores, we have dedicated mines; for batteries, we can use a neutral generator_red as a stub.
        if resource_name in ("ore_red", "ore_blue", "ore_green"):
            obj_name = {
                "ore_red": "mine_red",
                "ore_blue": "mine_blue",
                "ore_green": "mine_green",
            }[resource_name]
            proto = MINE_TYPES[obj_name].model_copy()
            # Ensure the correct output resource
            proto.output_resources = {resource_name: 1}
        else:
            # Fallback: generic converter symbol
            obj_name = "converter"
            proto = empty_converters.lab.model_copy()
            proto.input_resources = {}
            proto.output_resources = {resource_name: 1}

        proto.cooldown = int(self.config.deposit_cooldown)
        proto.initial_resource_count = int(self.config.deposit_initial_count)
        proto.max_conversions = int(self.config.deposit_max_conversions)
        proto.max_output = 5
        return obj_name, proto

    def _build_objects(
        self, rng: random.Random
    ) -> tuple[dict[str, Any], dict[str, int], list[str]]:
        cfg = _BuildCfg()

        # Build hub converters (unique types, placed by base type name)
        available_types = list(self.config.converter_pool)
        rng.shuffle(available_types)
        selected_types = available_types[
            : max(0, min(self.config.num_converters, len(available_types)))
        ]
        available_inputs = list(self.config.input_resource_pool)
        if not available_inputs:
            available_inputs = ["ore_red"]

        for type_name in selected_types:
            input_item = rng.choice(available_inputs)
            cd = rng.choice(self.config.cooldowns) if self.config.cooldowns else 0
            obj = self._make_hub_converter(type_name, input_item, cd)
            cfg.hub_objects[type_name] = obj
            cfg.used_objects.append(type_name)

        # Build outside deposits counts for the map builder (placement)
        outside_counts: dict[str, int] = {}
        for res_name, count in self.config.deposits_per_resource.items():
            if count <= 0:
                continue
            outside_counts[res_name] = outside_counts.get(res_name, 0) + int(count)

        # Assemble GameConfig objects dict
        objects: dict[str, Any] = {"wall": empty_converters.wall}
        # Add hub object prototypes under their base type names
        for type_name, proto in cfg.hub_objects.items():
            objects[type_name] = proto
        # Map builder expects object name -> count for placement
        map_builder_objects: dict[str, int] = {}
        # Hub: place exactly one instance per hub object (by base type)
        for type_name in cfg.hub_objects.keys():
            map_builder_objects[type_name] = 1
        # Outside resources: ensure a prototype and count for the object symbol name
        for res_name, count in outside_counts.items():
            if count > 0:
                obj_name, proto = self._make_deposit(res_name)
                objects[obj_name] = proto
                map_builder_objects[obj_name] = (
                    map_builder_objects.get(obj_name, 0) + count
                )

        hub_names = list(cfg.hub_objects.keys())
        return objects, map_builder_objects, hub_names

    def _generate_task(self, task_id: int, rng: random.Random) -> MettaGridConfig:
        objects, map_builder_objects, hub_names = self._build_objects(rng)

        # Build env with MapGen + ForagingMapBuilder
        map_builder = MapGen.Config(
            # Use num_agents to replicate single-agent instances across the grid
            num_agents=self.config.num_agents,
            instance_map=ForagingMapBuilder.Config(
                width=self.config.width,
                height=self.config.height,
                border_width=self.config.border_width,
                hub_objects={name: map_builder_objects[name] for name in hub_names},
                outside_objects={
                    k: v for k, v in map_builder_objects.items() if k not in hub_names
                },
                hub_layout=self.config.hub_layout,
                hub_box_radius=self.config.hub_box_radius,
                outside_min_radius=self.config.outside_min_radius,
                outside_max_radius=self.config.outside_max_radius,
                center_agent=True,
            ),
        )

        # Ensure required actions are explicitly enabled for interaction
        from metta.mettagrid.mettagrid_config import (
            ActionsConfig,
            ActionConfig,
        )  # local import to avoid cycles

        actions = ActionsConfig(
            move=ActionConfig(enabled=True),
            put_items=ActionConfig(enabled=True),
            get_items=ActionConfig(enabled=True),
        )

        game = GameConfig(
            max_steps=self.config.max_steps,
            num_agents=self.config.num_agents,
            objects=objects,
            actions=actions,
        )

        # Inventory configuration
        game.agent.default_resource_limit = self.config.default_resource_limit
        game.agent.resource_limits = {
            "heart": self.config.heart_limit,
            **self.config.resource_limits,
        }
        game.agent.rewards.inventory = {"heart": 1.0}

        env = MettaGridConfig(
            label="foraging",
            game=game,
        )
        env.game.map_builder = map_builder
        return env


def make_mettagrid() -> MettaGridConfig:
    task_generator_cfg = ForagingTaskGenerator.Config()
    task_generator = ForagingTaskGenerator(task_generator_cfg)
    return task_generator.get_task(0)


def make_curriculum(
    enable_detailed_slice_logging: bool = False,
    algorithm_config: Optional[LearningProgressConfig | DiscreteRandomConfig] = None,
) -> CurriculumConfig:
    if algorithm_config is None:
        algorithm_config = LearningProgressConfig(
            use_bidirectional=True,
            ema_timescale=0.001,
            exploration_bonus=0.1,
            max_memory_tasks=500,
            max_slice_axes=3,
            progress_smoothing=0.05,
            enable_detailed_slice_logging=enable_detailed_slice_logging,
        )

    # Simple sweep: converters, distances, cooldowns, carry capacity
    gens: list[AnyTaskGeneratorConfig] = []
    for num_converters in [2, 3, 4]:
        for carry in [1, 2]:
            for cd in [[0], [5], [10], [0, 5, 10]]:
                for rmin, rmax in [(2, 3), (3, 6)]:
                    label = f"n{num_converters}_carry{carry}_cd{'-'.join(map(str, cd))}_r{rmin}-{rmax}"
                    gens.append(
                        cast(
                            AnyTaskGeneratorConfig,
                            ForagingTaskGenerator.Config(
                                label=label,
                                num_converters=num_converters,
                                default_resource_limit=carry,
                                cooldowns=cd,
                                outside_min_radius=rmin,
                                outside_max_radius=rmax,
                            ),
                        )
                    )

    task_set = TaskGeneratorSet.Config(task_generators=gens, weights=[1.0] * len(gens))
    return CurriculumConfig(task_generator=task_set, algorithm_config=algorithm_config)


def train(
    curriculum: Optional[CurriculumConfig] = None,
) -> TrainTool:
    # Local import to avoid circular import
    from experiments.evals.foraging import make_foraging_eval_suite

    trainer_cfg = TrainerConfig(
        losses=LossConfig(),
        curriculum=curriculum or make_curriculum(),
        evaluation=EvaluationConfig(simulations=make_foraging_eval_suite()),
    )
    return TrainTool(trainer=trainer_cfg)


def play(env: Optional[MettaGridConfig] = None) -> PlayTool:
    eval_env = env or make_mettagrid()
    return PlayTool(
        sim=SimulationConfig(
            env=eval_env,
            name="foraging",
        ),
    )


def evaluate(simulations: Optional[Sequence[SimulationConfig]] = None) -> SimTool:
    # Local import to avoid circular import
    from experiments.evals.foraging import make_foraging_eval_suite

    sims: Sequence[SimulationConfig] = (
        simulations if simulations is not None else make_foraging_eval_suite()
    )
    return SimTool(simulations=sims)
