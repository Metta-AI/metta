import random
from dataclasses import dataclass, field
from typing import Any, Dict, List

from metta.cogworks.curriculum.task_generator import TaskGenerator, TaskGeneratorConfig
from metta.mettagrid.builder import empty_converters
from pydantic import Field

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
    used_objects: List[str] = field(default_factory=list)
    all_input_resources: List[str] = field(default_factory=list)
    converters: List[str] = field(default_factory=list)
    game_objects: Dict[str, Any] = field(default_factory=dict)
    map_builder_objects: Dict[str, int] = field(default_factory=dict)

    # unordered chain variables
    sources: List[str] = field(default_factory=list)


class ICLTaskGenerator(TaskGenerator):
    class Config(TaskGeneratorConfig["ICLTaskGenerator"]):
        """Configuration for UnorderedChainTaskGenerator."""

        num_resources: list[int] = Field(
            default_factory=list, description="Number of unique resources"
        )
        num_sinks: list[int] = Field(
            default_factory=list,
            description="Number of converters that take as input some resources",
        )
        room_sizes: list[str] = Field(
            default=["small"], description="Room size to sample from"
        )
        obstacle_types: list[str] = Field(
            default=[], description="Obstacle types to sample from"
        )
        densities: list[str] = Field(default=[], description="Density to sample from")
        # obstacle_complexity
        max_steps: int = Field(default=256, description="Episode length")

        # For source/mine regeneration behavior (None = don't override prototype)
        source_cooldown: int | None = Field(
            default=None,
            description="Ticks between source regenerations (None = prototype default)",
        )
        source_initial_resource_count: int | None = Field(
            default=None,
            description="Initial stock available at source (None = prototype default)",
        )
        source_max_conversions: int | None = Field(
            default=None,
            description="Max conversions before depletion (-1 = infinite, 0 = preload only, None = prototype)",
        )

        # Resources that cannot be used multiple times within a single recipe (no duplicates)
        # This is automatically derived from sources with max_conversions=1 or initial_resource_count=1
        non_reusable_resources: list[str] = Field(
            default_factory=list,
            description="Resource types that cannot repeat within one converter recipe (auto-derived)",
        )

    def __init__(self, config: "ICLTaskGenerator.Config"):
        super().__init__(config)
        self.resource_types = RESOURCE_TYPES.copy()
        self.converter_types = CONVERTER_TYPES.copy()
        self.config = config

    def _choose_converter_name(
        self, pool: Dict[str, Any], used: set[str], rng: random.Random
    ) -> str:
        choices = [name for name in pool.keys() if name not in used]
        if not choices:
            raise ValueError("No available converter names left to choose from.")
        return str(rng.choice(choices))

    def _make_env_cfg(self, cfg: _BuildCfg, rng: random.Random):
        pass

    def _setup_task(self, rng: random.Random):
        cfg = self.config
        num_resources = rng.choice(cfg.num_resources)
        num_sinks = rng.choice(cfg.num_sinks)
        resources = rng.sample(self.resource_types, num_resources)
        room_size = rng.choice(cfg.room_sizes)
        obstacle_type = (
            rng.choice(cfg.obstacle_types) if len(cfg.obstacle_types) > 0 else None
        )
        density = rng.choice(cfg.densities) if len(cfg.densities) > 0 else None

        size_range = (
            (8, 12)
            if room_size == "medium"
            else (12, 15)
            if room_size == "large"
            else (5, 8)
        )

        width, height = (
            rng.randint(size_range[0], size_range[1]),
            rng.randint(size_range[0], size_range[1]),
        )

        return resources, num_sinks, room_size, obstacle_type, density, width, height
