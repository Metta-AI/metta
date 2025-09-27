import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from softmax.cogworks.curriculum.task_generator import TaskGenerator, TaskGeneratorConfig
from mettagrid.builder import empty_converters
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

size_ranges: dict[str, tuple[int, int]] = {
    "tiny": (5, 8),
    "small": (8, 12),
    "medium": (12, 16),
    "large": (16, 25),
}


def calculate_avg_hop(room_size: str) -> float:
    return (size_ranges[room_size][0] + size_ranges[room_size][1]) / 2


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
        """Configuration for ICLTaskGenerator matching ordered_chains refactor."""

        num_resources: list[int] = Field(
            default_factory=list,
            description="Number of resources",
        )
        num_converters: list[int] = Field(
            default_factory=list,
            description="Number of converters, which are sinks for ordered chains",
        )
        room_sizes: list[str] = Field(
            default=["small"], description="Room size to sample from"
        )
        obstacle_types: list[str] = Field(
            default=[], description="Obstacle types to sample from"
        )
        densities: list[str] = Field(default=[], description="Density to sample from")
        # obstacle_complexity
        max_steps: int = Field(default=512, description="Episode length")
        map_dir: str | None = Field(
            default=None,
            description="Directory to load environments from",
        )
        max_recipe_inputs: Optional[list[int]] = Field(
            default=None,
            description="Maximum resources per converter for unordered chains",
        )
        source_initial_resource_count: Optional[int] = Field(
            default=None, description="Initial resource count per source"
        )
        source_max_conversions: Optional[int] = Field(
            default=None,
            description="Max conversions per source (0 for no regeneration)",
        )
        source_cooldown: int = Field(
            default=25, description="Cooldown for source regeneration"
        )
        non_reusable_resources: list[str] = Field(
            default_factory=list,
            description="List of resource types that are not reusable",
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
        # Safely determine counts for both ordered and unordered chains
        num_resources = rng.choice(cfg.num_resources)
        num_converters = rng.choice(cfg.num_converters)
        # Clamp to available resource types to avoid ValueError in sampling
        resources = rng.sample(self.resource_types, num_resources)
        room_size = rng.choice(cfg.room_sizes)
        obstacle_type = (
            rng.choice(cfg.obstacle_types) if len(cfg.obstacle_types) > 0 else None
        )
        density = rng.choice(cfg.densities) if len(cfg.densities) > 0 else None

        size_range = size_ranges[room_size]

        width, height = (
            rng.randint(size_range[0], size_range[1]),
            rng.randint(size_range[0], size_range[1]),
        )

        max_recipe_inputs = (
            rng.choice(cfg.max_recipe_inputs) if cfg.max_recipe_inputs else None
        )

        return (
            resources,
            num_converters,
            room_size,
            obstacle_type,
            density,
            width,
            height,
            max_recipe_inputs,
        )


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
