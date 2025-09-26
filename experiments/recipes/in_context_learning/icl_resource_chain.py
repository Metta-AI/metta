import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from metta.cogworks.curriculum.task_generator import TaskGenerator, TaskGeneratorConfig
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


@dataclass
class _BuildCfg:
    used_objects: List[str] = field(default_factory=list)
    all_input_resources: List[str] = field(default_factory=list)
    converters: List[str] = field(default_factory=list)
    game_objects: Dict[str, Any] = field(default_factory=dict)
    map_builder_objects: Dict[str, int] = field(default_factory=dict)

    # unordered chain variables
    sources: List[str] = field(default_factory=list)


def calculate_avg_hop(room_size: str) -> float:
    return (size_ranges[room_size][0] + size_ranges[room_size][1]) / 2


class ICLTaskGenerator(TaskGenerator):
    """
    Shared superclass for Ordered/Unordered chain generators.
    Subclasses should implement `_make_env_cfg(...)`.
    """

    class Config(TaskGeneratorConfig["ICLTaskGenerator"]):
        # Common knobs
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
        obstacle_types: list[str] = Field(
            default_factory=list, description="Terrain obstacle shapes."
        )
        densities: list[str] = Field(
            default_factory=list, description="Terrain densities."
        )
        map_dir: str | None = Field(
            default=None,
            description="Directory for pre-generated maps (None to build procedurally).",
        )

        # Unordered-only (ignored by Ordered subclasses)
        max_recipe_inputs: Optional[list[int]] = Field(
            default=None,
            description="Max inputs per recipe converter (sampled per env).",
        )
        source_initial_resource_count: Optional[int] = Field(
            default=None, description="Initial stock per source (None = infinite)."
        )
        source_max_conversions: Optional[int] = Field(
            default=None,
            description="Max regenerations per source (0 for no regen; None for default).",
        )
        source_cooldown: int = Field(
            default=25, description="Source regeneration cooldown (if used)."
        )

    def __init__(self, config: "ICLTaskGenerator.Config"):
        super().__init__(config)
        self.resource_types = RESOURCE_TYPES.copy()
        self.converter_types = CONVERTER_TYPES.copy()
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

    def _setup_task(self, rng: random.Random):
        """
        Sample the high-level task spec that both Ordered and Unordered builders use.
        Returns:
            resources: List[str]
            num_converters: int
            room_size: str
            obstacle_type: Optional[str]
            density: Optional[str]
            width: int
            height: int
            max_recipe_inputs: Optional[int]  # (unordered only; pass-thru for ordered)
        """
        cfg = self.config

        # counts
        num_resources = rng.choice(cfg.num_resources)
        num_converters = rng.choice(cfg.num_converters)

        # clamp and draw resource set
        num_resources = max(1, min(num_resources, len(self.resource_types)))
        resources = rng.sample(self.resource_types, num_resources)

        # geometry & terrain
        room_size = rng.choice(cfg.room_sizes)
        obstacle_type = rng.choice(cfg.obstacle_types) if cfg.obstacle_types else None
        density = rng.choice(cfg.densities) if cfg.densities else None

        lo, hi = size_ranges[room_size]
        width = rng.randint(lo, hi)
        height = rng.randint(lo, hi)

        # unordered-only param (safe to ignore downstream if None)
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
