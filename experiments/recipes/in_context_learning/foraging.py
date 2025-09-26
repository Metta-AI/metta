"""
Foraging-focused tasks and utilities (separated from assembler experiments).

Includes:
- Foraging map builders and env factories
- Directional altar variants
- Biased foraging task generator and curricula
"""

import random
from math import comb
from typing import Any, Dict, Optional

from metta.cogworks.curriculum.curriculum import CurriculumConfig
from metta.cogworks.curriculum.learning_progress_algorithm import (
    LearningProgressConfig,
)
from metta.rl.trainer_config import LossConfig, TrainerConfig
from metta.rl.training import EvaluatorConfig
from metta.rl.training.training_environment import TrainingEnvironmentConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.play import PlayTool
from metta.tools.sim import SimTool
from metta.tools.train import TrainTool
from mettagrid.builder import building, empty_converters
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
from mettagrid.map_builder.assembler_map_builder import RegionAssemblerMapBuilder
from mettagrid.mapgen.mapgen import MapGen
from pydantic import Field

from experiments.recipes.in_context_learning.icl_resource_chain import ICLTaskGenerator

# Converters with assembler semantics used in foraging tasks
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

# -------- Biased foraging generator --------


class BiasedForagingTaskGenerator(ICLTaskGenerator):
    """Task generator for biased foraging with small/medium/large/extra_large maps."""

    class Config(ICLTaskGenerator.Config):
        num_agents: list[int] = Field(default=[1])
        num_assemblers: list[int] = Field(default=[1])
        max_steps: int = 512
        map_sizes: dict[str, dict] = Field(
            default={
                "small": {"width": 10, "height": 10, "resource_count": 2},
                "medium": {"width": 16, "height": 16, "resource_count": 3},
                "large": {"width": 32, "height": 32, "resource_count": 4},
                "extra_large": {"width": 64, "height": 64, "resource_count": 5},
            }
        )
        size_weights: Optional[list[float]] = Field(default=None)
        altar_cooldown: int = Field(default=60)
        recipe_mode: list[str] = Field(
            default=["simple"]
        )  # "simple" | "directional" | "unordered_chain"
        recipe_mode_weights: Optional[list[float]] = Field(default=None)
        max_recipe_inputs: Optional[list[int]] = Field(default=[1, 2, 3])
        # Number of distinct resource types to include in a map
        resource_type_counts: list[int] = Field(default=[4])
        # Regions to sample for resource placement (used when randomize_regions=True)
        regions: list[str] = Field(
            default=[
                "north",
                "south",
                "east",
                "west",
                "northeast",
                "northwest",
                "southeast",
                "southwest",
            ]
        )
        # Randomize mapping from resource type -> region each task
        randomize_regions: bool = Field(default=True)
        # Number of regional clusters to sample per task (controls how many distinct regions used)
        cluster_counts: list[int] = Field(default=[1, 2, 3, 4])
        non_reusable_resources: list[str] = Field(default=[])
        resource_configs: list[dict] = Field(
            default=[
                {"resource": "ore_red", "region": "north"},
                {"resource": "ore_blue", "region": "south"},
                {"resource": "ore_green", "region": "east"},
                {"resource": "battery_red", "region": "west"},
                {"resource": "battery_blue", "region": "north"},
                {"resource": "battery_green", "region": "south"},
            ]
        )
        resource_positions: Optional[list[dict]] = Field(default=None)
        separation_modes: list[str] = Field(default=["strict", "soft"])
        separation_weights: Optional[list[float]] = Field(default=None)
        soft_mode_bias: float = Field(default=0.75)
        altar_positions: list[list[Position]] = Field(
            default=[["Any"], ["N"], ["S"], ["E"], ["W"], ["N", "S"], ["E", "W"]]
        )
        direction_recipes: list[list[Position]] = Field(
            default=[["N"], ["S"], ["E"], ["W"]]
        )

    def __init__(self, config: "BiasedForagingTaskGenerator.Config"):
        super().__init__(config)
        self.config = config

    def _generate_task(self, task_id: int, rng: random.Random) -> MettaGridConfig:
        num_agents = rng.choice(self.config.num_agents)
        if 24 % num_agents != 0:
            raise ValueError(
                f"Number of agents ({num_agents}) must be a divisor of 24."
            )
        num_instances = 24 // num_agents

        size_names = list(self.config.map_sizes.keys())
        if isinstance(self.config.size_weights, list) and len(
            self.config.size_weights
        ) == len(size_names):
            size_weights = self.config.size_weights
        else:
            size_weights = None
        size_name = rng.choices(size_names, weights=size_weights)[0]
        size_config = self.config.map_sizes[size_name]

        width = size_config["width"]
        height = size_config["height"]
        base_resource_count = size_config["resource_count"]

        num_assemblers = rng.choice(self.config.num_assemblers)

        base_configs = list(self.config.resource_configs)
        rng.shuffle(base_configs)
        # Determine how many distinct resource types to include
        try:
            k_types = int(rng.choice(self.config.resource_type_counts))
        except Exception:
            k_types = 4
        k_types = max(1, min(k_types, len(base_configs)))

        selected_base = base_configs[:k_types]

        # Optionally randomize resource->region mapping and cluster regions
        if self.config.randomize_regions:
            # Choose how many distinct regions to use as clusters
            try:
                k_clusters = int(rng.choice(self.config.cluster_counts))
            except Exception:
                k_clusters = min(4, len(self.config.regions))
            k_clusters = max(1, min(k_clusters, len(self.config.regions)))
            region_pool = list(self.config.regions)
            rng.shuffle(region_pool)
            chosen_regions = region_pool[:k_clusters]

            # Assign each resource a region sampled from chosen clusters
            selected_configs = []
            for cfg in selected_base:
                region = rng.choice(chosen_regions)
                selected_configs.append({"resource": cfg["resource"], "region": region})
        else:
            selected_configs = selected_base

        if isinstance(self.config.separation_weights, list) and len(
            self.config.separation_weights
        ) == len(self.config.separation_modes):
            sep_weights = self.config.separation_weights
        else:
            sep_weights = None
        separation_mode = rng.choices(
            self.config.separation_modes, weights=sep_weights
        )[0]

        resource_regions = []
        chosen_resources = []
        for cfg in selected_configs:
            resource = cfg["resource"]
            region = cfg["region"]
            chosen_resources.append(resource)

            resource_type, color = resource.split("_")
            if resource_type == "ore":
                converter_name = f"mine_{color}"
            elif resource_type == "battery":
                converter_name = f"generator_{color}"
            else:
                converter_name = f"mine_{color}"

            resource_regions.append(
                {"name": converter_name, "count": base_resource_count, "region": region}
            )

        game_objects: Dict[str, Any] = {"wall": empty_converters.wall}
        for spec in resource_regions:
            converter_key = spec["name"]
            if converter_key in CONVERTER_TYPES:
                converter_cfg = CONVERTER_TYPES[converter_key].model_copy(deep=True)
                game_objects[converter_key] = converter_cfg

        altar = building.assembler_altar.model_copy(deep=True)

        def _sample_composition(total: int, parts: int) -> list[int]:
            if parts <= 0:
                return []
            if total == 0:
                return [0] * parts
            if parts == 1:
                return [total]
            bars = sorted(rng.sample(range(total + parts - 1), parts - 1))
            prev = -1
            counts: list[int] = []
            for b in bars + [total + parts - 1]:
                counts.append(b - prev - 1)
                prev = b
            return counts

        def _weighted_choice(weights: list[int]) -> int:
            total = sum(weights)
            if total == 0:
                return len(weights) - 1
            cumsum: list[int] = []
            acc = 0
            for w in weights:
                acc += w
                cumsum.append(acc)
            x = rng.random() * total
            lo, hi = 0, len(cumsum)
            while lo < hi:
                mid = (lo + hi) // 2
                if x <= cumsum[mid]:
                    hi = mid
                else:
                    lo = mid + 1
            return lo

        if self.config.recipe_mode_weights and len(
            self.config.recipe_mode_weights
        ) == len(self.config.recipe_mode):
            recipe_mode = rng.choices(
                self.config.recipe_mode, weights=self.config.recipe_mode_weights
            )[0]
        else:
            recipe_mode = rng.choice(self.config.recipe_mode)
        max_inputs_cap = None
        if self.config.max_recipe_inputs:
            max_inputs_cap = rng.choice(self.config.max_recipe_inputs)

        recipes: list[tuple[list[Position], RecipeConfig]] = []
        if recipe_mode == "unordered_chain":
            non_reusable_set = set(self.config.non_reusable_resources)
            available = [r for r in chosen_resources]
            reusable = [r for r in available if r not in non_reusable_set]
            unique_non_reusable = list({r for r in available if r in non_reusable_set})

            L = max(1, max_inputs_cap or 2)
            L = min(L, max(1, len(available)))

            if len(non_reusable_set.intersection(available)) == 0:
                if len(reusable) == 0:
                    recipe_resources_counts: dict[str, int] = {rng.choice(available): 1}
                else:
                    counts = _sample_composition(L, len(reusable))
                    recipe_resources_counts = {
                        typ: c for typ, c in zip(reusable, counts) if c > 0
                    }
            else:
                nR = len(reusable)
                nNR = len(unique_non_reusable)
                max_m = min(L, nNR)
                weights: list[int] = []
                for m in range(0, max_m + 1):
                    r = L - m
                    if r < 0 or (nR == 0 and r > 0):
                        weights.append(0)
                        continue
                    ways_nr = comb(nNR, m)
                    ways_r = (
                        1
                        if r == 0 and nR >= 0
                        else (comb(r + nR - 1, nR - 1) if nR > 0 else 0)
                    )
                    weights.append(ways_nr * ways_r)
                m = _weighted_choice(weights)
                r = max(0, L - m)
                chosen_nr = rng.sample(unique_non_reusable, m) if m > 0 else []
                counts = _sample_composition(r, nR) if nR > 0 else []
                recipe_resources_counts = {nr: 1 for nr in chosen_nr}
                for typ, c in zip(reusable, counts):
                    if c > 0:
                        recipe_resources_counts[typ] = (
                            recipe_resources_counts.get(typ, 0) + c
                        )

            recipes = [
                (
                    ["Any"],
                    RecipeConfig(
                        input_resources=recipe_resources_counts,
                        output_resources={"heart": 1},
                        cooldown=self.config.altar_cooldown,
                    ),
                )
            ]
        elif recipe_mode == "directional":
            k_cap = max_inputs_cap if max_inputs_cap is not None else 3
            recipes = []

            # Require 2 sides if 2+ agents; otherwise 1
            base_dirs: list[Position] = []
            for patt in self.config.direction_recipes:
                for d in patt:
                    if d not in base_dirs:
                        base_dirs.append(d)
            required_positions = 2 if num_agents >= 2 else 1
            if required_positions <= 1:
                pattern_list = list(self.config.direction_recipes)
            else:
                pattern_list: list[list[Position]] = []
                for i in range(len(base_dirs)):
                    for j in range(i + 1, len(base_dirs)):
                        pattern_list.append([base_dirs[i], base_dirs[j]])

            for patt in pattern_list:
                k = min(k_cap, max(1, len(chosen_resources)))
                inputs = rng.sample(chosen_resources, k=min(k, len(chosen_resources)))
                recipes.append(
                    (
                        patt,
                        RecipeConfig(
                            input_resources={r: 1 for r in inputs},
                            output_resources={"heart": 1},
                            cooldown=self.config.altar_cooldown,
                        ),
                    )
                )
        else:
            if size_name == "small":
                k = 2
            elif size_name == "medium":
                k = rng.choice([2, 3])
            else:
                k = min(3, len(chosen_resources))
            if max_inputs_cap is not None:
                k = min(k, max_inputs_cap)
            # Ensure we never sample more inputs than available resource types
            k = min(k, len(chosen_resources))
            recipe_resources = rng.sample(chosen_resources, k=k)
            recipes = [
                (
                    ["Any"],
                    RecipeConfig(
                        input_resources={r: 1 for r in recipe_resources},
                        output_resources={"heart": 1},
                        cooldown=self.config.altar_cooldown,
                    ),
                )
            ]

        altar.recipes = recipes
        game_objects["altar"] = altar

        env_cfg = MettaGridConfig(
            game=GameConfig(
                max_steps=self.config.max_steps,
                num_agents=num_agents * num_instances,
                objects=game_objects,
                map_builder=MapGen.Config(
                    instances=num_instances,
                    instance_map=RegionAssemblerMapBuilder.Config(
                        agents=num_agents,
                        width=width,
                        height=height,
                        border_width=1,
                        num_assemblers=num_assemblers,
                        resource_regions=resource_regions,
                        separation_mode=separation_mode,
                        region_bias=self.config.soft_mode_bias
                        if separation_mode == "soft"
                        else 1.0,
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

        env_cfg.label = f"biased_foraging_{size_name}_{separation_mode}_{recipe_mode}"
        return env_cfg


def make_env(
    size: str = "medium",
    separation: Optional[str] = None,
    agents: int = 1,
    soft_bias: float = 0.75,
    recipe_mode: Optional[list[str]] = None,
    num_assemblers: Optional[int] = None,
    max_steps: int = 512,
    seed: int = 42,
) -> MettaGridConfig:
    """Generic biased-foraging env factory with configurable arguments."""
    cfg_kwargs: dict[str, Any] = {
        "num_agents": [agents],
        "max_steps": max_steps,
        "soft_mode_bias": soft_bias,
    }

    if separation in ("strict", "soft"):
        cfg_kwargs["separation_modes"] = [separation]
        cfg_kwargs["separation_weights"] = [1.0]

    if recipe_mode:
        cfg_kwargs["recipe_mode"] = recipe_mode

    if num_assemblers is not None:
        cfg_kwargs["num_assemblers"] = [num_assemblers]

    task_gen = BiasedForagingTaskGenerator(
        BiasedForagingTaskGenerator.Config(**cfg_kwargs)
    )

    rng = random.Random(seed)

    # Find an env matching the requested size if provided
    env = task_gen._generate_task(0, rng)
    if size is not None:
        found = None
        for i in range(30):
            candidate = task_gen._generate_task(i, rng)
            if size in candidate.label:
                found = candidate
                break
        if found is not None:
            env = found
    return env


def train(
    separation: str = "mixed",
    agents: int = 1,
    soft_bias: float = 0.75,
    recipe_mode: Optional[list[str]] = None,
    run_evals: bool = True,
) -> TrainTool:
    """Train with configurable separation bias and agent count.

    CLI example:
      uv run ./tools/run.py experiments.recipes.in_context_learning.foraging.train separation=soft agents=2
    """
    if separation == "strict":
        sep_modes = ["strict"]
        sep_weights: Optional[list[float]] = [1.0]
    elif separation == "soft":
        sep_modes = ["soft"]
        sep_weights = [1.0]
    else:  # mixed
        sep_modes = ["strict", "soft"]
        sep_weights = [0.5, 0.5]

    task_generator_cfg = BiasedForagingTaskGenerator.Config(
        num_agents=[agents],
        max_steps=512,
        map_sizes={
            "small": {"width": 10, "height": 10, "resource_count": 2},
            "medium": {"width": 16, "height": 16, "resource_count": 3},
            "large": {"width": 24, "height": 24, "resource_count": 4},
            "extra_large": {"width": 32, "height": 32, "resource_count": 5},
            "extra_extra_large": {"width": 64, "height": 64, "resource_count": 6},
        },
        size_weights=[0.4, 0.4, 0.2, 0.1, 0.1],
        separation_modes=sep_modes,
        separation_weights=sep_weights,
        soft_mode_bias=soft_bias,
        recipe_mode=recipe_mode or ["simple"],
    )
    curriculum = CurriculumConfig(
        task_generator=task_generator_cfg,
        algorithm_config=LearningProgressConfig(),
    )
    trainer_cfg = TrainerConfig(losses=LossConfig())
    trainer_cfg.batch_size = 4177920
    trainer_cfg.bptt_horizon = 512

    evaluator_cfg = (
        EvaluatorConfig(simulations=make_eval_suite())
        if run_evals
        else EvaluatorConfig()
    )

    return TrainTool(
        trainer=trainer_cfg,
        training_env=TrainingEnvironmentConfig(curriculum=curriculum),
        evaluator=evaluator_cfg,
    )


def play(
    size: str = "medium",
    separation: Optional[str] = None,
    agents: int = 1,
    soft_bias: float = 0.75,
    recipe_mode: Optional[list[str]] = None,
    num_assemblers: Optional[int] = None,
    max_steps: int = 512,
    seed: int = 42,
) -> PlayTool:
    """Generic biased-foraging play entrypoint with configurable arguments."""
    env = make_env(
        size=size,
        separation=separation,
        agents=agents,
        soft_bias=soft_bias,
        recipe_mode=recipe_mode,
        num_assemblers=num_assemblers,
        max_steps=max_steps,
        seed=seed,
    )

    parts: list[str] = ["biased_foraging"]
    if size:
        parts.append(size)
    if separation in ("strict", "soft"):
        parts.append(separation)
    if recipe_mode:
        parts.append("_".join(recipe_mode))
    sim_name = "_".join(parts)

    return PlayTool(
        sim=SimulationConfig(env=env, name=sim_name, suite="in_context_learning")
    )


def make_eval_suite() -> list[SimulationConfig]:
    """Create a comprehensive foraging evaluation suite."""
    eval_configs = [
        # Baseline single agent
        {
            "agents": 1,
            "size": "small",
            "separation": "strict",
            "recipe_mode": ["simple"],
        },
        {
            "agents": 1,
            "size": "medium",
            "separation": "strict",
            "recipe_mode": ["simple"],
        },
        # Generalization to soft separation
        {
            "agents": 1,
            "size": "medium",
            "separation": "soft",
            "recipe_mode": ["simple"],
        },
        # Generalization to more complex recipes
        {
            "agents": 1,
            "size": "medium",
            "separation": "strict",
            "recipe_mode": ["directional"],
        },
        {
            "agents": 1,
            "size": "medium",
            "separation": "strict",
            "recipe_mode": ["unordered_chain"],
        },
        # Generalization to larger maps
        {
            "agents": 1,
            "size": "large",
            "separation": "strict",
            "recipe_mode": ["simple"],
        },
        # Multi-agent scenarios
        {
            "agents": 2,
            "size": "medium",
            "separation": "strict",
            "recipe_mode": ["simple"],
        },
        {
            "agents": 2,
            "size": "medium",
            "separation": "soft",
            "recipe_mode": ["simple"],
        },
        {
            "agents": 2,
            "size": "large",
            "separation": "strict",
            "recipe_mode": ["directional"],
        },
        {
            "agents": 2,
            "size": "large",
            "separation": "strict",
            "recipe_mode": ["unordered_chain"],
        },
        # Multi-assembler
        {
            "agents": 1,
            "size": "large",
            "separation": "strict",
            "recipe_mode": ["simple"],
            "num_assemblers": 2,
        },
        {
            "agents": 2,
            "size": "large",
            "separation": "soft",
            "recipe_mode": ["simple"],
            "num_assemblers": 2,
        },
    ]

    configs = []
    for params in eval_configs:
        name_parts = [
            f"{params.get('agents', 1)}a",
            params["size"],
            params["separation"],
            params["recipe_mode"][0],
        ]
        if "num_assemblers" in params:
            name_parts.append(f"{params['num_assemblers']}asm")

        name = f"foraging_eval_{'_'.join(name_parts)}"

        env = make_env(seed=42, **params)

        configs.append(
            SimulationConfig(
                env=env,
                name=name,
                suite="in_context_learning",
            )
        )

    return configs


def evaluate(
    policy_uri: str | None = None, simulations: list[SimulationConfig] | None = None
) -> SimTool:
    """Create a SimTool to run the foraging evaluation suite.

    If `simulations` not provided, uses the comprehensive foraging `make_eval_suite()`.
    """
    sims = simulations or make_eval_suite()
    return SimTool(simulations=sims, policy_uris=[policy_uri] if policy_uri else None)
