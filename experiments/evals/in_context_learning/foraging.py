from typing import cast

from metta.sim.simulation_config import SimulationConfig
from metta.tools.sim import SimTool
from mettagrid.config.mettagrid_config import MettaGridConfig, Position

from experiments.recipes.in_context_learning.foraging import (
    BiasedForagingTaskGenerator,
)


def make_foraging_eval_env(
    size: str = "medium",
    separation: str | None = "strict",
    agents: int = 1,
    soft_bias: float = 0.75,
    recipe_mode: list[str] | None = None,
    max_recipe_inputs: int | None = None,
    direction_recipes: list[list[Position]] | None = None,
    num_assemblers: int | None = None,
    resource_type_counts: list[int] | None = None,
    max_steps: int = 512,
    seed: int = 42,
) -> MettaGridConfig:
    """Build a foraging evaluation environment with explicit control over recipes.

    This constructs the generator directly so we can target specific patterns (e.g.,
    N, NE, Any) and input complexities (1 or 2 resources) deterministically.
    """
    if size not in {"small", "medium", "large", "extra_large"}:
        raise ValueError("size must be one of 'small','medium','large','extra_large'")

    # Fix the map to a single requested size for determinism
    size_cfg = {
        "small": {"width": 10, "height": 10, "resource_count": 2},
        "medium": {"width": 16, "height": 16, "resource_count": 3},
        "large": {"width": 24, "height": 24, "resource_count": 4},
        "extra_large": {"width": 32, "height": 32, "resource_count": 5},
    }
    map_sizes = {size: size_cfg[size]}

    cfg_kwargs: dict = {
        "num_agents": [agents],
        "num_assemblers": [num_assemblers] if num_assemblers is not None else [1],
        "max_steps": max_steps,
        "map_sizes": map_sizes,
        "size_weights": [1.0],
        "soft_mode_bias": soft_bias,
        "separation_modes": [separation or "strict"],
        "separation_weights": [1.0],
        "recipe_mode": recipe_mode or ["simple"],
    }
    if max_recipe_inputs is not None:
        cfg_kwargs["max_recipe_inputs"] = [max_recipe_inputs]
    if direction_recipes is not None:
        cfg_kwargs["direction_recipes"] = direction_recipes
    if resource_type_counts is not None:
        cfg_kwargs["resource_type_counts"] = resource_type_counts

    task_gen = BiasedForagingTaskGenerator(
        BiasedForagingTaskGenerator.Config(**cfg_kwargs)
    )

    rng = __import__("random").Random(seed)
    # Deterministic single-sample because map_sizes contains one size
    env = task_gen._generate_task(0, rng)

    # Always disable inventory actions in evals per requirement
    if hasattr(env.game, "actions"):
        actions = env.game.actions
        if hasattr(actions, "get_items") and hasattr(actions.get_items, "enabled"):
            actions.get_items.enabled = False
        if hasattr(actions, "put_items") and hasattr(actions.put_items, "enabled"):
            actions.put_items.enabled = False

    return env


def make_foraging_eval_suite() -> list[SimulationConfig]:
    """Evaluation suite covering directional (N,S,E,W,NE,NS,NW,EW,ES,WS)
    and ANY with 1- or 2-resource inputs on small and medium maps.
    """
    sizes = ["small", "medium", "large", "extra_large"]
    patterns = [
        "N",
        "S",
        "E",
        "W",
        "NE",
        "NS",
        "NW",
        "EW",
        "ES",
        "WS",
    ]
    complexities = [1, 2]
    agents_list = [1, 2]

    def to_dirs(label: str) -> list[Position]:
        return [cast(Position, c) for c in label]

    sims: list[SimulationConfig] = []
    seed_base = 42
    sim_counter = 0

    # Directional patterns with 1- and 2-resource inputs
    for size in sizes:
        for agents in agents_list:
            for patt in patterns:
                for k in complexities:
                    env = make_foraging_eval_env(
                        size=size,
                        separation="strict",
                        agents=agents,
                        recipe_mode=["directional"],
                        max_recipe_inputs=k,
                        direction_recipes=[to_dirs(patt)],
                        resource_type_counts=[1, 2, 3, 4],
                        seed=seed_base + sim_counter,
                    )
                    name = f"foraging_eval_{agents}a_{size}_dir_{patt}_k{k}"
                    sims.append(
                        SimulationConfig(
                            env=env, name=name, suite="in_context_learning"
                        )
                    )
                    sim_counter += 1

    # ANY position with exactly 1 or 2 inputs
    for size in sizes:
        for agents in agents_list:
            for k in complexities:
                env = make_foraging_eval_env(
                    size=size,
                    separation="strict",
                    agents=agents,
                    recipe_mode=["simple"],
                    max_recipe_inputs=k,
                    resource_type_counts=[1, 2, 3, 4],
                    seed=seed_base + sim_counter,
                )
                name = f"foraging_eval_{agents}a_{size}_any_k{k}"
                sims.append(
                    SimulationConfig(env=env, name=name, suite="in_context_learning")
                )
                sim_counter += 1

    return sims


def evaluate(
    policy_uri: str | None = None,
    simulations: list[SimulationConfig] | None = None,
) -> SimTool:
    """Create a SimTool to run the foraging evaluation suite.

    If `simulations` not provided, uses the comprehensive foraging `make_foraging_eval_suite()`.
    """
    sims = simulations or make_foraging_eval_suite()
    return SimTool(simulations=sims, policy_uris=[policy_uri] if policy_uri else None)
