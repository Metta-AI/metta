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
    max_steps: int = 512,
    seed: int = 42,
) -> MettaGridConfig:
    """Build a foraging evaluation environment with explicit control over recipes.

    This constructs the generator directly so we can target specific patterns (e.g.,
    N, NE, Any) and input complexities (1 or 2 resources) deterministically.
    """
    if size not in {"small", "medium"}:
        raise ValueError("size must be 'small' or 'medium' for evals")

    # Fix the map to a single requested size for determinism
    map_sizes = (
        {"small": {"width": 10, "height": 10, "resource_count": 2}}
        if size == "small"
        else {"medium": {"width": 16, "height": 16, "resource_count": 3}}
    )

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

    task_gen = BiasedForagingTaskGenerator(
        BiasedForagingTaskGenerator.Config(**cfg_kwargs)
    )

    rng = __import__("random").Random(seed)
    # Deterministic single-sample because map_sizes contains one size
    return task_gen._generate_task(0, rng)


def make_foraging_eval_suite() -> list[SimulationConfig]:
    """Evaluation suite covering directional (N,S,E,W,NE,NS,NW,EW,ES,WS)
    and ANY with 1- or 2-resource inputs on small and medium maps.
    """
    sizes = ["small", "medium"]
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

    def to_dirs(label: str) -> list[Position]:
        return [cast(Position, c) for c in label]

    sims: list[SimulationConfig] = []

    # Directional patterns with 1- and 2-resource inputs
    for size in sizes:
        for patt in patterns:
            for k in complexities:
                env = make_foraging_eval_env(
                    size=size,
                    separation="strict",
                    agents=1,
                    recipe_mode=["directional"],
                    max_recipe_inputs=k,
                    direction_recipes=[to_dirs(patt)],
                    seed=42,
                )
                name = f"foraging_eval_{size}_dir_{patt}_k{k}"
                sims.append(
                    SimulationConfig(env=env, name=name, suite="in_context_learning")
                )

    # ANY position with exactly 1 or 2 inputs
    for size in sizes:
        for k in complexities:
            env = make_foraging_eval_env(
                size=size,
                separation="strict",
                agents=1,
                recipe_mode=["simple"],
                max_recipe_inputs=k,
                seed=42,
            )
            name = f"foraging_eval_{size}_any_k{k}"
            sims.append(
                SimulationConfig(env=env, name=name, suite="in_context_learning")
            )

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
