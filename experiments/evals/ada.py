"""AdA-inspired evaluation suite that mirrors XLand-style multi-objective tasks."""

from __future__ import annotations

from typing import Callable, Dict, Tuple

from metta.sim.simulation_config import SimulationConfig
from mettagrid.builder import building
from mettagrid.config.mettagrid_config import (
    ActionConfig,
    ActionsConfig,
    AgentConfig,
    AgentRewards,
    AttackActionConfig,
    ChestConfig,
    ConverterConfig,
    GameConfig,
    MettaGridConfig,
    WallConfig,
)
from mettagrid.mapgen.mapgen import MapGen
from mettagrid.mapgen.scenes.mean_distance import MeanDistance


def _make_actions(allow_attack: bool) -> ActionsConfig:
    """Create a standard action set used across AdA evals."""
    actions = ActionsConfig(
        move=ActionConfig(),
        rotate=ActionConfig(),
        get_items=ActionConfig(),
        put_items=ActionConfig(),
    )
    if allow_attack:
        actions.attack = AttackActionConfig(
            consumed_resources={"laser": 1}, defense_resources={"armor": 1}
        )
    return actions


def _make_ada_env(
    *,
    name: str,
    max_steps: int,
    num_agents: int,
    num_instances: int,
    objects: Dict[str, ConverterConfig | ChestConfig | WallConfig],
    rewards: Dict[str, float],
    map_objects: Dict[str, int],
    map_size: Tuple[int, int],
    mean_distance: int,
    allow_attack: bool,
    default_resource_limit: int,
    resource_limits: Dict[str, int] | None = None,
) -> MettaGridConfig:
    """Shared helper that assembles a MettaGridConfig for AdA-style tasks."""
    effective_limits: Dict[str, int] = {"heart": 255}
    if resource_limits:
        effective_limits.update(resource_limits)

    return MettaGridConfig(
        label=f"ada.{name}",
        game=GameConfig(
            num_agents=num_agents * num_instances,
            max_steps=max_steps,
            actions=_make_actions(allow_attack),
            agent=AgentConfig(
                default_resource_limit=default_resource_limit,
                resource_limits=effective_limits,
                rewards=AgentRewards(inventory=rewards),
            ),
            objects=objects,
            map_builder=MapGen.Config(
                instances=num_instances,
                border_width=8,
                instance_border_width=4,
                instance=MapGen.Config(
                    width=map_size[0],
                    height=map_size[1],
                    border_width=4,
                    instance=MeanDistance.Config(
                        mean_distance=mean_distance,
                        objects=map_objects,
                    ),
                ),
            ),
        ),
    )


def _make_resource_chain_env() -> MettaGridConfig:
    """Pipeline task: mine -> generator -> altar conversion loop."""
    altar = building.altar.model_copy()
    altar.input_resources = {"battery_red": 2}
    altar.output_resources = {"heart": 1}
    altar.cooldown = 8

    generator = building.generator_red.model_copy()
    generator.input_resources = {"ore_red": 1}
    generator.output_resources = {"battery_red": 1}
    generator.cooldown = 6

    mine = building.mine_red.model_copy()
    mine.cooldown = 6

    wall = building.wall.model_copy()

    objects: Dict[str, ConverterConfig | WallConfig] = {
        "altar": altar,
        "generator_red": generator,
        "mine_red": mine,
        "wall": wall,
    }

    rewards = {"heart": 1.0, "battery_red": 0.05, "ore_red": 0.02}
    map_objects = {"altar": 3, "generator_red": 3, "mine_red": 6, "wall": 20}

    return _make_ada_env(
        name="resource_chain",
        max_steps=360,
        num_agents=4,
        num_instances=3,
        objects=objects,
        rewards=rewards,
        map_objects=map_objects,
        map_size=(23, 23),
        mean_distance=6,
        allow_attack=False,
        default_resource_limit=12,
        resource_limits={"battery_red": 14, "ore_red": 14},
    )


def _make_weapons_lab_env() -> MettaGridConfig:
    """Competitive task that rewards crafting lasers and spending them to score."""
    altar = building.altar.model_copy()
    altar.input_resources = {"laser": 1}
    altar.output_resources = {"heart": 2}
    altar.cooldown = 10

    lasery = building.lasery.model_copy()
    lasery.input_resources = {"battery_red": 1, "ore_red": 2}
    lasery.output_resources = {"laser": 1}
    lasery.cooldown = 8

    generator = building.generator_red.model_copy()
    generator.input_resources = {"ore_red": 1}
    generator.output_resources = {"battery_red": 1}
    generator.cooldown = 6

    mine = building.mine_red.model_copy()
    mine.cooldown = 6

    armory = building.armory.model_copy()
    armory.input_resources = {"ore_red": 2}
    armory.output_resources = {"armor": 1}
    armory.cooldown = 8

    wall = building.wall.model_copy()

    objects: Dict[str, ConverterConfig | WallConfig] = {
        "altar": altar,
        "lasery": lasery,
        "generator_red": generator,
        "mine_red": mine,
        "armory": armory,
        "wall": wall,
    }

    rewards = {"heart": 3.0, "laser": 0.15, "armor": 0.1}
    map_objects = {
        "altar": 4,
        "lasery": 3,
        "generator_red": 3,
        "mine_red": 6,
        "armory": 2,
        "wall": 24,
    }

    return _make_ada_env(
        name="weapons_lab",
        max_steps=420,
        num_agents=6,
        num_instances=3,
        objects=objects,
        rewards=rewards,
        map_objects=map_objects,
        map_size=(27, 27),
        mean_distance=7,
        allow_attack=True,
        default_resource_limit=16,
        resource_limits={"laser": 8, "armor": 6, "battery_red": 12, "ore_red": 16},
    )


def _make_cooperative_delivery_env() -> MettaGridConfig:
    """Cooperative delivery task that mirrors AdA's goal-conditioned logistics games."""
    altar = building.altar.model_copy()
    altar.input_resources = {"battery_red": 1}
    altar.output_resources = {"heart": 1}
    altar.cooldown = 6

    generator = building.generator_red.model_copy()
    generator.input_resources = {"ore_red": 1}
    generator.output_resources = {"battery_red": 1}
    generator.cooldown = 6

    mine = building.mine_red.model_copy()
    mine.cooldown = 6

    score_chest: ChestConfig = building.make_chest(
        resource_type="heart",
        type_id=21,
        name="score_chest",
        map_char="c",
        render_symbol="📥",
        position_deltas=[("N", 1), ("S", 1), ("E", 1), ("W", 1)],
        initial_inventory=0,
        max_inventory=-1,
    )

    wall = building.wall.model_copy()

    objects: Dict[str, ConverterConfig | ChestConfig | WallConfig] = {
        "altar": altar,
        "generator_red": generator,
        "mine_red": mine,
        "score_chest": score_chest,
        "wall": wall,
    }

    rewards = {"heart": 0.75, "battery_red": 0.05}
    map_objects = {
        "altar": 3,
        "generator_red": 3,
        "mine_red": 6,
        "score_chest": 2,
        "wall": 16,
    }

    return _make_ada_env(
        name="cooperative_delivery",
        max_steps=480,
        num_agents=6,
        num_instances=2,
        objects=objects,
        rewards=rewards,
        map_objects=map_objects,
        map_size=(29, 29),
        mean_distance=8,
        allow_attack=False,
        default_resource_limit=18,
        resource_limits={"battery_red": 14, "ore_red": 14},
    )


_ADA_BUILDERS: Dict[str, Callable[[], MettaGridConfig]] = {
    "resource_chain": _make_resource_chain_env,
    "weapons_lab": _make_weapons_lab_env,
    "cooperative_delivery": _make_cooperative_delivery_env,
}


def make_ada_eval_suite() -> list[SimulationConfig]:
    """Return the AdA-inspired evaluation suite."""
    return [
        SimulationConfig(suite="ada", name=name, env=builder())
        for name, builder in _ADA_BUILDERS.items()
    ]
