from metta.sim.simulation_config import SimulationConfig
from mettagrid.builder import building, empty_converters
from mettagrid.config.mettagrid_config import (
    ActionConfig,
    ActionsConfig,
    AgentConfig,
    AgentRewards,
    ConverterConfig,
    GameConfig,
    MettaGridConfig,
    WallConfig,
)
from mettagrid.mapgen.mapgen import MapGen
from mettagrid.mapgen.scenes.mean_distance import MeanDistance


def make_object_use_env(
    name: str,
    max_steps: int,
    objects: dict[str, ConverterConfig | WallConfig],
    map_objects: dict[str, int],
    rewards: dict[str, float],
    num_agents: int = 1,
    num_instances: int = 4,
) -> MettaGridConfig:
    """Create an object use evaluation environment."""

    # Base configuration for object use
    env = MettaGridConfig(
        game=GameConfig(
            num_agents=num_agents * num_instances,
            max_steps=max_steps,
            objects=objects,
            actions=ActionsConfig(
                move=ActionConfig(),
                rotate=ActionConfig(),
                get_items=ActionConfig(),
                put_items=ActionConfig(),
            ),
            agent=AgentConfig(
                default_resource_limit=50,
                rewards=AgentRewards(inventory=rewards),
            ),
            map_builder=MapGen.Config(
                instances=num_instances,
                border_width=6,
                instance_border_width=3,
                instance=MapGen.Config(
                    width=11,
                    height=11,
                    border_width=3,
                    instance=MeanDistance.Config(
                        mean_distance=6,
                        objects=map_objects,
                    ),
                ),
            ),
        )
    )
    return env


def make_object_use_ascii_env(
    name: str,
    ascii_map: str,
    max_steps: int,
    objects: dict[str, ConverterConfig | WallConfig],
    rewards: dict[str, float],
    num_agents: int = 1,
    num_instances: int = 4,
) -> MettaGridConfig:
    """Create an object use evaluation environment from ASCII map."""

    env = MettaGridConfig(
        game=GameConfig(
            num_agents=num_agents * num_instances,
            max_steps=max_steps,
            objects=objects,
            actions=ActionsConfig(
                move=ActionConfig(),
                get_items=ActionConfig(),
                put_items=ActionConfig(),
            ),
            agent=AgentConfig(
                default_resource_limit=50,
                rewards=AgentRewards(inventory=rewards),
            ),
            map_builder=MapGen.Config(
                instances=num_instances,
                border_width=6,
                instance_border_width=3,
                instance=MapGen.Config.with_ascii_uri(
                    f"packages/mettagrid/configs/maps/object_use/{ascii_map}.map",
                    border_width=1,
                ),
            ),
        )
    )
    return env


def make_armory_use_env() -> MettaGridConfig:
    """Test armory: ore_red -> armor"""

    armory = building.armory.model_copy()
    armory.initial_resource_count = 0
    armory.cooldown = 255

    mine = building.mine_red.model_copy()
    mine.initial_resource_count = armory.input_resources.get("ore_red", 0)
    mine.cooldown = 255

    return make_object_use_env(
        name="armory_use",
        max_steps=100,
        objects={"wall": building.wall, "mine_red": mine, "armory": armory},
        map_objects={"armory": 1, "mine_red": 1},
        rewards={"armor": 1, "ore_red": 0},
    )


def make_armory_use_free_env() -> MettaGridConfig:
    """Test armory with free resources"""
    armory = building.armory.model_copy()
    armory.initial_resource_count = 1
    armory.cooldown = 255

    return make_object_use_env(
        name="armory_use_free",
        max_steps=80,
        objects={"wall": building.wall, "armory": armory},
        map_objects={"armory": 1},
        rewards={"armor": 1},
    )


def make_generator_use_env() -> MettaGridConfig:
    """Test generator: ore_red -> battery_red"""

    generator = building.generator_red.model_copy()
    generator.initial_resource_count = 0
    generator.cooldown = 255

    mine = building.mine_red.model_copy()
    mine.initial_resource_count = generator.input_resources.get("ore_red", 0)
    mine.cooldown = 255

    return make_object_use_env(
        name="generator_use",
        max_steps=100,
        objects={"wall": building.wall, "mine_red": mine, "generator_red": generator},
        map_objects={"generator_red": 1, "mine_red": 1},
        rewards={"battery_red": 1, "ore_red": 0},
    )


def make_generator_use_free_env() -> MettaGridConfig:
    """Test generator with free resources"""
    generator = building.generator_red.model_copy()
    generator.initial_resource_count = 1
    generator.cooldown = 255

    return make_object_use_env(
        name="generator_use_free",
        max_steps=80,
        objects={"wall": building.wall, "generator_red": generator},
        map_objects={"generator_red": 1},
        rewards={"battery_red": 1},
    )


def make_lasery_use_env() -> MettaGridConfig:
    """Test lasery: ore_red + battery_red -> laser"""

    lasery = building.lasery.model_copy()
    lasery.initial_resource_count = 0
    lasery.cooldown = 255

    generator = building.generator_red.model_copy()
    generator.initial_resource_count = lasery.input_resources.get("battery_red", 0)
    generator.cooldown = 255

    mine = building.mine_red.model_copy()
    mine.initial_resource_count = lasery.input_resources.get("ore_red", 0)
    mine.cooldown = 255

    return make_object_use_env(
        name="lasery_use",
        max_steps=200,
        objects={
            "wall": building.wall,
            "mine_red": mine,
            "generator_red": generator,
            "lasery": lasery,
        },
        map_objects={"lasery": 1, "generator_red": 1, "mine_red": 1},
        rewards={"laser": 1, "ore_red": 0, "battery_red": 0},
    )


def make_lasery_use_free_env() -> MettaGridConfig:
    """Test lasery with free resources"""
    lasery = building.lasery.model_copy()
    lasery.initial_resource_count = 1
    lasery.cooldown = 255

    return make_object_use_env(
        name="lasery_use_free",
        max_steps=80,
        objects={"wall": building.wall, "lasery": lasery},
        map_objects={"lasery": 1},
        rewards={"laser": 1},
    )


def make_mine_use_env() -> MettaGridConfig:
    """Test mine: produces ore_red"""
    mine = building.mine_red.model_copy()
    mine.initial_resource_count = 0
    mine.cooldown = 255

    return make_object_use_env(
        name="mine_use",
        max_steps=80,
        objects={"wall": building.wall, "mine_red": mine},
        map_objects={"mine_red": 1},
        rewards={"ore_red": 1},
    )


def make_temple_use_free_env() -> MettaGridConfig:
    """Test temple: produces hearts"""
    # Use empty template and set outputs explicitly; there is no temple in building
    temple = empty_converters.temple.model_copy()
    temple.output_resources = {"heart": 1}
    temple.initial_resource_count = 1
    temple.cooldown = 255

    return make_object_use_env(
        name="temple_use_free",
        max_steps=80,
        objects={"wall": building.wall, "temple": temple},
        map_objects={"temple": 1},
        rewards={"heart": 1},
    )


def make_altar_use_free_env() -> MettaGridConfig:
    """Test altar: produces hearts"""
    altar = building.altar.model_copy()
    altar.initial_resource_count = 1
    altar.cooldown = 255

    return make_object_use_env(
        name="altar_use_free",
        max_steps=80,
        objects={"wall": building.wall, "altar": altar},
        map_objects={"altar": 1},
        rewards={"heart": 1},
    )


def make_shoot_out_env() -> MettaGridConfig:
    """Test shooting mechanics with ASCII map"""

    lasery = building.lasery.model_copy()
    lasery.initial_resource_count = 20

    altar = building.altar.model_copy()
    altar.initial_resource_count = 1
    altar.cooldown = 255

    return make_object_use_ascii_env(
        name="shoot_out",
        ascii_map="shoot_out",
        max_steps=60,
        objects={
            "wall": building.wall,
            "altar": altar,
            "lasery": lasery,
        },
        rewards={"heart": 1},
    )


def make_swap_in_env() -> MettaGridConfig:
    """Test swap in mechanics with ASCII map"""
    altar = building.altar.model_copy()
    altar.initial_resource_count = 1
    altar.cooldown = 255

    env = make_object_use_ascii_env(
        name="swap_in",
        ascii_map="swap_in",
        max_steps=30,
        objects={
            "wall": building.wall,
            "block": building.block,
            "altar": altar,
        },
        rewards={"heart": 1},
    )
    env.game.actions.swap = ActionConfig()
    return env


def make_swap_out_env() -> MettaGridConfig:
    """Test swap out mechanics with ASCII map"""
    altar = building.altar.model_copy()
    altar.initial_resource_count = 1
    altar.cooldown = 255

    env = make_object_use_ascii_env(
        name="swap_out",
        ascii_map="swap_out",
        max_steps=30,
        objects={
            "wall": building.wall,
            "block": building.block,
            "altar": altar,
        },
        rewards={"heart": 1},
    )
    env.game.actions.swap = ActionConfig()
    return env


def make_full_sequence_env() -> MettaGridConfig:
    """Full sequence test: mine -> generator -> altar"""

    altar = building.altar.model_copy()
    altar.initial_resource_count = 0
    altar.cooldown = 255

    generator = building.generator_red.model_copy()
    generator.initial_resource_count = 0
    generator.input_resources = {"ore_red": 1}
    generator.output_resources = {"battery_red": 1}
    generator.cooldown = 1

    mine = building.mine_red.model_copy()
    mine.initial_resource_count = altar.input_resources.get("battery_red", 0)
    mine.output_resources = {"ore_red": 1}
    mine.cooldown = 255

    return make_object_use_env(
        name="full_sequence",
        max_steps=100,
        objects={
            "wall": building.wall,
            "mine_red": mine,
            "generator_red": generator,
            "altar": altar,
        },
        map_objects={"mine_red": 1, "generator_red": 1, "altar": 1},
        rewards={"heart": 1, "ore_red": 0, "battery_red": 0},
    )


def make_object_use_eval_suite() -> list[SimulationConfig]:
    """Create the full object use evaluation suite."""
    return [
        SimulationConfig(
            suite="object_use", name="altar_use_free", env=make_altar_use_free_env()
        ),
        SimulationConfig(
            suite="object_use", name="armory_use_free", env=make_armory_use_free_env()
        ),
        SimulationConfig(
            suite="object_use", name="armory_use", env=make_armory_use_env()
        ),
        SimulationConfig(
            suite="object_use",
            name="generator_use_free",
            env=make_generator_use_free_env(),
        ),
        SimulationConfig(
            suite="object_use", name="generator_use", env=make_generator_use_env()
        ),
        SimulationConfig(
            suite="object_use", name="lasery_use_free", env=make_lasery_use_free_env()
        ),
        SimulationConfig(
            suite="object_use", name="lasery_use", env=make_lasery_use_env()
        ),
        SimulationConfig(suite="object_use", name="mine_use", env=make_mine_use_env()),
        SimulationConfig(
            suite="object_use", name="shoot_out", env=make_shoot_out_env()
        ),
        SimulationConfig(suite="object_use", name="swap_in", env=make_swap_in_env()),
        SimulationConfig(suite="object_use", name="swap_out", env=make_swap_out_env()),
        SimulationConfig(
            suite="object_use", name="temple_use_free", env=make_temple_use_free_env()
        ),
        SimulationConfig(
            suite="object_use", name="full_sequence", env=make_full_sequence_env()
        ),
    ]
