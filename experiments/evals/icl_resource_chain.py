from metta.map.mapgen import MapGen
from metta.mettagrid.config import empty_converters
from metta.mettagrid.config.envs import make_icl_resource_chain
from metta.mettagrid.mettagrid_config import EnvConfig
from metta.sim.simulation_config import SimulationConfig


def make_icl_resource_chain_eval_env(
    name: str,
    num_agents: int,
    max_steps: int,
    game_objects: dict,
    map_builder_objects: dict,
    border_width: int = 1,
) -> EnvConfig:
    ascii_map = f"mettagrid/configs/maps/icl_resource_chain/{name}.map"
    env = make_icl_resource_chain(
        num_agents=num_agents,
        max_steps=max_steps,
        game_objects=game_objects,
        map_builder_objects=map_builder_objects,
    )
    env.game.max_steps = max_steps
    env.game.map_builder = MapGen.Config(
        instances=num_agents,
        instance_map=MapGen.Config.with_ascii_uri(ascii_map, border_width=border_width),
    )

    env.game.agent.resource_limits["heart"] = 5

    return env

def update_recipe(converter, input_resource = None, output_resource = None):
    if input_resource is not None:
        converter.input_resources = {input_resource: 1}
    if output_resource is not None:
        converter.output_resources = {output_resource: 1}


def chain_length2_0sink_env(name: str, max_steps: int = 50) -> EnvConfig:
    # recipe is laser -> heart
    map_builder_objects = {
        "lasery": 1,
        "factory": 1,
    }

    game_objects = {
        "lasery": update_recipe(empty_converters.lasery, output_resource="laser"),
        "factory": update_recipe(empty_converters.factory, input_resource="laser", output_resource="heart"),
    }

    return make_icl_resource_chain_eval_env(
        name="2chain0sink",
        num_agents=24,
        max_steps=max_steps,
        game_objects=game_objects,
        map_builder_objects=map_builder_objects,
    )


def chain_length2_1sink_env(name: str, max_steps: int = 50) -> EnvConfig:
    # recipe is laser -> heart
    map_builder_objects = {
        "lasery": 1,
        "factory": 1,
        "mine_red": 1,
    }

    game_objects = {
        "lasery": update_recipe(empty_converters.lasery, output_resource="laser"),
        "factory": update_recipe(empty_converters.factory, input_resource="laser", output_resource="heart"),
        "mine_red": update_recipe(empty_converters.mine_red, input_resource="laser"),
    }

    return make_icl_resource_chain_eval_env(
        name="2chain1sink",
        num_agents=24,
        max_steps=max_steps,
        game_objects=game_objects,
        map_builder_objects=map_builder_objects,
    )


def chain_length2_2sink_env(name: str, max_steps: int = 50) -> EnvConfig:
    # recipe is laser -> heart, two sinks
    game_objects = {
        "lasery": update_recipe(empty_converters.lasery, output_resource="laser"),
        "factory": update_recipe(empty_converters.factory, input_resource="laser", output_resource="heart"),
        "mine_red": update_recipe(empty_converters.mine_red, input_resource="laser"),
        "temple": update_recipe(empty_converters.temple, input_resource="laser"),
    }

    map_builder_objects = {
        "lasery": 1,
        "factory": 1,
        "mine_red": 1,
        "temple": 1,
    }

    return make_icl_resource_chain_eval_env(
        name="2chain2sink",
        num_agents=24,
        max_steps=max_steps,
        game_objects=game_objects,
        map_builder_objects=map_builder_objects,
    )


def chain_length3_0sink_env(name: str, max_steps: int = 100) -> EnvConfig:
    # recipe is laser -> blueprint -> heart
    game_objects = {
        "lasery": update_recipe(empty_converters.lasery, output_resource="laser"),
        "factory": update_recipe(empty_converters.factory, input_resource="laser", output_resource="blueprint"),
        "temple": update_recipe(empty_converters.temple, input_resource="blueprint", output_resource="heart"),
    }

    map_builder_objects = {
        "lasery": 1,
        "factory": 1,
        "temple": 1,
    }

    return make_icl_resource_chain_eval_env(
        name="3chain0sink",
        num_agents=24,
        max_steps=max_steps,
        game_objects=game_objects,
        map_builder_objects=map_builder_objects,
    )


def chain_length3_1sink_env(name: str, max_steps: int = 100) -> EnvConfig:
    # recipe is laser -> blueprint -> heart
    game_objects = {
        "lasery": update_recipe(empty_converters.lasery, output_resource="laser"),
        "factory": update_recipe(empty_converters.factory, input_resource="laser", output_resource="blueprint"),
        "temple": update_recipe(empty_converters.temple, input_resource="blueprint", output_resource="heart"),
        "generator_blue": update_recipe(empty_converters.generator_blue, input_resource="blueprint"),
    }

    map_builder_objects = {
        "lasery": 1,
        "factory": 1,
        "temple": 1,
        "generator_blue": 1,
    }

    return make_icl_resource_chain_eval_env(
        name="3chain1sink",
        num_agents=24,
        max_steps=max_steps,
        game_objects=game_objects,
        map_builder_objects=map_builder_objects,
    )


def chain_length3_2sink_env(name: str, max_steps: int = 100) -> EnvConfig:
    # recipe is laser -> blueprint -> heart
    game_objects = {
        "lasery": update_recipe(empty_converters.lasery, output_resource="laser"),
        "factory": update_recipe(empty_converters.factory, input_resource="laser", output_resource="blueprint"),
        "temple": update_recipe(empty_converters.temple, input_resource="blueprint", output_resource="heart"),
        "generator_blue": update_recipe(empty_converters.generator_blue, input_resource="blueprint"),
        "mine_red": update_recipe(empty_converters.mine_red, input_resource="laser"),
    }

    map_builder_objects = {
        "lasery": 1,
        "factory": 1,
        "temple": 1,
        "generator_blue": 1,
        "mine_red": 1,
    }

    return make_icl_resource_chain_eval_env(
        name="3chain2sink",
        num_agents=24,
        max_steps=max_steps,
        game_objects=game_objects,
        map_builder_objects=map_builder_objects,
    )


def chain_length4_0sink_env(name: str, max_steps: int = 200) -> EnvConfig:
    # recipe is laser -> blueprint -> armor -> heart
    game_objects = {
        "lasery": update_recipe(empty_converters.lasery, output_resource="laser"),
        "factory": update_recipe(empty_converters.factory, input_resource="laser", output_resource="blueprint"),
        "temple": update_recipe(empty_converters.temple, input_resource="blueprint", output_resource="armor"),
        "generator_blue": update_recipe(empty_converters.generator_blue, input_resource="armor", output_resource="heart"),
    }

    map_builder_objects = {
        "lasery": 1,
        "factory": 1,
        "temple": 1,
        "generator_blue": 1,
    }

    return make_icl_resource_chain_eval_env(
        name="4chain0sink",
        num_agents=24,
        max_steps=max_steps,
        game_objects=game_objects,
        map_builder_objects=map_builder_objects,
    )


def chain_length4_1sink_env(name: str, max_steps: int = 200) -> EnvConfig:
    # recipe is laser -> blueprint -> armor -> heart
    game_objects = {
        "lasery": update_recipe(empty_converters.lasery, output_resource="laser"),
        "factory": update_recipe(empty_converters.factory, input_resource="laser", output_resource="blueprint"),
        "temple": update_recipe(empty_converters.temple, input_resource="blueprint", output_resource="armor"),
        "generator_blue": update_recipe(empty_converters.generator_blue, input_resource="armor", output_resource="heart"),
        "mine_red": update_recipe(empty_converters.mine_red, input_resource="laser"),
    }

    map_builder_objects = {
        "lasery": 1,
        "factory": 1,
        "temple": 1,
        "generator_blue": 1,
        "mine_red": 1,
    }

    return make_icl_resource_chain_eval_env(
        name="4chain1sink",
        num_agents=24,
        max_steps=max_steps,
        game_objects=game_objects,
        map_builder_objects=map_builder_objects,
    )


def make_icl_resource_chain_eval_suite() -> list[SimulationConfig]:
    return [
        SimulationConfig(
            name="chain_length2_0sink",
            env=chain_length2_0sink_env("chain_length2_0sink", 50),
        ),
        SimulationConfig(
            name="chain_length2_1sink",
            env=chain_length2_1sink_env("chain_length2_1sink", 50),
        ),
        SimulationConfig(
            name="chain_length2_2sink",
            env=chain_length2_2sink_env("chain_length2_2sink", 50),
        ),
        SimulationConfig(
            name="chain_length3_0sink",
            env=chain_length3_0sink_env("chain_length3_0sink", 100),
        ),
        SimulationConfig(
            name="chain_length3_1sink",
            env=chain_length3_1sink_env("chain_length3_1sink", 100),
        ),
        SimulationConfig(
            name="chain_length3_2sink",
            env=chain_length3_2sink_env("chain_length3_2sink", 100),
        ),
        SimulationConfig(
            name="chain_length4_0sink",
            env=chain_length4_0sink_env("chain_length4_0sink", 200),
        ),
        SimulationConfig(
            name="chain_length4_1sink",
            env=chain_length4_1sink_env("chain_length4_1sink", 200),
        ),
    ]
