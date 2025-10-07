from metta.sim.simulation_config import SimulationConfig
from mettagrid.builder.envs import make_navigation_sequence
from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.map_builder.ascii import AsciiMapBuilder
from mettagrid.mapgen.mapgen import MapGen

from experiments.evals.cfg import NAVIGATION_EVALS


def make_nav_sequence_ascii_env(
    name: str,
    max_steps: int,
    num_agents=1,
    num_instances=4,
    border_width: int = 1,
    instance_border_width: int = 3,
) -> MettaGridConfig:
    ascii_map = f"packages/mettagrid/configs/maps/navigation_sequence/{name}.map"
    env = make_navigation_sequence(num_agents=num_agents * num_instances)
    env.game.max_steps = max_steps

    instance_config = MapGen.Config.with_ascii_uri(ascii_map, border_width=border_width)
    ascii_builder = instance_config.instance
    if isinstance(ascii_builder, AsciiMapBuilder.Config):
        override_map = {o.map_char: o.name for o in env.game.objects.values()}
        if override_map:
            merged = ascii_builder.char_to_name_map | override_map
            ascii_builder = ascii_builder.model_copy(
                update={"char_to_name_map": merged}
            )
            instance_config = instance_config.model_copy(
                update={"instance": ascii_builder}
            )

    env.game.map_builder = MapGen.Config(
        instances=num_instances,
        border_width=border_width,
        instance_border_width=instance_border_width,
        instance=instance_config,
    )

    # in evals, only complete the sequence once
    env.game.agent.resource_limits["heart"] = 1

    return env


def make_navigation_sequence_eval_suite() -> list[SimulationConfig]:
    evals = [
        SimulationConfig(
            suite="navigation_sequence",
            name=eval["name"],
            env=make_nav_sequence_ascii_env(
                name=eval["name"],
                max_steps=eval["max_steps"],
                num_agents=eval["num_agents"],
                num_instances=eval["num_instances"],
            ),
        )
        for eval in NAVIGATION_EVALS
    ]
    return evals
