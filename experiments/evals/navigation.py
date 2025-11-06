import metta.sim.simulation_config
import mettagrid.builder.envs
import mettagrid.config.mettagrid_config
import mettagrid.map_builder.ascii
import mettagrid.mapgen.mapgen
import mettagrid.mapgen.scenes.mean_distance

import experiments.evals.cfg


def make_nav_eval_env(
    env: mettagrid.config.mettagrid_config.MettaGridConfig,
) -> mettagrid.config.mettagrid_config.MettaGridConfig:
    """Set the heart reward to 0.333 for normalization"""
    env.game.agent.rewards.inventory["heart"] = 0.333
    return env


def make_nav_ascii_env(
    name: str,
    max_steps: int,
    num_agents=1,
    num_instances=4,
    border_width: int = 6,
    instance_border_width: int = 3,
) -> mettagrid.config.mettagrid_config.MettaGridConfig:
    # we re-use nav sequence maps, but replace all objects with altars
    path = f"packages/mettagrid/configs/maps/navigation_sequence/{name}.map"

    env = mettagrid.builder.envs.make_navigation(num_agents=num_agents * num_instances)
    env.game.max_steps = max_steps

    map_instance = mettagrid.map_builder.ascii.AsciiMapBuilder.Config.from_uri(path)

    # replace objects with altars
    map_instance.char_to_name_map["n"] = "altar"
    map_instance.char_to_name_map["m"] = "altar"

    env.game.map_builder = mettagrid.mapgen.mapgen.MapGen.Config(
        instances=num_instances,
        border_width=border_width,
        instance_border_width=instance_border_width,
        instance=map_instance,
    )

    return make_nav_eval_env(env)


def make_emptyspace_sparse_env() -> mettagrid.config.mettagrid_config.MettaGridConfig:
    env = mettagrid.builder.envs.make_navigation(num_agents=4)
    env.game.max_steps = 300
    env.game.map_builder = mettagrid.mapgen.mapgen.MapGen.Config(
        instances=4,
        instance=mettagrid.mapgen.mapgen.MapGen.Config(
            width=60,
            height=60,
            border_width=3,
            instance=mettagrid.mapgen.scenes.mean_distance.MeanDistance.Config(
                mean_distance=30,
                objects={"altar": 3},
            ),
        ),
    )
    return make_nav_eval_env(env)


def make_navigation_eval_suite() -> list[metta.sim.simulation_config.SimulationConfig]:
    evals = [
        metta.sim.simulation_config.SimulationConfig(
            suite="navigation",
            name=eval["name"],
            env=make_nav_ascii_env(
                name=eval["name"],
                max_steps=eval["max_steps"],
                num_agents=eval["num_agents"],
                num_instances=eval["num_instances"],
            ),
        )
        for eval in experiments.evals.cfg.NAVIGATION_EVALS
    ] + [
        metta.sim.simulation_config.SimulationConfig(
            suite="navigation",
            name="emptyspace_sparse",
            env=make_emptyspace_sparse_env(),
        )
    ]
    return evals
