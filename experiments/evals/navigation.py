from metta.sim.simulation_config import SimulationConfig
from mettagrid.builder.envs import make_navigation
from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.mapgen.mapgen import MapGen
from mettagrid.mapgen.scenes.mean_distance import MeanDistance

from experiments.evals.cfg import NAVIGATION_EVALS


def make_nav_eval_env(env: MettaGridConfig) -> MettaGridConfig:
    """Set the heart reward to 0.333 for normalization"""
    env.game.agent.rewards.inventory["heart"] = 0.333
    return env


def replace_objects_with_altars(name: str) -> str:
    ascii_map = f"packages/mettagrid/configs/maps/navigation_sequence/{name}.map"

    with open(ascii_map, "r") as f:
        map_content = f.read()

    return map_content.replace("n", "_").replace("m", "_")


def make_nav_ascii_env(
    name: str,
    max_steps: int,
    num_agents=1,
    num_instances=4,
    border_width: int = 6,
    instance_border_width: int = 3,
) -> MettaGridConfig:
    # we re-use nav sequence maps, but replace all objects with altars
    ascii_map = replace_objects_with_altars(name)

    env = make_navigation(num_agents=num_agents * num_instances)
    env.game.max_steps = max_steps
    env.game.map_builder = MapGen.Config(
        instances=num_instances,
        border_width=border_width,
        instance_border_width=instance_border_width,
        instance_map=MapGen.Config.with_ascii_map(ascii_map, border_width=border_width),
    )

    return make_nav_eval_env(env)


def make_emptyspace_sparse_env() -> MettaGridConfig:
    env = make_navigation(num_agents=4)
    env.game.max_steps = 300
    env.game.map_builder = MapGen.Config(
        instances=4,
        instance_map=MapGen.Config(
            width=60,
            height=60,
            border_width=3,
            root=MeanDistance.factory(
                params=MeanDistance.Params(
                    mean_distance=30,
                    objects={"altar": 3},
                )
            ),
        ),
    )
    return make_nav_eval_env(env)


def make_navigation_eval_suite() -> list[SimulationConfig]:
    evals = [
        SimulationConfig(
            suite="navigation",
            name=eval["name"],
            env=make_nav_ascii_env(
                name=eval["name"],
                max_steps=eval["max_steps"],
                num_agents=eval["num_agents"],
                num_instances=eval["num_instances"],
            ),
        )
        for eval in NAVIGATION_EVALS
    ] + [
        SimulationConfig(
            suite="navigation",
            name="emptyspace_sparse",
            env=make_emptyspace_sparse_env(),
        )
    ]
    return evals
