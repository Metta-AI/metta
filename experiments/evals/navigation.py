from metta.map.mapgen import MapGen
from metta.map.scenes.mean_distance import MeanDistance
from metta.mettagrid.config.envs import make_navigation
from metta.mettagrid.mettagrid_config import EnvConfig
from metta.sim.simulation_config import SimulationConfig


def make_nav_eval_env(env: EnvConfig) -> EnvConfig:
    """Set the heart reward to 0.333 for normalization"""
    env.game.agent.rewards.inventory.heart = 0.333
    return env


def make_nav_ascii_env(
    name: str, max_steps: int, border_width: int = 1, num_agents=4
) -> EnvConfig:
    ascii_map = f"mettagrid/configs/maps/navigation/{name}.map"
    env = make_navigation(num_agents=num_agents)
    env.game.max_steps = max_steps
    env.game.map_builder = MapGen.Config(
        instances=num_agents,
        border_width=6,
        instance_border_width=3,
        instance_map=MapGen.Config.with_ascii_uri(ascii_map, border_width=border_width),
    )

    return make_nav_eval_env(env)


def make_emptyspace_sparse_env() -> EnvConfig:
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
    return [
        SimulationConfig(name="corridors", env=make_nav_ascii_env("corridors", 450)),
        SimulationConfig(
            name="cylinder_easy", env=make_nav_ascii_env("cylinder_easy", 250)
        ),
        SimulationConfig(name="cylinder", env=make_nav_ascii_env("cylinder", 250)),
        SimulationConfig(name="honeypot", env=make_nav_ascii_env("honeypot", 300)),
        SimulationConfig(name="knotty", env=make_nav_ascii_env("knotty", 500)),
        SimulationConfig(
            name="memory_palace", env=make_nav_ascii_env("memory_palace", 200)
        ),
        SimulationConfig(name="obstacles0", env=make_nav_ascii_env("obstacles0", 100)),
        SimulationConfig(name="obstacles1", env=make_nav_ascii_env("obstacles1", 300)),
        SimulationConfig(name="obstacles2", env=make_nav_ascii_env("obstacles2", 350)),
        SimulationConfig(name="obstacles3", env=make_nav_ascii_env("obstacles3", 300)),
        SimulationConfig(
            name="radial_large", env=make_nav_ascii_env("radial_large", 1000)
        ),
        SimulationConfig(
            name="radial_mini", env=make_nav_ascii_env("radial_mini", 150)
        ),
        SimulationConfig(
            name="radial_small", env=make_nav_ascii_env("radial_small", 120)
        ),
        SimulationConfig(
            name="radial_maze", env=make_nav_ascii_env("radial_maze", 200)
        ),
        SimulationConfig(name="swirls", env=make_nav_ascii_env("swirls", 350)),
        SimulationConfig(name="thecube", env=make_nav_ascii_env("thecube", 350)),
        SimulationConfig(name="walkaround", env=make_nav_ascii_env("walkaround", 250)),
        SimulationConfig(name="wanderout", env=make_nav_ascii_env("wanderout", 500)),
        SimulationConfig(
            name="emptyspace_outofsight",
            env=make_nav_ascii_env("emptyspace_outofsight", 150),
        ),
        SimulationConfig(
            name="walls_outofsight", env=make_nav_ascii_env("walls_outofsight", 250)
        ),
        SimulationConfig(
            name="walls_withinsight", env=make_nav_ascii_env("walls_withinsight", 120)
        ),
        SimulationConfig(name="labyrinth", env=make_nav_ascii_env("labyrinth", 250)),
        SimulationConfig(name="emptyspace_sparse", env=make_emptyspace_sparse_env()),
    ]
