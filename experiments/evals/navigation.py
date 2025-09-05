from metta.mettagrid.builder.envs import make_navigation
from metta.mettagrid.mapgen.mapgen import MapGen
from metta.mettagrid.mapgen.scenes.mean_distance import MeanDistance
from metta.mettagrid.mettagrid_config import MettaGridConfig
from metta.sim.simulation_config import SimulationConfig


def make_nav_eval_env(env: MettaGridConfig) -> MettaGridConfig:
    """Set the heart reward to 0.333 for normalization"""
    env.game.agent.rewards.inventory["heart"] = 0.333
    return env


def replace_objects_with_altars(name: str) -> str:
    ascii_map = f"mettagrid/configs/maps/navigation_sequence/{name}.map"

    with open(ascii_map, "r") as f:
        map_content = f.read()

    return map_content.replace("n", "_").replace("m", "_")


def make_nav_ascii_env(
    name: str,
    max_steps: int,
    border_width: int = 6,
    num_agents=1,
    num_instances=4,
) -> MettaGridConfig:
    # we re-use nav sequence maps, but replace all objects with altars
    ascii_map = replace_objects_with_altars(name)

    env = make_navigation(num_agents=num_agents * num_instances)
    env.game.max_steps = max_steps
    env.game.map_builder = MapGen.Config(
        instances=num_instances,
        border_width=6,
        instance_border_width=3,
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
        SimulationConfig(
            name="boxout",
            env=make_nav_ascii_env("boxout", 150),
        ),
        SimulationConfig(
            name="choose_wisely",
            env=make_nav_ascii_env("choose_wisely", 200),
        ),
        SimulationConfig(
            name="corners",
            env=make_nav_ascii_env("corners", 300),
        ),
        SimulationConfig(
            name="hall_of_mirrors",
            env=make_nav_ascii_env("hall_of_mirrors", 150),
        ),
        SimulationConfig(
            name="journey_home",
            env=make_nav_ascii_env("journey_home", 110),
        ),
        SimulationConfig(
            name="little_landmark_hard",
            env=make_nav_ascii_env("little_landmark_hard", 100),
        ),
        SimulationConfig(
            name="lobster_legs_cues",
            env=make_nav_ascii_env("lobster_legs_cues", 210),
        ),
        SimulationConfig(
            name="lobster_legs",
            env=make_nav_ascii_env("lobster_legs", 210),
        ),
        SimulationConfig(
            name="memory_swirls_hard",
            env=make_nav_ascii_env("memory_swirls_hard", 300),
        ),
        SimulationConfig(
            name="memory_swirls",
            env=make_nav_ascii_env("memory_swirls", 300),
        ),
        SimulationConfig(
            name="passing_things",
            env=make_nav_ascii_env("passing_things", 320),
        ),
        SimulationConfig(name="rooms", env=make_nav_ascii_env("rooms", 350)),
        SimulationConfig(
            name="spacey_memory",
            env=make_nav_ascii_env("spacey_memory", 200),
        ),
        SimulationConfig(
            name="spiral_chamber",
            env=make_nav_ascii_env("spiral_chamber", 300),
        ),
        SimulationConfig(
            name="tease_small",
            env=make_nav_ascii_env("tease_small", 300),
        ),
        SimulationConfig(name="tease", env=make_nav_ascii_env("tease", 300)),
        SimulationConfig(
            name="venture_out",
            env=make_nav_ascii_env("venture_out", 300),
        ),
        SimulationConfig(
            name="you_shall_not_pass",
            env=make_nav_ascii_env("you_shall_not_pass", 120),
        ),
        SimulationConfig(
            name="easy_memory",
            env=make_nav_ascii_env("easy_sequence", 42, num_agents=2, num_instances=2),
        ),
        SimulationConfig(
            name="medium_memory",
            env=make_nav_ascii_env(
                "medium_sequence", 58, num_agents=2, num_instances=2
            ),
        ),
        SimulationConfig(
            name="hard_memory",
            env=make_nav_ascii_env("hard_sequence", 70, num_agents=2, num_instances=2),
        ),
    ]
