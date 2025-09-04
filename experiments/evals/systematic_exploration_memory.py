from metta.mettagrid.builder.envs import make_navigation
from metta.mettagrid.mapgen.mapgen import MapGen
from metta.mettagrid.mettagrid_config import MettaGridConfig
from metta.sim.simulation_config import SimulationConfig


def make_systematic_exploration_memory_eval_env(
    env: MettaGridConfig,
) -> MettaGridConfig:
    """Set the heart reward to 0.333 for normalization"""
    env.game.agent.rewards.inventory["heart"] = 0.333
    return env


def make_systematic_exploration_memory_ascii_env(
    name: str, max_steps: int, border_width: int = 1
) -> MettaGridConfig:
    ascii_map = f"mettagrid/configs/maps/systematic_exploration_memory/{name}.map"
    env = make_navigation(num_agents=1)
    env.game.max_steps = max_steps
    env.game.map_builder = MapGen.Config.with_ascii_uri(
        ascii_map, border_width=border_width
    )
    return make_systematic_exploration_memory_eval_env(env)


def make_systematic_exploration_memory_eval_suite() -> list[SimulationConfig]:
    return [
        SimulationConfig(
            name="boxout",
            env=make_systematic_exploration_memory_ascii_env("boxout", 150),
        ),
        SimulationConfig(
            name="choose_wisely",
            env=make_systematic_exploration_memory_ascii_env("choose_wisely", 200),
        ),
        SimulationConfig(
            name="corners",
            env=make_systematic_exploration_memory_ascii_env("corners", 300),
        ),
        SimulationConfig(
            name="hall_of_mirrors",
            env=make_systematic_exploration_memory_ascii_env("hall_of_mirrors", 150),
        ),
        SimulationConfig(
            name="journey_home",
            env=make_systematic_exploration_memory_ascii_env("journey_home", 110),
        ),
        SimulationConfig(
            name="little_landmark_hard",
            env=make_systematic_exploration_memory_ascii_env(
                "little_landmark_hard", 100
            ),
        ),
        SimulationConfig(
            name="lobster_legs_cues",
            env=make_systematic_exploration_memory_ascii_env("lobster_legs_cues", 210),
        ),
        SimulationConfig(
            name="lobster_legs",
            env=make_systematic_exploration_memory_ascii_env("lobster_legs", 210),
        ),
        SimulationConfig(
            name="memory_swirls_hard",
            env=make_systematic_exploration_memory_ascii_env("memory_swirls_hard", 300),
        ),
        SimulationConfig(
            name="memory_swirls",
            env=make_systematic_exploration_memory_ascii_env("memory_swirls", 300),
        ),
        SimulationConfig(
            name="passing_things",
            env=make_systematic_exploration_memory_ascii_env("passing_things", 320),
        ),
        SimulationConfig(
            name="rooms", env=make_systematic_exploration_memory_ascii_env("rooms", 350)
        ),
        SimulationConfig(
            name="spacey_memory",
            env=make_systematic_exploration_memory_ascii_env("spacey_memory", 200),
        ),
        SimulationConfig(
            name="spiral_chamber",
            env=make_systematic_exploration_memory_ascii_env("spiral_chamber", 300),
        ),
        SimulationConfig(
            name="tease_small",
            env=make_systematic_exploration_memory_ascii_env("tease_small", 300),
        ),
        SimulationConfig(
            name="tease", env=make_systematic_exploration_memory_ascii_env("tease", 300)
        ),
        SimulationConfig(
            name="venture_out",
            env=make_systematic_exploration_memory_ascii_env("venture_out", 300),
        ),
        SimulationConfig(
            name="you_shall_not_pass",
            env=make_systematic_exploration_memory_ascii_env("you_shall_not_pass", 120),
        ),
    ]
