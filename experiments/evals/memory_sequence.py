from typing import cast

from metta.map.mapgen import MapGen
from metta.mettagrid.config.envs import make_memory_sequence
from metta.mettagrid.mettagrid_config import ConverterConfig, EnvConfig
from metta.sim.simulation_config import SimulationConfig


def make_memory_eval_env(env: EnvConfig) -> EnvConfig:
    """Set the heart reward to 1 for normalization"""
    env.game.agent.rewards.inventory.heart = 1
    return env


def make_memory_ascii_env(
    name: str,
    max_steps: int,
    border_width: int = 1,
    num_agents: int = 1,
    reward_heart: float = 1.0,
    initial_resource_count: int = 0,
) -> EnvConfig:
    ascii_map = f"mettagrid/configs/maps/memory_sequence/{name}.map"
    env = make_memory_sequence(num_agents=num_agents)
    env.game.max_steps = max_steps
    env.game.agent.rewards.inventory.heart = reward_heart
    cast(
        ConverterConfig, env.game.objects["altar"]
    ).initial_resource_count = initial_resource_count
    env.game.map_builder = MapGen.Config.with_ascii_uri(
        ascii_map, border_width=border_width
    )
    return env


def make_memory_sequence_eval_suite() -> list[SimulationConfig]:
    return [
        SimulationConfig(
            name="access_cross",
            env=make_memory_ascii_env("access_cross", max_steps=150),
        ),
        SimulationConfig(
            name="boxout", env=make_memory_ascii_env("boxout", max_steps=350)
        ),
        SimulationConfig(
            name="choose_wisely",
            env=make_memory_ascii_env("choose_wisely", max_steps=210),
        ),
        SimulationConfig(
            name="corners", env=make_memory_ascii_env("corners", max_steps=100)
        ),
        SimulationConfig(
            name="hall_of_mirrors",
            env=make_memory_ascii_env("hall_of_mirrors", max_steps=175),
        ),
        SimulationConfig(
            name="journey_home",
            env=make_memory_ascii_env("journey_home", max_steps=300),
        ),
        SimulationConfig(
            name="little_landmark_easy",
            env=make_memory_ascii_env("little_landmark_easy", max_steps=110),
        ),
        SimulationConfig(
            name="little_landmark_hard",
            env=make_memory_ascii_env("little_landmark_hard", max_steps=140),
        ),
        SimulationConfig(
            name="lobster_legs",
            env=make_memory_ascii_env("lobster_legs", max_steps=350),
        ),
        SimulationConfig(
            name="lobster_legs_cues",
            env=make_memory_ascii_env("lobster_legs_cues", max_steps=350),
        ),
        SimulationConfig(
            name="memory_swirls",
            env=make_memory_ascii_env("memory_swirls", max_steps=500),
        ),
        SimulationConfig(
            name="memory_swirls_hard",
            env=make_memory_ascii_env("memory_swirls_hard", max_steps=600),
        ),
        SimulationConfig(
            name="passing_things",
            env=make_memory_ascii_env("passing_things", max_steps=300),
        ),
        SimulationConfig(
            name="spacey_memory",
            env=make_memory_ascii_env("spacey_memory", max_steps=250),
        ),
        SimulationConfig(
            name="tease", env=make_memory_ascii_env("tease", max_steps=500)
        ),
        SimulationConfig(
            name="tease_small", env=make_memory_ascii_env("tease_small", max_steps=375)
        ),
        SimulationConfig(
            name="venture_out", env=make_memory_ascii_env("venture_out", max_steps=375)
        ),
        SimulationConfig(
            name="which_way", env=make_memory_ascii_env("which_way", max_steps=140)
        ),
        SimulationConfig(
            name="you_shall_not_pass",
            env=make_memory_ascii_env("you_shall_not_pass", max_steps=140),
        ),
        SimulationConfig(
            name="easy_sequence",
            env=make_memory_ascii_env("easy_sequence", max_steps=90, num_agents=2),
        ),
        SimulationConfig(
            name="medium_sequence",
            env=make_memory_ascii_env("medium_sequence", max_steps=100, num_agents=2),
        ),
        SimulationConfig(
            name="hard_sequence",
            env=make_memory_ascii_env("hard_sequence", max_steps=170, num_agents=2),
        ),
        SimulationConfig(
            name="easy",
            env=make_memory_ascii_env(
                "easy",
                max_steps=42,
                num_agents=2,
                reward_heart=0.333,
                initial_resource_count=1,
            ),
        ),
        SimulationConfig(
            name="medium",
            env=make_memory_ascii_env(
                "medium",
                max_steps=58,
                num_agents=2,
                reward_heart=0.333,
                initial_resource_count=1,
            ),
        ),
        SimulationConfig(
            name="hard",
            env=make_memory_ascii_env(
                "hard",
                max_steps=70,
                num_agents=2,
                reward_heart=0.333,
                initial_resource_count=1,
            ),
        ),
    ]
