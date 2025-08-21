from typing import Callable

from metta.experiments.evals.navigation import make_ascii_env
from metta.mettagrid.mettagrid_config import EnvConfig
from metta.sim.simulation_config import SimulationConfig


def make_boxout_env() -> EnvConfig:
    return make_ascii_env(
        max_steps=150,
        ascii_map="mettagrid/configs/maps/systematic_exploration_memory/boxout.map",
    )


def make_choose_wisely_env() -> EnvConfig:
    return make_ascii_env(
        max_steps=200,
        ascii_map="mettagrid/configs/maps/systematic_exploration_memory/choose_wisely.map",
    )


def make_corners_env() -> EnvConfig:
    return make_ascii_env(
        max_steps=300,
        ascii_map="mettagrid/configs/maps/systematic_exploration_memory/corners.map",
    )


def make_hall_of_mirrors_env() -> EnvConfig:
    return make_ascii_env(
        max_steps=150,
        ascii_map="mettagrid/configs/maps/systematic_exploration_memory/hall_of_mirrors.map",
    )


def make_journey_home_env() -> EnvConfig:
    return make_ascii_env(
        max_steps=110,
        ascii_map="mettagrid/configs/maps/systematic_exploration_memory/journey_home.map",
    )


def make_little_landmark_hard_env() -> EnvConfig:
    return make_ascii_env(
        max_steps=100,
        ascii_map="mettagrid/configs/maps/systematic_exploration_memory/little_landmark_hard.map",
    )


def make_lobster_legs_cues_env() -> EnvConfig:
    return make_ascii_env(
        max_steps=210,
        ascii_map="mettagrid/configs/maps/systematic_exploration_memory/maze.map",
    )


def make_lobster_legs_env() -> EnvConfig:
    return make_ascii_env(
        max_steps=210,
        ascii_map="mettagrid/configs/maps/systematic_exploration_memory/lobster_legs.map",
    )


def make_memory_swirls_hard_env() -> EnvConfig:
    return make_ascii_env(
        max_steps=300,
        ascii_map="mettagrid/configs/maps/systematic_exploration_memory/memory_swirls_hard.map",
    )


def make_memory_swirls_env() -> EnvConfig:
    return make_ascii_env(
        max_steps=300,
        ascii_map="mettagrid/configs/maps/systematic_exploration_memory/memory_swirls.map",
    )


def make_passing_things() -> EnvConfig:
    return make_ascii_env(
        max_steps=320,
        ascii_map="mettagrid/configs/maps/systematic_exploration_memory/passing_things.map",
    )


def make_rooms_env() -> EnvConfig:
    return make_ascii_env(
        max_steps=350,
        ascii_map="mettagrid/configs/maps/systematic_exploration_memory/rooms.map",
    )


def make_spacey_memory_env() -> EnvConfig:
    return make_ascii_env(
        max_steps=200,
        ascii_map="mettagrid/configs/maps/systematic_exploration_memory/spacey_memory.map",
    )


def make_spiral_chamber_env() -> EnvConfig:
    return make_ascii_env(
        max_steps=300,
        ascii_map="mettagrid/configs/maps/systematic_exploration_memory/spiral_chamber.map",
    )


def make_tease_small_env() -> EnvConfig:
    return make_ascii_env(
        max_steps=300,
        ascii_map="mettagrid/configs/maps/systematic_exploration_memory/tease_small.map",
    )


def make_tease_env() -> EnvConfig:
    return make_ascii_env(
        max_steps=300,
        ascii_map="mettagrid/configs/maps/systematic_exploration_memory/tease.map",
    )


def make_venture_out_env() -> EnvConfig:
    return make_ascii_env(
        max_steps=300,
        ascii_map="mettagrid/configs/maps/systematic_exploration_memory/venture_out.map",
    )


def make_you_shall_not_pass_env() -> EnvConfig:
    return make_ascii_env(
        max_steps=120,
        ascii_map="mettagrid/configs/maps/systematic_exploration_memory/you_shall_not_pass.map",
    )


def make_systematic_exploration_memory_eval_suite() -> list[SimulationConfig]:
    env_config_makers: list[Callable[[], EnvConfig]] = [
        make_boxout_env,
        make_choose_wisely_env,
        make_corners_env,
        make_hall_of_mirrors_env,
        make_journey_home_env,
        make_little_landmark_hard_env,
        make_lobster_legs_cues_env,
        make_lobster_legs_env,
        make_memory_swirls_hard_env,
        make_memory_swirls_env,
        make_passing_things,
        make_rooms_env,
        make_spacey_memory_env,
        make_spiral_chamber_env,
        make_tease_small_env,
        make_tease_env,
        make_venture_out_env,
        make_you_shall_not_pass_env,
    ]

    def fn_to_sim_name(fn: Callable[[], EnvConfig]) -> str:
        return fn.__name__.replace("make_", "").replace("_env", "")

    return [
        SimulationConfig(
            name=fn_to_sim_name(env_config_maker),
            env=env_config_maker(),
        )
        for env_config_maker in env_config_makers
    ]
