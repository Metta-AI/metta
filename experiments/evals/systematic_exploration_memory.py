from metta.experiments.evals.navigation import ascii_eval_env
from metta.sim.simulation_config import SimulationConfig


def make_systematic_exploration_memory_eval_suite() -> list[SimulationConfig]:
    return [
        SimulationConfig(name="boxout", env=ascii_eval_env("boxout", 150)),
        SimulationConfig(
            name="choose_wisely", env=ascii_eval_env("choose_wisely", 200)
        ),
        SimulationConfig(name="corners", env=ascii_eval_env("corners", 300)),
        SimulationConfig(
            name="hall_of_mirrors", env=ascii_eval_env("hall_of_mirrors", 150)
        ),
        SimulationConfig(name="journey_home", env=ascii_eval_env("journey_home", 110)),
        SimulationConfig(
            name="little_landmark_hard", env=ascii_eval_env("little_landmark_hard", 100)
        ),
        SimulationConfig(
            name="lobster_legs_cues", env=ascii_eval_env("lobster_legs_cues", 210)
        ),
        SimulationConfig(name="lobster_legs", env=ascii_eval_env("lobster_legs", 210)),
        SimulationConfig(
            name="memory_swirls_hard", env=ascii_eval_env("memory_swirls_hard", 300)
        ),
        SimulationConfig(
            name="memory_swirls", env=ascii_eval_env("memory_swirls", 300)
        ),
        SimulationConfig(
            name="passing_things", env=ascii_eval_env("passing_things", 320)
        ),
        SimulationConfig(name="rooms", env=ascii_eval_env("rooms", 350)),
        SimulationConfig(
            name="spacey_memory", env=ascii_eval_env("spacey_memory", 200)
        ),
        SimulationConfig(
            name="spiral_chamber", env=ascii_eval_env("spiral_chamber", 300)
        ),
        SimulationConfig(name="tease_small", env=ascii_eval_env("tease_small", 300)),
        SimulationConfig(name="tease", env=ascii_eval_env("tease", 300)),
        SimulationConfig(name="venture_out", env=ascii_eval_env("venture_out", 300)),
        SimulationConfig(
            name="you_shall_not_pass", env=ascii_eval_env("you_shall_not_pass", 120)
        ),
    ]
