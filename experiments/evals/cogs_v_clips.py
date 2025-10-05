from cogames.cogs_vs_clips.scenarios import make_game
from mettagrid.mapgen.mapgen import MapGen
import random
from experiments.evals.foraging import make_foraging_eval_suite
from metta.sim.simulation_config import SimulationConfig


terrain_evals = {
    "12_agent_pairs_small_uniform": {
        "num_cogs": 12,
        "positions": ["Any", "Any"],
        "regeneration_rate": 2,
        "sizes": "small",
        "use_base": False,
    },
    "12_agent_triplets_small_uniform": {
        "num_cogs": 12,
        "positions": ["Any", "Any", "Any"],
        "regeneration_rate": 2,
        "sizes": "small",
        "use_base": False,
    },
    "12_agent_pairs_medium_uniform": {
        "num_cogs": 12,
        "positions": ["Any", "Any"],
        "regeneration_rate": 2,
        "sizes": "medium",
        "use_base": False,
    },
    "12_agent_triplets_medium_uniform": {
        "num_cogs": 12,
        "positions": ["Any", "Any", "Any"],
        "regeneration_rate": 2,
        "sizes": "medium",
        "use_base": False,
    },
    "12_agent_pairs_small_bases": {
        "num_cogs": 12,
        "positions": ["Any", "Any"],
        "regeneration_rate": 2,
        "sizes": "small",
        "use_base": True,
    },
    "12_agent_triplets_small_bases": {
        "num_cogs": 12,
        "positions": ["Any", "Any", "Any"],
        "regeneration_rate": 2,
        "sizes": "small",
        "use_base": True,
    },
    "12_agent_pairs_medium_bases": {
        "num_cogs": 12,
        "positions": ["Any", "Any"],
        "regeneration_rate": 2,
        "sizes": "medium",
        "use_base": True,
    },
    "12_agent_triplets_medium_bases": {
        "num_cogs": 12,
        "positions": ["Any", "Any", "Any"],
        "regeneration_rate": 2,
        "sizes": "medium",
        "use_base": True,
    },
}


def make_ascii_eval_env(map_file, dir="packages/cogames/src/cogames/maps/"):
    env = make_game()
    num_instances = 6
    env.game.map_builder = MapGen.Config(
        instances=num_instances,
        instance=MapGen.Config.with_ascii_uri(f"{dir}/{map_file}"),
    )
    env.game.max_steps = 250
    env.game.num_agents = 24

    return env


def make_terrain_eval_env(
    num_cogs=4,
    positions=["Any", "Any"],
    regeneration_rate=2,
    use_base=True,
    sizes="small",
):
    from experiments.recipes.cogs_v_clips.generalized_terrain import (
        GeneralizedTerrainTaskGenerator,
    )

    task_generator = GeneralizedTerrainTaskGenerator(
        config=GeneralizedTerrainTaskGenerator.Config(
            num_cogs=[num_cogs],
            positions=[positions],
            regeneration_rate=[regeneration_rate],
            use_base=[use_base],
            sizes=[sizes],
        )
    )
    return task_generator.get_task(random.randint(0, 1000000))


ascii_evals = {
    "training_facility_open_1": "training_facility_open_1.map",
    "training_facility_open_2": "training_facility_open_2.map",
    "training_facility_open_3": "training_facility_open_3.map",
    "training_facility_tight_4": "training_facility_tight_4.map",
    "training_facility_tight_5": "training_facility_tight_5.map",
    "machina_100_stations": "machina_100_stations.map",
    "machina_200_stations": "machina_200_stations.map",
    "cave_base_50": "cave_base_50.map",
}


def make_cogs_v_clips_ascii_evals() -> list[SimulationConfig]:
    return [
        SimulationConfig(
            suite="cogs_v_clips", name=name, env=make_ascii_eval_env(mapfile)
        )
        for name, mapfile in ascii_evals.items()
    ]


def make_cogs_v_clips_terrain_evals() -> list[SimulationConfig]:
    return [
        SimulationConfig(
            suite="cogs_v_clips", name=name, env=make_terrain_eval_env(**kwargs)
        )
        for name, kwargs in terrain_evals.items()
    ]


def make_cogs_v_clips_evals() -> list[SimulationConfig]:
    return (
        make_cogs_v_clips_ascii_evals()
        + make_cogs_v_clips_terrain_evals()
        + make_foraging_eval_suite()
    )
