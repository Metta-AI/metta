from experiments.recipes.cogs_v_clips.foraging import (
    make_env,
)
from metta.sim.simulation_config import SimulationConfig

evals = {
    "extractor_assembler_quadruplet": {
        "num_cogs": 4,
        "num_assemblers": 1,
        "num_extractors": 4,
        "num_extractor_types": 4,
        "num_chests": 1,
        "size": 10,
        "assembler_position": ["Any", "Any", "Any", "Any"],
    },
    "extractor_assembler_chest_pair": {
        "num_cogs": 4,
        "num_assemblers": 1,
        "num_extractors": 2,
        "num_extractor_types": 2,
        "num_chests": 1,
        "size": 10,
        "assembler_position": ["Any", "Any", "Any"],
    },
    "extractor_assembler_chest_triplet": {
        "num_cogs": 4,
        "num_assemblers": 1,
        "num_extractors": 3,
        "num_extractor_types": 3,
        "num_chests": 1,
        "size": 10,
        "assembler_position": ["Any", "Any", "Any"],
    },
    "extractor_assembler_chest_quadruplet": {
        "num_cogs": 4,
        "num_assemblers": 1,
        "num_extractors": 4,
        "num_extractor_types": 4,
        "num_chests": 1,
        "size": 10,
        "assembler_position": ["Any", "Any", "Any", "Any"],
    },
    "assembler_foraging_pairs": {
        "num_cogs": 4,
        "num_assemblers": 10,
        "num_chests": 1,
        "size": 14,
        "assembler_position": ["Any", "Any"],
    },
    "assembler_foraging_triplets": {
        "num_cogs": 4,
        "num_assemblers": 10,
        "num_chests": 1,
        "size": 14,
        "assembler_position": ["Any", "Any", "Any"],
    },
    "assembler_foraging_quadruplets": {
        "num_cogs": 4,
        "num_assemblers": 10,
        "num_chests": 1,
        "size": 14,
        "assembler_position": ["Any", "Any", "Any", "Any"],
    },
    "assembler_foraging_chests_pairs": {
        "num_cogs": 4,
        "num_assemblers": 10,
        "num_chests": 4,
        "size": 12,
        "assembler_position": ["Any", "Any"],
    },
    "assembler_foraging_chests_triplets": {
        "num_cogs": 4,
        "num_assemblers": 10,
        "num_chests": 4,
        "size": 12,
        "assembler_position": ["Any", "Any", "Any"],
    },
    "assembler_foraging_chests_quadruplets": {
        "num_cogs": 4,
        "num_assemblers": 10,
        "num_chests": 4,
        "size": 12,
        "assembler_position": ["Any", "Any", "Any", "Any"],
    },
}


def make_foraging_eval_suite() -> list[SimulationConfig]:
    return [
        SimulationConfig(name=n, suite="foraging", env=make_env(**args))
        for n, args in evals.items()
    ]
