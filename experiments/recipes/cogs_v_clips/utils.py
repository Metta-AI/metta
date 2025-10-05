from cogames.cogs_vs_clips.stations import assembler as assembler_config
from typing import Dict, Any
from dataclasses import dataclass, field
from cogames.cogs_vs_clips.stations import (
    carbon_extractor,
    oxygen_extractor,
    germanium_extractor,
    silicon_extractor,
    chest as chest_cfg,
    charger,
)
from mettagrid.config.mettagrid_config import RecipeConfig, AgentConfig, AgentRewards


#FORAGING CONFIGS
# 1) perimeter and center objects are all assemblers, inputs nothing, outputs heart, no chest. PAIRS, TRIPLETS, QUADRUPLETS, AND COMBINED
# 2) assemblers with chests. case 1: scattered chests, equal to the number of altars. case 2: center chest


foraging_curriculum_args = {
    "assembly_lines_chests_pairs": {
        "num_cogs": [4],
        "num_assemblers": [1],
        "num_extractors": [0, 1, 2, 3, 4],
        "num_extractor_types": [1, 2, 3, 4],
        "num_chests": [1],
        "size": [4],
        "positions": [["Any", "Any"]],
    },
    "assembly_lines_chests_triplets": {
        "num_cogs": [4],
        "num_assemblers": [1],
        "num_extractors": [0, 1, 2, 3, 4],
        "num_extractor_types": [1, 2, 3, 4],
        "num_chests": [1],
        "size": [4],
        "positions": [["Any", "Any","Any"]],
    },
    "assembly_lines_chests_quadruplets": {
        "num_cogs": [4],
        "num_assemblers": [1],
        "num_extractors": [0, 1, 2, 3, 4],
        "num_extractor_types": [1, 2, 3, 4],
        "num_chests": [1],
        "size": [4],
        "positions": [["Any", "Any","Any", "Any"]],
    },
    "assembly_lines_chests_combined": {
        "num_cogs": [4],
        "num_assemblers": [1],
        "num_extractors": [0, 1, 2, 3, 4],
        "num_extractor_types": [1, 2, 3, 4],
        "num_chests": [1],
        "size": [4],
        "positions": [["Any", "Any"]],
    },
    "assembly_lines_no_chests_pairs": {
        "num_cogs": [4],
        "num_assemblers": [1],
        "num_extractors": [0, 1, 2, 3, 4],
        "num_extractor_types": [1, 2, 3, 4],
        "num_chests": [0],
        "size": [4],
        "positions": [["Any", "Any"]],
    },
    "assembly_lines_no_chests_triplets": {
        "num_cogs": [4],
        "num_assemblers": [1],
        "num_extractors": [0, 1, 2, 3, 4],
        "num_extractor_types": [1, 2, 3, 4],
        "num_chests": [0],
        "size": [4],
        "positions": [["Any", "Any","Any"]],
    },
    "assembly_lines_no_chests_quadruplets": {
        "num_cogs": [4],
        "num_assemblers": [1],
        "num_extractors": [0, 1, 2, 3, 4],
        "num_extractor_types": [1, 2, 3, 4],
        "num_chests": [0],
        "size": [4],
        "positions": [["Any", "Any","Any", "Any"]],
    },
    "assembly_lines_no_chests_combined": {
        "num_cogs": [4],
        "num_assemblers": [1],
        "num_extractors": [0, 1, 2, 3, 4],
        "num_extractor_types": [1, 2, 3, 4],
        "num_chests": [0],
        "size": [4],
        "positions": [["Any", "Any"]],
    },
    "no_chests_pairs": {
                "num_cogs": [4],
                 "num_assemblers": [5,10,20],
                 "num_chests": [0],
                 "size": [8],
                 "position": [["Any", "Any"]],
                 },
    "no_chests_triplets": {"num_cogs": [4],
                 "num_assemblers": [5,10,20],
                 "num_chests": [0],
                 "size": [8],
                 "position": [["Any", "Any","Any"]],
                 },
    "no_chests_quadruplets": {"num_cogs": [4],
                 "num_assemblers": [5,10,20],
                 "num_chests": [0],
                 "size": [8],
                 "position": [["Any", "Any","Any", "Any"]],
                 },
    "no_chests_combined": {"num_cogs": [4],
                 "num_assemblers": [5,10,20],
                 "num_chests": [0],
                 "size": [8],
                 "position": [["Any","Any"],["Any", "Any","Any"],["Any", "Any","Any","Any"]],
                 },
    "center_chests_pairs": {
                "num_cogs": [4],
                 "num_assemblers": [5,10,20],
                 "num_chests": [1, 4, 10],
                 "size": [8],
                 "position": [["Any", "Any"]],
                 },
    "center_chests_triplets": {"num_cogs": [4],
                 "num_assemblers": [5,10,20],
                 "num_chests": [1, 4, 10],
                 "size": [8],
                 "position": [["Any", "Any","Any"]],
                 },
    "center_chests_quadruplets": {"num_cogs": [4],
                 "num_assemblers": [5,10,20],
                 "num_chests": [1, 4, 10],
                 "size": [8],
                 "position": [["Any", "Any","Any", "Any"]],
                 },
    "center_chests_combined": {"num_cogs": [4],
                 "num_assemblers": [5,10,20],
                 "num_chests": [1, 4, 10],
                 "size": [8],
                 "position": [["Any","Any"],["Any", "Any","Any"],["Any", "Any","Any","Any"]],
                 },

    #EXTRACTORS
    "extractor_pairs": {
                "num_cogs": [4],
                 "num_assemblers": [4,8,12],
                 "num_chests": [0],
                 "num_extractors": [0, 4, 8, 12],
                 "num_extractor_types": [1, 2, 3, 4],
                 "size": [8],
                 "position": [["Any", "Any"]],
                 },
    "extractor_triplets": {"num_cogs": [4],
                 "num_assemblers": [4,8,12],
                "num_chests": [0],
                 "num_extractors": [0, 4, 8, 12, 16],
                 "num_extractor_types": [1, 2, 3, 4],
                 "size": [8],
                 "position": [["Any", "Any","Any"]],
                 },
    "extractor_quadruplets": {"num_cogs": [4],
                 "num_assemblers": [4,8,12],
                 "num_chests": [0],
                 "num_extractors": [0, 4, 8, 12, 16],
                 "num_extractor_types": [1, 2, 3, 4],
                 "size": [8],
                 "position": [["Any", "Any","Any", "Any"]],
                 },
    "extractor_combined": {"num_cogs": [4],
                 "num_assemblers": [4,8,12],
                 "num_chests": [0, 1, 4],
                 "num_extractors": [0, 4, 8, 12, 16],
                 "num_extractor_types": [1, 2, 3, 4],
                 "size": [8],
                 "position": [["Any","Any"],["Any", "Any","Any"],["Any", "Any","Any","Any"]],
                 },

    "extractor_chests_pairs": {
                "num_cogs": [4],
                 "num_assemblers": [4,8,12],
                 "num_chests": [0, 1, 4],
                 "num_extractors": [0, 4, 8, 12],
                 "num_extractor_types": [1, 2, 3, 4],
                 "size": [8],
                 "position": [["Any", "Any"]],
                 },
    "extractor_chests_triplets": {"num_cogs": [4],
                 "num_assemblers": [4,8,12],
                 "num_chests": [0, 1, 4],
                 "num_extractors": [0, 4, 8, 12, 16],
                 "num_extractor_types": [1, 2, 3, 4],
                 "size": [8],
                 "position": [["Any", "Any","Any"]],
                 },
    "extractor_chests_quadruplets": {"num_cogs": [4],
                 "num_assemblers": [4,8,12],
                 "num_chests": [0, 1, 4],
                 "num_extractors": [0, 4, 8, 12, 16],
                 "num_extractor_types": [1, 2, 3, 4],
                 "size": [8],
                 "position": [["Any", "Any","Any", "Any"]],
                 },
    "extractor_chests_combined": {"num_cogs": [4],
                 "num_assemblers": [4,8,12],
                 "num_chests": [0, 1, 4],
                 "num_extractors": [0, 4, 8, 12, 16],
                 "num_extractor_types": [1, 2, 3, 4],
                 "size": [8],
                 "position": [["Any","Any"],["Any", "Any","Any"],["Any", "Any","Any","Any"]],
                 },
}


num_agents_to_positions = {
    1: [["N"], ["S"], ["E"], ["W"], ["Any"]],
    2: [
        ["N", "S"],
        ["E", "W"],
        ["N", "E"],  # one agent must be north, the other agent must be east
        ["N", "W"],  # one agent must be north, the other agent must be west
        ["S", "E"],
        ["S", "W"],
    ],
    3: [
        ["N", "S", "E"],
        ["E", "W", "N"],
        ["W", "E", "S"],
        ["N", "S", "W"],
        ["S", "N", "E"],
    ],
    4: [
        ["N", "S", "E", "W"],
        ["E", "W", "N", "S"],
        ["W", "E", "S", "N"],
        ["N", "S", "W", "E"],
        ["S", "N", "E", "W"],
    ],
}

obj_distribution_by_room_size = {
    "small": [2, 4, 6],
    "medium": [5, 8, 10],
    "large": [5, 10, 20, 25],
}

size_ranges = {
    "tiny": (5, 10),  # 2 objects 2 agents max for assemblers
    "small": (7, 13),  # 9 objects, 5 agents max
    "medium": (10, 15),
    "large": (30, 40),
    "xlarge": (40, 50),
}

RESOURCES = ["carbon", "silicon", "germanium", "oxygen"]

EXTRACTORS = {
    "carbon": carbon_extractor(max_uses=0),
    "silicon": silicon_extractor(max_uses=0),
    "germanium": germanium_extractor(max_uses=0),
    "oxygen": oxygen_extractor(max_uses=0),
}


def make_assembler(inputs, outputs, positions, cooldown=1):
    assembler = assembler_config()
    assembler.recipes = []
    assembler.recipes.append(
        (
            positions,
            RecipeConfig(
                input_resources=inputs,
                output_resources=outputs,
                cooldown=cooldown,
            ),
        )
    )
    return assembler


def make_extractor(resource, inputs, outputs, position):
    extractor = EXTRACTORS[resource]

    reduced_outputs = {}
    for output in outputs:
        reduced_outputs[output] = 1
    extractor.recipes = []
    extractor.recipes.append(
        (
            position,
            RecipeConfig(
                input_resources=inputs,
                output_resources=reduced_outputs,
                cooldown=1,
            ),
        )
    )

    return extractor


def make_chest(position_deltas):
    chest = chest_cfg()
    chest.position_deltas = position_deltas
    return chest


def make_agent(
    stat_rewards,
    inventory_rewards,
    resource_limits,
    inventory_regen_amounts={},
    initial_inventory={},
    shareable_resources=["energy"],
):
    agent = AgentConfig(
        rewards=AgentRewards(
            stats=stat_rewards,
            inventory=inventory_rewards,
        ),
        default_resource_limit=1,
        resource_limits=resource_limits,
        initial_inventory=initial_inventory,
        inventory_regen_amounts=inventory_regen_amounts,
        shareable_resources=shareable_resources,
    )
    return agent



def make_charger():
    return charger()


@dataclass
class BuildCfg:
    game_objects: Dict[str, Any] = field(default_factory=dict)
    map_builder_objects: Dict[str, int] = field(default_factory=dict)
