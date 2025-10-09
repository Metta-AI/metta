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


# FORAGING CONFIGS
# 1) perimeter and center objects are all assemblers, inputs nothing, outputs heart, no chest. PAIRS, TRIPLETS, QUADRUPLETS, AND COMBINED
# 2) assemblers with chests. case 1: scattered chests, equal to the number of altars. case 2: center chest


num_agents_to_positions = {
    1: [["N"], ["S"], ["E"], ["W"], ["Any"]],
    2: [
        ["N", "S"],
        ["E", "W"],
        ["N", "E"],  # one agent must be north, the other agent must be east
        ["N", "W"],  # one agent must be north, the other agent must be west
        ["S", "E"],
        ["S", "W"],
        ["Any", "Any"],
    ],
    3: [
        ["N", "S", "E"],
        ["E", "W", "N"],
        ["W", "E", "S"],
        ["N", "S", "W"],
        ["S", "N", "E"],
        ["Any", "Any", "Any"],
    ],
    4: [
        ["N", "S", "E", "W"],
        ["E", "W", "N", "S"],
        ["W", "E", "S", "N"],
        ["N", "S", "W", "E"],
        ["S", "N", "E", "W"],
        ["Any", "Any", "Any", "Any"],
    ],
}

foraging_curriculum_args = {
    "assembly_lines_chests_pairs": {
        "num_cogs": [2],
        "num_assemblers": [1],
        "num_extractors": [0, 1, 2],
        "num_extractor_types": [1, 2, 3, 4],
        "num_chests": [1],
        "size": [6, 10, 12],
        "assembler_positions": num_agents_to_positions[1] + num_agents_to_positions[2],
        "extractor_positions": num_agents_to_positions[1] + num_agents_to_positions[2],
    },
    "assembly_lines_chests_triplets": {
        "num_cogs": [3],
        "num_assemblers": [1],
        "num_extractors": [0, 1, 2],
        "num_extractor_types": [1, 2, 3, 4],
        "num_chests": [1],
        "size": [6, 10, 12],
        "assembler_positions": num_agents_to_positions[1]
        + num_agents_to_positions[2]
        + num_agents_to_positions[3],
        "extractor_positions": num_agents_to_positions[1]
        + num_agents_to_positions[2]
        + num_agents_to_positions[3],
    },
    "assembly_lines_chests_quadruplets": {
        "num_cogs": [4],
        "num_assemblers": [1],
        "num_extractors": [0, 1, 2],
        "num_extractor_types": [1, 2, 3, 4],
        "num_chests": [1],
        "size": [6, 10, 12],
        "assembler_positions": num_agents_to_positions[1]
        + num_agents_to_positions[2]
        + num_agents_to_positions[3]
        + num_agents_to_positions[4],
        "extractor_positions": num_agents_to_positions[1]
        + num_agents_to_positions[2]
        + num_agents_to_positions[3]
        + num_agents_to_positions[4],
    },
    "extractor_chests_pairs": {
        "num_cogs": [2],
        "num_assemblers": [1, 3, 5, 8],
        "num_chests": [1],
        "size": [15, 20, 30],
        "num_extractors": [0, 4, 8],
        "num_extractor_types": [1, 2, 3],
        "assembler_positions": num_agents_to_positions[1] + num_agents_to_positions[2],
    },
    "extractor_chests_triplets": {
        "num_cogs": [3],
        "num_assemblers": [1, 3, 5, 8],
        "num_chests": [1],
        "num_extractors": [0, 4, 8],
        "num_extractor_types": [1, 2, 3],
        "size": [15, 20, 30],
        "assembler_positions": num_agents_to_positions[1]
        + num_agents_to_positions[2]
        + num_agents_to_positions[3],
    },
    "extractor_chests_quadruplets": {
        "num_cogs": [4],
        "num_assemblers": [1, 3, 5, 8],
        "num_chests": [1],
        "num_extractors": [0, 4, 8],
        "num_extractor_types": [1, 2, 3],
        "size": [15, 20, 30],
        "assembler_positions": num_agents_to_positions[1]
        + num_agents_to_positions[2]
        + num_agents_to_positions[3]
        + num_agents_to_positions[4],
    },
}

generalized_terrain_curriculum_args = {
    "test": {
        "num_cogs": [4],
        "positions": [["N", "S"]],
        "regeneration_rate": [2],
        "use_base": [True],
        "sizes": ["small"],
    }
}

assembly_lines_curriculum_args = {
    "test": {
        "num_cogs": [4],
        "chain_length": [1],
        "room_size": ["small"],
        "positions": [["N", "S"]],
    }
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
                **{
                    "input_resources": inputs,
                    "output_resources": outputs,
                    "cooldown": cooldown,
                }
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
                **{
                    "input_resources": inputs,
                    "output_resources": reduced_outputs,
                    "cooldown": 1,
                }
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
            stats_max={"chest.heart.amount": 10},
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
