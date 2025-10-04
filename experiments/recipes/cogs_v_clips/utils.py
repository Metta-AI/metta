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

generalized_terrain_curriculum_args = {
    "multi_agent_pairs": {
        "num_cogs": [2, 4],
        "positions": [["Any"], ["Any", "Any"]],
        "regeneration_rate": [1, 2, 3],
        "sizes": ["small"],
        "use_base": [True, False],
    },
    "multi_agent_triplets": {
        "num_cogs": [3, 6],
        "positions": [["Any"], ["Any", "Any"], ["Any", "Any", "Any"]],
        "regeneration_rate": [1, 2, 3],
        "sizes": ["small"],
        "use_base": [True, False],
    },
    "multi_agent_pairs_bases": {
        "num_cogs": [2, 4],
        "positions": [["Any"], ["Any", "Any"]],
        "regeneration_rate": [1, 2, 3],
        "sizes": ["small"],
        "use_base": [True],
    },
    "multi_agent_triplets_bases": {
        "num_cogs": [3, 6],
        "positions": [["Any"], ["Any", "Any"], ["Any", "Any", "Any"]],
        "regeneration_rate": [1, 2, 3],
        "sizes": ["small"],
        "use_base": [True],
    },
    "multi_agent_pairs_uniform": {
        "num_cogs": [2, 4],
        "positions": [["Any"], ["Any", "Any"]],
        "regeneration_rate": [1, 2, 3],
        "sizes": ["small"],
        "use_base": [False],
    },
    "multi_agent_triplets_uniform": {
        "num_cogs": [3, 6],
        "positions": [["Any"], ["Any", "Any"], ["Any", "Any", "Any"]],
        "regeneration_rate": [1, 2, 3],
        "sizes": ["small"],
        "use_base": [False],
    },
}

foraging_curriculum_args = {
    "pairs": {
        "num_cogs": [2, 4],
        "num_assemblers": [2, 4, 6, 8],
        "num_extractors": [0, 1, 4, 6, 8],
        "room_size": ["small", "medium"],
        "positions": [["Any"], ["Any", "Any"]],
        "num_chests": [0, 5, 8],
    },
    "triplets": {
        "num_cogs": [3, 6],
        "num_assemblers": [2, 4, 6, 8],
        "num_extractors": [0, 1, 4, 6, 8],
        "room_size": ["small", "medium"],
        "positions": [["Any"], ["Any", "Any"], ["Any", "Any", "Any"]],
        "num_chests": [0, 5, 8],
    },
    "quadruplets": {
        "num_cogs": [4],
        "num_assemblers": [2, 4, 6, 8],
        "num_extractors": [0, 1, 4, 6, 8],
        "room_size": ["small", "medium"],
        "positions": [
            ["Any"],
            ["Any", "Any"],
            ["Any", "Any", "Any"],
            ["Any", "Any", "Any", "Any"],
        ],
        "num_chests": [0, 5, 8],
    },
}

assembly_lines_curriculum_args = {
    "easy": {
        "num_cogs": [2, 4],
        "chain_length": [2, 3],
        "room_size": ["small"],
        "positions": [["Any"], ["Any", "Any"]],
    },
    "pairs": {
        "num_cogs": [2, 4],
        "chain_length": [1, 2, 3, 4, 5],
        "room_size": ["small"],
        "positions": [["Any"], ["Any", "Any"]],
    },
    "triplets": {
        "num_cogs": [3, 6],
        "chain_length": [1, 2, 3, 4, 5],
        "room_size": ["small"],
        "positions": [["Any"], ["Any", "Any"], ["Any", "Any", "Any"]],
    },
    "quadruplets": {
        "num_cogs": [4, 8],
        "chain_length": [1, 2, 3, 4, 5],
        "room_size": ["small"],
        "positions": [
            ["Any"],
            ["Any", "Any"],
            ["Any", "Any", "Any"],
            ["Any", "Any", "Any", "Any"],
        ],
    },
}

obj_distribution_by_room_size = {
    "small": [2, 4, 6],
    "medium": [5, 8, 10],
    "large": [5, 10, 20, 25],
}

size_ranges = {
    "tiny": (5, 10),  # 2 objects 2 agents max for assemblers
    "small": (10, 20),  # 9 objects, 5 agents max
    "medium": (20, 30),
    "large": (30, 40),
    "xlarge": (40, 50),
}

RESOURCES = ["carbon", "silicon", "germanium", "oxygen"]

EXTRACTORS = {
    "carbon": carbon_extractor(),
    "silicon": silicon_extractor(),
    "germanium": germanium_extractor(),
    "oxygen": oxygen_extractor(),
}


def make_assembler(inputs, outputs, positions):
    assembler = assembler_config()
    assembler.recipes = []
    if positions == ["Any"]:
        all_positions = [
            ["Any"],
            ["Any", "Any"],
            ["Any", "Any", "Any"],
            ["Any", "Any", "Any", "Any"],
        ]
    else:
        all_positions = [positions]
    for position in all_positions:
        assembler.recipes.append(
            (
                position,
                RecipeConfig(
                    input_resources=inputs,
                    output_resources=outputs,
                    cooldown=1,
                ),
            )
        )
    return assembler


def make_extractor(resource, inputs, outputs, position):
    extractor = EXTRACTORS[resource]
    # we only want to output a single resource
    reduced_outputs = {}
    for output in outputs:
        reduced_outputs[output] = 1
    extractor.recipes = [
        (
            position,
            RecipeConfig(
                input_resources=inputs,
                output_resources=reduced_outputs,
                cooldown=1,
            ),
        )
    ]
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


def add_extractor_to_game_cfg(extractor, game_cfg):
    game_cfg.game_objects[extractor.name] = extractor
    if extractor.name not in game_cfg.map_builder_objects:
        game_cfg.map_builder_objects[extractor.name] = 1
    else:
        game_cfg.map_builder_objects[extractor.name] += 1
    return game_cfg


def make_charger():
    return charger()


@dataclass
class BuildCfg:
    game_objects: Dict[str, Any] = field(default_factory=dict)
    map_builder_objects: Dict[str, int] = field(default_factory=dict)
