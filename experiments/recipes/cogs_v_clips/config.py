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
        "num_cogs": [2, 4, 8, 12],
        "positions": [["Any"], ["Any", "Any"]],
        "regeneration_rate": [2, 3, 4],
        "sizes": ["small", "medium"],
        "use_base": [True, False],
    },
    "multi_agent_triplets": {
        "num_cogs": [2, 4, 8, 12],
        "positions": [["Any"], ["Any", "Any"], ["Any", "Any", "Any"]],
        "regeneration_rate": [2, 3, 4],
        "sizes": ["small", "medium"],
        "use_base": [True, False],
    },
    "multi_agent_pairs_bases": {
        "num_cogs": [2, 4, 8, 12],
        "positions": [["Any"], ["Any", "Any"]],
        "regeneration_rate": [2, 3, 4],
        "sizes": ["small", "medium"],
        "use_base": [True],
    },
    "multi_agent_triplets_bases": {
        "num_cogs": [2, 4, 8, 12],
        "positions": [["Any"], ["Any", "Any"], ["Any", "Any", "Any"]],
        "regeneration_rate": [2, 3, 4],
        "sizes": ["small", "medium"],
        "use_base": [True],
    },
    "multi_agent_pairs_uniform": {
        "num_cogs": [2, 4, 8, 12],
        "positions": [["Any"], ["Any", "Any"]],
        "regeneration_rate": [2, 3, 4],
        "sizes": ["small", "medium"],
        "use_base": [False],
    },
    "multi_agent_triplets_uniform": {
        "num_cogs": [2, 4, 8, 12],
        "positions": [["Any"], ["Any", "Any"], ["Any", "Any", "Any"]],
        "regeneration_rate": [2, 3, 4],
        "sizes": ["small", "medium"],
        "use_base": [False],
    },
}

foraging_curriculum_args = {
    "all": {
        "num_cogs": [2, 4, 8, 12, 24],
        "num_assemblers": [1, 3, 5],
        "num_extractors": [0, 1, 4, 10],
        "room_size": ["small", "medium", "large"],
        "positions": num_agents_to_positions[1]
        + num_agents_to_positions[2]
        + num_agents_to_positions[3],
        "num_chests": [0, 5, 8, 15],
    },
    "pairs": {
        "num_cogs": [2, 4, 8, 12, 24],
        "num_assemblers": [1, 3, 5],
        "num_extractors": [0, 1, 4, 10],
        "room_size": ["small", "medium", "large"],
        "positions": num_agents_to_positions[2],
        "num_chests": [0, 5, 8, 15],
    },
    "triplets": {
        "num_cogs": [3, 6, 12, 24],
        "num_assemblers": [1, 3, 5],
        "num_extractors": [0, 1, 4, 10],
        "room_size": ["small", "medium", "large"],
        "positions": num_agents_to_positions[3],
        "num_chests": [0, 5, 8, 15],
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
