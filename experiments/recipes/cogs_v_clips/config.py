generalized_terrain_curriculum_args = {
    "multi_agent_pairs": {
        "num_cogs": [2, 4, 8, 12, 24],
        "positions": [["Any"], ["Any", "Any"]],
        "regeneration_rate": [2, 4, 6],
        "sizes": ["small", "medium"],
        "use_base": [True, False],
    },
    "multi_agent_triplets": {
        "num_cogs": [2, 4, 8, 12, 24],
        "positions": [["Any"], ["Any", "Any"], ["Any", "Any", "Any"]],
        "regeneration_rate": [2, 4, 6],
        "sizes": ["small", "medium"],
        "use_base": [True, False],
    },
    "multi_agent_pairs_bases": {
        "num_cogs": [2, 4, 8, 12, 24],
        "positions": [["Any"], ["Any", "Any"]],
        "regeneration_rate": [2, 4, 6],
        "sizes": ["small", "medium"],
        "use_base": [True],
    },
    "multi_agent_triplets_bases": {
        "num_cogs": [2, 4, 8, 12, 24],
        "positions": [["Any"], ["Any", "Any"], ["Any", "Any", "Any"]],
        "regeneration_rate": [2, 4, 6],
        "sizes": ["small", "medium"],
        "use_base": [True],
    },
    "multi_agent_pairs_uniform": {
        "num_cogs": [2, 4, 8, 12, 24],
        "positions": [["Any"], ["Any", "Any"]],
        "regeneration_rate": [2, 4, 6],
        "sizes": ["small", "medium"],
        "use_base": [False],
    },
    "multi_agent_triplets_uniform": {
        "num_cogs": [2, 4, 8, 12, 24],
        "positions": [["Any"], ["Any", "Any"], ["Any", "Any", "Any"]],
        "regeneration_rate": [2, 3, 4, 5],
        "sizes": ["small", "medium"],
        "use_base": [False],
    },
}

obj_distribution_by_room_size = {
    "small": [2, 4, 6],
    "medium": [5, 8, 10],
    "large": [5, 10, 20, 25],
}
