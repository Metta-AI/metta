# first element is the primary character, rest are aliases for backwards compatibility
NAME_TO_CHAR: dict[str, list[str]] = {
    # agents
    "agent.agent": ["@", "A"],
    "agent.team_1": ["1"],
    "agent.team_2": ["2"],
    "agent.team_3": ["3"],
    "agent.team_4": ["4"],
    "agent.prey": ["p"],
    "agent.predator": ["P"],
    # terrain
    "wall": ["#", "W"],
    "empty": [".", " "],
    "block": ["s"],
    # mines
    "mine_red": ["m", "r"],
    "mine_blue": ["b"],
    "mine_green": ["g"],
    # generators
    "generator_red": ["n", "R"],
    "generator_blue": ["B"],
    "generator_green": ["G"],
    # other objects
    "altar": ["_", "a"],
    "armory": ["o"],
    "lasery": ["S"],
    "lab": ["L"],
    "factory": ["F"],
    "temple": ["T"],
    "converter": ["c"],
    "C": ["C"],  # Cyclical converter
    "C1": ["1"],  # Cyclical converter 1
    "C2": ["2"],  # Cyclical converter 2
}

CHAR_TO_NAME: dict[str, str] = {}

for k, v in NAME_TO_CHAR.items():
    for c in v:
        CHAR_TO_NAME[c] = k


def grid_object_to_char(name: str) -> str:
    if name in NAME_TO_CHAR:
        return NAME_TO_CHAR[name][0]

    raise ValueError(f"Unknown object type: {name}")


def char_to_grid_object(char: str) -> str:
    if char in CHAR_TO_NAME:
        return CHAR_TO_NAME[char]

    raise ValueError(f"Unknown character: {char}")


def normalize_grid_char(char: str) -> str:
    return grid_object_to_char(char_to_grid_object(char))
