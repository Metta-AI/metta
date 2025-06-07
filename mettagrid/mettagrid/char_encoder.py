# first element is the primary character, rest are aliases
NAME_TO_CHAR: dict[str, list[str]] = {
    # agents
    "agent.agent": ["@", "A"],
    "agent.team_1": ["1"],
    "agent.team_2": ["2"],
    "agent.team_3": ["3"],
    "agent.team_4": ["4"],
    # generators
    "generator": ["n", "⚙"],
    "generator.red": ["R"],
    "generator.blue": ["B"],
    "generator.green": ["G"],
    # mines
    "mine": ["m"],
    "mine.red": ["r"],
    "mine.blue": ["b"],
    "mine.green": ["g"],
    # other objects
    "altar": ["a", "⛩"],
    "converter": ["c"],
    "wall": ["#", "W", "🧱"],
    "empty": [".", " "],
    "block": ["s"],
    "lasery": ["L"],
    "armory": ["o"],
    "factory": ["🏭"],
    "lab": ["🔬"],
    "temple": ["🏰"],
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
