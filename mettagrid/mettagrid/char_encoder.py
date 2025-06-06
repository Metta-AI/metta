# first element is the primary character, rest are aliases
NAME_TO_CHAR: dict[str, list[str]] = {
    # agents
    "agent.agent": ["@", "A"],
    "agent.prey": ["Ap"],
    "agent.predator": ["AP"],
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
    "wall": ["W", "#", "🧱"],
    "empty": [".", " "],
    "block": ["s"],
    "lasery": ["L"],
    "factory": ["🏭"],
    "lab": ["🔬"],
    "temple": ["🏰"],
}

CHAR_TO_NAME: dict[str, str] = {}

for k, v in NAME_TO_CHAR.items():
    for c in v:
        CHAR_TO_NAME[c] = k
