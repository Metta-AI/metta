# Valid agent subtypes - single source of truth
VALID_AGENT_SUBTYPES = {"team_1", "team_2", "team_3", "team_4", "prey", "predator", "agent"}

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
    """Convert a character or object name to its canonical form.

    This handles single-character aliases and ensures agent subtypes
    are properly formatted.
    """
    if not char:
        raise ValueError("Object name cannot be empty")

    # Check for known single-character aliases first
    if char in CHAR_TO_NAME:
        return CHAR_TO_NAME[char]

    if char in NAME_TO_CHAR:
        return char

    # Now check for invalid patterns in multi-character names
    # Reject trailing dots that would create empty tag segments
    if char.endswith("."):
        raise ValueError(f"Object name cannot end with a dot: '{char}'. This would create an empty tag segment.")

    # Check for empty segments (consecutive dots)
    if ".." in char:
        raise ValueError(f"Object name cannot contain empty segments (consecutive dots): '{char}'")

    if char == "agent":
        return "agent.agent"

    if char.startswith("agent."):
        parts = char.split(".", 2)
        if len(parts) >= 2:
            subtype = parts[1]

            # Validate that no segment is empty
            for part in parts:
                if not part:
                    raise ValueError(f"Object name cannot contain empty segments: '{char}'")

            if subtype not in VALID_AGENT_SUBTYPES:
                return f"agent.agent.{char[6:]}"

    # For single-character unknown glyphs, raise an error as they're likely typos
    if len(char) == 1 and char not in CHAR_TO_NAME and char not in NAME_TO_CHAR:
        raise ValueError(f"Unknown single character glyph: '{char}'. This is likely a typo in the ASCII map.")

    return char


def normalize_grid_char(char: str) -> str:
    return grid_object_to_char(char_to_grid_object(char))
