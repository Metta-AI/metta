from typing import TYPE_CHECKING, Optional

from metta.mettagrid.object_types import ObjectTypes

if TYPE_CHECKING:
    from metta.mettagrid.type_mapping import TypeMapping

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
    if char in CHAR_TO_NAME:
        return CHAR_TO_NAME[char]

    raise ValueError(f"Unknown character: {char}")


def normalize_grid_char(char: str) -> str:
    return grid_object_to_char(char_to_grid_object(char))


# Int-based encoding support using ObjectTypes
def char_to_type_id(char: str, type_mapping: Optional["TypeMapping"] = None) -> int:
    """Convert a character to a type_id using ObjectTypes or TypeMapping."""
    obj_name = char_to_grid_object(char)

    if type_mapping:
        if type_mapping.has_name(obj_name):
            return type_mapping.get_type_id(obj_name)

    # Fallback to standard ObjectTypes mapping
    standard_mappings = ObjectTypes.get_standard_mappings()
    if obj_name in standard_mappings:
        return standard_mappings[obj_name]

    # Default to empty for unknown objects
    return ObjectTypes.EMPTY


def type_id_to_char(type_id: int, type_mapping: Optional["TypeMapping"] = None) -> str:
    """Convert a type_id to a character using TypeMapping or ObjectTypes."""
    if type_mapping:
        obj_name = type_mapping.get_name(type_id)
    else:
        # Fallback to standard ObjectTypes reverse mapping
        reverse_mappings = ObjectTypes.get_reverse_mappings()
        obj_name = reverse_mappings.get(type_id, "empty")

    return grid_object_to_char(obj_name)


def grid_object_to_type_id(name: str, type_mapping: Optional["TypeMapping"] = None) -> int:
    """Convert an object name to a type_id."""
    if type_mapping:
        if type_mapping.has_name(name):
            return type_mapping.get_type_id(name)

    # Fallback to standard ObjectTypes mapping
    standard_mappings = ObjectTypes.get_standard_mappings()
    return standard_mappings.get(name, ObjectTypes.EMPTY)


def type_id_to_grid_object(type_id: int, type_mapping: Optional["TypeMapping"] = None) -> str:
    """Convert a type_id to an object name."""
    if type_mapping:
        return type_mapping.get_name(type_id)

    # Fallback to standard ObjectTypes reverse mapping
    reverse_mappings = ObjectTypes.get_reverse_mappings()
    return reverse_mappings.get(type_id, "empty")
