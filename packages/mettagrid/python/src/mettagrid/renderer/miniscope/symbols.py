"""Emoji symbols for miniscope rendering."""

from typing import Dict

# Using emoji that are consistently rendered as double-width characters
# These are selected to ensure visual clarity and consistent alignment
MINISCOPE_SYMBOLS: Dict[str, str] = {
    # Basic terrain
    "wall": "⬛",  # Wall/barrier (#) - dark square
    "empty": "⬜",  # Empty space (white square for visibility)
    # Agents
    "agent": "🤖",  # Default agent
    "agent.agent": "🤖",  # Standard agent (A)
    "agent.team_1": "🔵",  # Team 1 (1)
    "agent.team_2": "🔴",  # Team 2 (2)
    "agent.team_3": "🟢",  # Team 3 (3)
    "agent.team_4": "🟡",  # Team 4 (4)
    "agent.prey": "🐰",  # Prey agent (Ap)
    "agent.predator": "🦁",  # Predator agent (AP)
    # Resources and Items
    "mine_red": "🔺",  # Red mine (r)
    "mine_blue": "🔷",  # Blue mine (b)
    "mine_green": "💚",  # Green mine (g)
    # Generators/Converters
    "generator": "⚡",  # Generic generator (n)
    "generator_red": "🔋",  # Red generator (R)
    "generator_blue": "🔌",  # Blue generator (B)
    "generator_green": "🟢",  # Green generator (G)
    "converter": "🔄",  # Converter (c)
    # Special Objects
    "altar": "🎯",  # Altar/shrine (a) - using target instead of torii
    "block": "📦",  # Movable block (s)
    "lasery": "🟥",  # Laser weapon ('L' in ASCII maps)
    "factory": "🟪",  # Factory
    "lab": "🔵",  # Laboratory
    "temple": "🟨",  # Temple
    # Markers and indicators
    "marker": "🟠",  # Location marker ('m' in ASCII maps)
    "shrine": "🟣",  # Shrine/checkpoint ('s' in ASCII maps)
    "launcher": "⬛",  # Launcher
    # CoGames Stations
    "charger": "⚡",  # Energy charger
    "carbon_extractor": "⚫",  # Carbon extraction (ready)
    "oxygen_extractor": "🔵",  # Oxygen extraction (ready)
    "germanium_extractor": "🟣",  # Germanium extraction (ready)
    "silicon_extractor": "🔷",  # Silicon extraction (ready)
    "silicon_ex_dep": "🔹",  # Silicon extractor (cooldown)
    "germanium_ex_dep": "🟪",  # Germanium extractor (cooldown)
    "oxygen_ex_dep": "⬜",  # Oxygen extractor (cooldown)
    "carbon_ex_dep": "⬛",  # Carbon extractor (cooldown)
    "chest": "📦",  # Storage chest
    "assembler": "🔄",  # Assembler station
    # Fallback
    "?": "❓",  # Unknown object
    # Selection cursor
    "cursor": "🎯",  # Selection target cursor
}

# Colored squares for agent IDs 0-9 (consistent width)
AGENT_SQUARES = ["🟦", "🟧", "🟩", "🟨", "🟪", "🟥", "🟫", "⬛", "🟦", "🟧"]


def get_symbol_for_object(obj: dict, object_type_names: list[str]) -> str:
    """Get the emoji symbol for an object.

    Args:
        obj: Object dict with 'type' and optional 'agent_id'
        object_type_names: List mapping type IDs to names

    Returns:
        Emoji symbol string
    """
    type_name = object_type_names[obj["type"]]

    # Handle numbered agents specially
    if type_name.startswith("agent"):
        agent_id = obj.get("agent_id")
        if agent_id is not None and 0 <= agent_id < 10:
            return AGENT_SQUARES[agent_id]

    # Try full type name first, then base type
    if type_name in MINISCOPE_SYMBOLS:
        return MINISCOPE_SYMBOLS[type_name]

    base = type_name.split(".")[0]
    return MINISCOPE_SYMBOLS.get(base, MINISCOPE_SYMBOLS["?"])
