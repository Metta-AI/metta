# Agent-specific colored squares for agent IDs 0-9 (consistent width)
AGENT_SQUARES = ["🟦", "🟧", "🟩", "🟨", "🟪", "🟥", "🟫", "⬛", "🟦", "🟧"]


DEFAULT_SYMBOL_MAP = {
    # Terrain
    "wall": "⬛",
    "empty": "⬜",
    "block": "📦",
    # Agents
    "agent": "🤖",
    "agent.agent": "🤖",
    "agent.team_1": "🔵",
    "agent.team_2": "🔴",
    "agent.team_3": "🟢",
    "agent.team_4": "🟡",
    "agent.prey": "🐰",
    "agent.predator": "🦁",
    # UI elements
    "cursor": "🎯",
    "?": "❓",
}


def get_symbol_for_object(obj: dict, object_type_names: list[str], symbol_map: dict[str, str]) -> str:
    """Get the emoji symbol for an object.

    Args:
        obj: Object dict with 'type' and optional 'agent_id'
        object_type_names: List mapping type IDs to names
        symbol_map: Map from object type names to render symbols

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
    if type_name in symbol_map:
        return symbol_map[type_name]

    base = type_name.split(".")[0]
    return symbol_map.get(base, symbol_map.get("?", "❓"))
