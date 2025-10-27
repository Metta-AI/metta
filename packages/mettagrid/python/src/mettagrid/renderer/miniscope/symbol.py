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
    """Resolve the display symbol for an object dictionary."""
    # Prefer type_name field if available, otherwise look up by type ID
    if "type_name" in obj:
        type_name = obj["type_name"]
    else:
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
