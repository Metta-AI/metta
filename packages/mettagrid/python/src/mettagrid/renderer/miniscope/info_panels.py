"""Info panel builders for miniscope interactive mode."""

from typing import Dict, List, Optional

import numpy as np


def build_agent_info_panel(
    grid_objects: Dict[int, dict],
    object_type_names: list[str],
    selected_agent: Optional[int],
    resource_names: List[str],
    panel_height: int,
    total_rewards: np.ndarray,
) -> List[str]:
    """Build info panel showing selected agent's inventory.

    Args:
        grid_objects: Dictionary of grid objects
        object_type_names: List mapping type IDs to names
        selected_agent: Selected agent ID (None if no selection)
        resource_names: List of resource names
        panel_height: Height of the panel in lines
        total_rewards: Array of cumulative rewards for each agent

    Returns:
        List of strings, one per line
    """
    lines = []

    if selected_agent is None:
        lines.append("┌─ Info ──────────────┐")
        lines.append("│ No agent            │")
        lines.append("│ selected            │")
        lines.append("└─────────────────────┘")
    else:
        # Find the agent in grid_objects
        agent_obj = None
        for obj in grid_objects.values():
            if obj.get("agent_id") == selected_agent:
                agent_obj = obj
                break

        if agent_obj is None:
            lines.append("┌─ Info ──────────────┐")
            lines.append(f"│ Agent {selected_agent}            │")
            lines.append("│ (not found)         │")
            lines.append("└─────────────────────┘")
        else:
            # Build inventory display
            reward = total_rewards[selected_agent] if selected_agent < len(total_rewards) else 0.0
            lines.append(f"┌─ Agent {selected_agent} ───────────┐")
            lines.append(f"│ Reward: {reward:11.1f} │")
            lines.append("├─ Inventory ─────────┤")

            inventory = agent_obj.get("inventory", {})
            if not inventory or not isinstance(inventory, dict):
                lines.append("│ (empty)             │")
            else:
                # Show resources with amounts (inventory is dict of resource_id -> amount)
                has_items = False
                for resource_id, amount in sorted(inventory.items()):
                    if resource_id < len(resource_names) and amount > 0:
                        resource_name = resource_names[resource_id]
                        # Display longer names (13 chars instead of 8)
                        name_display = resource_name[:13]
                        lines.append(f"│ {name_display:13s}:  {amount:3d} │")
                        has_items = True
                if not has_items:
                    lines.append("│ (empty)             │")

            lines.append("└─────────────────────┘")

    # Pad to panel_height
    while len(lines) < panel_height:
        lines.append("                       ")

    return lines[:panel_height]


def build_object_info_panel(
    grid_objects: Dict[int, dict],
    object_type_names: list[str],
    cursor_row: int,
    cursor_col: int,
    panel_height: int,
) -> List[str]:
    """Build info panel showing selected object's properties.

    Args:
        grid_objects: Dictionary of grid objects
        object_type_names: List mapping type IDs to names
        cursor_row: Row position of cursor
        cursor_col: Column position of cursor
        panel_height: Height of the panel in lines

    Returns:
        List of strings, one per line
    """
    lines = []

    # Find object at cursor position
    selected_obj = None
    for obj in grid_objects.values():
        if obj["r"] == cursor_row and obj["c"] == cursor_col:
            selected_obj = obj
            break

    if selected_obj is None:
        lines.append("┌─ Object ────────────┐")
        lines.append("│ (empty space)       │")
        lines.append("└─────────────────────┘")
    else:
        type_name = object_type_names[selected_obj["type"]]
        lines.append("┌─ Object ────────────┐")
        lines.append(f"│ Type: {type_name[:15]:15s} │")
        lines.append(f"│ Pos: ({cursor_row:2d},{cursor_col:2d})       │")
        lines.append("├─ Properties ────────┤")

        # Show relevant properties based on object type
        props_shown = 0
        for key, value in sorted(selected_obj.items()):
            if key in ["r", "c", "type"]:
                continue  # Skip position and type (already shown)

            # Format the value
            if isinstance(value, dict):
                if key == "inventory" and value:
                    lines.append(f"│ {key[:13]:13s}: dict │")
                    props_shown += 1
                elif value:
                    lines.append(f"│ {key[:13]:13s}: dict │")
                    props_shown += 1
            elif isinstance(value, (int, float)):
                lines.append(f"│ {key[:13]:13s}: {value:5} │")
                props_shown += 1
            elif isinstance(value, str):
                val_display = str(value)[:7]
                lines.append(f"│ {key[:13]:13s}: {val_display:7s} │")
                props_shown += 1
            elif isinstance(value, bool):
                lines.append(f"│ {key[:13]:13s}: {str(value):5s} │")
                props_shown += 1

            if props_shown >= panel_height - 5:  # Leave room for header/footer
                break

        if props_shown == 0:
            lines.append("│ (no properties)     │")

        lines.append("└─────────────────────┘")

    # Pad to panel_height
    while len(lines) < panel_height:
        lines.append("                       ")

    return lines[:panel_height]
