"""Info panel builders for miniscope interactive mode."""

from typing import Dict, List, Optional

import numpy as np
from rich.table import Table


def build_agent_info_panel(
    grid_objects: Dict[int, dict],
    object_type_names: list[str],
    selected_agent: Optional[int],
    resource_names: List[str],
    panel_height: int,
    total_rewards: np.ndarray,
) -> Table:
    """Build info panel showing selected agent's inventory using rich.Table.

    Args:
        grid_objects: Dictionary of grid objects
        object_type_names: List mapping type IDs to names
        selected_agent: Selected agent ID (None if no selection)
        resource_names: List of resource names
        panel_height: Height of the panel in lines (unused, kept for compatibility)
        total_rewards: Array of cumulative rewards for each agent

    Returns:
        Rich Table object
    """
    table = Table(title="Agent Info", show_header=False, box=None, padding=(0, 1))
    table.add_column("Key", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")

    if selected_agent is None:
        table.add_row("Status", "No agent selected")
    else:
        # Find the agent in grid_objects
        agent_obj = None
        for obj in grid_objects.values():
            if obj.get("agent_id") == selected_agent:
                agent_obj = obj
                break

        if agent_obj is None:
            table.add_row("Agent", str(selected_agent))
            table.add_row("Status", "(not found)")
        else:
            # Build inventory display
            reward = total_rewards[selected_agent] if selected_agent < len(total_rewards) else 0.0
            table.add_row("Agent", str(selected_agent))
            table.add_row("Reward", f"{reward:.1f}")

            inventory = agent_obj.get("inventory", {})
            if not inventory or not isinstance(inventory, dict):
                table.add_row("Inventory", "(empty)")
            else:
                # Show resources with amounts (inventory is dict of resource_id -> amount)
                has_items = False
                for resource_id, amount in sorted(inventory.items()):
                    if resource_id < len(resource_names) and amount > 0:
                        resource_name = resource_names[resource_id]
                        table.add_row(resource_name, str(amount))
                        has_items = True
                if not has_items:
                    table.add_row("Inventory", "(empty)")

    return table


def build_object_info_panel(
    grid_objects: Dict[int, dict],
    object_type_names: list[str],
    cursor_row: int,
    cursor_col: int,
    panel_height: int,
) -> Table:
    """Build info panel showing selected object's properties using rich.Table.

    Args:
        grid_objects: Dictionary of grid objects
        object_type_names: List mapping type IDs to names
        cursor_row: Row position of cursor
        cursor_col: Column position of cursor
        panel_height: Height of the panel in lines (unused, kept for compatibility)

    Returns:
        Rich Table object
    """
    table = Table(title="Object Info", show_header=False, box=None, padding=(0, 1))
    table.add_column("Key", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")

    # Find object at cursor position
    selected_obj = None
    for obj in grid_objects.values():
        if obj["r"] == cursor_row and obj["c"] == cursor_col:
            selected_obj = obj
            break

    if selected_obj is None:
        table.add_row("Status", "(empty space)")
    else:
        type_name = object_type_names[selected_obj["type"]]
        table.add_row("Type", type_name)
        table.add_row("Position", f"({cursor_row}, {cursor_col})")

        # Show relevant properties based on object type
        props_shown = 0
        for key, value in sorted(selected_obj.items()):
            if key in ["r", "c", "type"]:
                continue  # Skip position and type (already shown)

            # Format the value
            if isinstance(value, dict):
                if value:
                    table.add_row(key, "dict")
                    props_shown += 1
            elif isinstance(value, (int, float)):
                table.add_row(key, str(value))
                props_shown += 1
            elif isinstance(value, str):
                table.add_row(key, value)
                props_shown += 1
            elif isinstance(value, bool):
                table.add_row(key, str(value))
                props_shown += 1

        if props_shown == 0:
            table.add_row("Properties", "(none)")

    return table
