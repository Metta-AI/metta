"""Info panel builders for miniscope interactive mode."""

from typing import Dict, List, Optional

import numpy as np
from rich.table import Table

try:
    from cogames.cogs_vs_clips.glyphs import GLYPH_DATA
except ImportError:
    GLYPH_DATA = None


def build_agent_info_panel(
    grid_objects: Dict[int, dict],
    object_type_names: list[str],
    selected_agent: Optional[int],
    resource_names: List[str],
    panel_height: int,
    total_rewards: np.ndarray,
    glyphs: list[str] | None = None,
    symbol_map: dict[str, str] | None = None,
    manual_agents: set[int] | None = None,
) -> Table:
    """Build info panel showing selected agent's inventory using rich.Table.

    Args:
        grid_objects: Dictionary of grid objects
        object_type_names: List mapping type IDs to names
        selected_agent: Selected agent ID (None if no selection)
        resource_names: List of resource names
        panel_height: Height of the panel in lines (unused, kept for compatibility)
        total_rewards: Array of cumulative rewards for each agent
        glyphs: Optional list of glyph symbols to display
        symbol_map: Optional map from object type names to render symbols
        manual_agents: Optional set of agent IDs in manual mode

    Returns:
        Rich Table object
    """
    from rich import box

    from .buffer import get_symbol_for_object

    table = Table(title="Agent Info", show_header=False, box=box.ROUNDED, padding=(0, 1), width=46)
    table.add_column("Key", style="cyan", no_wrap=True, width=12)
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

            # Get agent symbol
            agent_symbol = ""
            if symbol_map:
                agent_symbol = get_symbol_for_object(agent_obj, object_type_names, symbol_map)
                agent_symbol = f" {agent_symbol}"

            table.add_row("Agent", f"{selected_agent}{agent_symbol}")
            table.add_row("Reward", f"{reward:.1f}")

            # Show manual mode status
            if manual_agents and selected_agent in manual_agents:
                table.add_row("Mode", "MANUAL")
            else:
                table.add_row("Mode", "Policy")

            # Show glyph if available
            glyph_id = agent_obj.get("glyph")
            if glyph_id is not None and glyphs and 0 <= glyph_id < len(glyphs):
                glyph_symbol = glyphs[glyph_id]
                table.add_row("Glyph", f"{glyph_id} {glyph_symbol}")

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


def build_symbols_table(
    object_type_names: list[str],
    symbol_map: dict[str, str],
    max_rows: int = 10,
) -> Table:
    """Build a table showing all symbols and their names in the game.

    Args:
        object_type_names: List mapping type IDs to names
        symbol_map: Map from object type names to render symbols
        max_rows: Maximum number of rows to display

    Returns:
        Rich Table object
    """
    from rich import box

    table = Table(title="Symbols", show_header=False, box=box.ROUNDED, padding=(0, 1), width=46)
    table.add_column("Symbol", no_wrap=True, style="white", width=3)
    table.add_column("Name", style="cyan", overflow="ellipsis", width=15)
    table.add_column("Symbol", no_wrap=True, style="white", width=3)
    table.add_column("Name", style="cyan", overflow="ellipsis", width=15)

    # Build list of (symbol, name) tuples from symbol_map, sorted by name
    symbols_list = []
    seen_names = set()

    for name, symbol in sorted(symbol_map.items()):
        # Skip special entries
        if name in ["empty", "cursor", "?"] or not symbol:
            continue
        # Skip if we've seen this base name already
        base_name = name.split(".")[0]
        if base_name in seen_names:
            continue
        seen_names.add(base_name)

        # Format display name
        display_name = base_name.replace("_", " ").title()
        symbols_list.append((symbol, display_name))

    # Add rows in two columns
    for i in range(min(max_rows, (len(symbols_list) + 1) // 2)):
        left_idx = i
        right_idx = i + max_rows

        left_symbol, left_name = symbols_list[left_idx] if left_idx < len(symbols_list) else ("", "")
        right_symbol, right_name = symbols_list[right_idx] if right_idx < len(symbols_list) else ("", "")

        table.add_row(left_symbol, left_name, right_symbol, right_name)

    # If there are more symbols, add ellipsis
    if len(symbols_list) > max_rows * 2:
        remaining = len(symbols_list) - max_rows * 2
        table.add_row("⋯", f"({remaining} more)", "", "")

    return table


def build_object_info_panel(
    grid_objects: Dict[int, dict],
    object_type_names: list[str],
    cursor_row: int,
    cursor_col: int,
    panel_height: int,
    resource_names: list[str] | None = None,
) -> Table:
    """Build info panel showing selected object's properties using rich.Table.

    Args:
        grid_objects: Dictionary of grid objects
        object_type_names: List mapping type IDs to names
        cursor_row: Row position of cursor
        cursor_col: Column position of cursor
        panel_height: Height of the panel in lines to limit table size
        resource_names: List mapping resource IDs to names

    Returns:
        Rich Table object
    """
    from rich import box

    table = Table(title="Object Info", show_header=False, box=box.ROUNDED, padding=(0, 1), width=46)
    table.add_column("Key", style="cyan", no_wrap=True, width=12)
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
        actual_r = selected_obj.get("r", "?")
        actual_c = selected_obj.get("c", "?")
        table.add_row("Cursor pos", f"({cursor_row}, {cursor_col})")
        table.add_row("Object pos", f"({actual_r}, {actual_c})")

        # Check if this object has recipes (e.g., assembler)
        has_recipes = "recipes" in selected_obj
        current_recipe_inputs = selected_obj.get("current_recipe_inputs")

        # Show relevant properties based on object type, limited by panel_height
        # Account for table border (3 lines) and the 2 rows we already added
        max_property_rows = max(1, panel_height - 5)
        props_shown = 0

        # Special handling for current recipe - only show the active one
        if current_recipe_inputs:
            # Find the current recipe in the recipes list
            if has_recipes and isinstance(selected_obj["recipes"], list):
                for recipe in selected_obj["recipes"]:
                    if isinstance(recipe, dict):
                        inputs = recipe.get("inputs", {})
                        if inputs == current_recipe_inputs:
                            # Found the current recipe
                            outputs = recipe.get("outputs", {})

                            # Format resource strings with names if available
                            if resource_names:
                                inputs_str = ", ".join(f"{resource_names[k]}:{v}" for k, v in inputs.items())
                                outputs_str = ", ".join(f"{resource_names[k]}:{v}" for k, v in outputs.items())
                            else:
                                inputs_str = ", ".join(f"{k}:{v}" for k, v in inputs.items())
                                outputs_str = ", ".join(f"{k}:{v}" for k, v in outputs.items())

                            # Show current recipe
                            table.add_row("", "")  # Spacer
                            table.add_row("Recipe", f"{inputs_str} → {outputs_str}")
                            props_shown += 2
                            break

        # Show other properties
        for key, value in sorted(selected_obj.items()):
            if props_shown >= max_property_rows:
                # Add indicator that there are more properties
                remaining = len(selected_obj) - props_shown - 3
                if has_recipes:
                    remaining -= 1  # Account for recipes key
                if remaining > 0:
                    table.add_row("...", f"({remaining} more)")
                break

            # Skip keys we've already handled or don't want to show
            if key in ["r", "c", "type", "recipes", "current_recipe_inputs", "current_recipe_outputs"]:
                continue

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
