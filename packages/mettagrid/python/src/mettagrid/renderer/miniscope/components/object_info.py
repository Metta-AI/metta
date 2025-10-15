"""Object info panel component for miniscope renderer."""

from typing import Dict

from rich import box
from rich.table import Table

from mettagrid import MettaGridEnv
from mettagrid.core import BoundingBox
from mettagrid.renderer.miniscope.miniscope_panel import PanelLayout
from mettagrid.renderer.miniscope.miniscope_state import MiniscopeState, RenderMode

from .base import MiniscopeComponent


class ObjectInfoComponent(MiniscopeComponent):
    """Component for displaying object information at cursor position."""

    def __init__(
        self,
        env: MettaGridEnv,
        state: MiniscopeState,
        panels: PanelLayout,
    ):
        """Initialize the object info component.

        Args:
            env: MettaGrid environment reference
            state: Miniscope state reference
            panels: Panel layout containing all panels
        """
        super().__init__(env=env, state=state, panels=panels)
        self._set_panel(panels.sidebar)

    def _get_object_type_names(self) -> list[str]:
        """Get object type names from state."""
        return self.state.object_type_names if self.state else []

    def _get_resource_names(self) -> list[str]:
        """Get resource names from state."""
        return self.state.resource_names if self.state else []

    def update(self) -> None:
        """Render the object info panel using current environment and state."""
        if not self.env or not self.state:
            return

        # Only display in SELECT mode
        if self.state.mode != RenderMode.SELECT:
            return

        # Get grid objects from environment
        bbox = BoundingBox(
            min_row=0,
            max_row=self.env.map_height,
            min_col=0,
            max_col=self.env.map_width,
        )
        grid_objects = self.env.grid_objects(bbox)

        # Use viewport height for panel height
        panel_height = self.state.viewport_height // 2 if self.state.viewport_height else 20

        table = self._build_table(
            grid_objects,
            self.state.cursor_row,
            self.state.cursor_col,
            panel_height,
        )
        self._panel.set_content(table)

    def _build_table(
        self,
        grid_objects: Dict[int, dict],
        cursor_row: int,
        cursor_col: int,
        panel_height: int,
    ) -> Table:
        """Build the object info table.

        Returns:
            Rich Table object
        """
        table = Table(
            title="Object Info",
            show_header=False,
            box=box.ROUNDED,
            padding=(0, 1),
            width=self._width,
        )
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
            object_type_names = self._get_object_type_names()
            type_name = object_type_names[selected_obj["type"]] if object_type_names else str(selected_obj["type"])
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
                                resource_names = self._get_resource_names()
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
