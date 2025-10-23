"""Object info panel component for miniscope renderer."""

from typing import Dict

from mettagrid import MettaGridEnv
from mettagrid.core import BoundingBox
from mettagrid.renderer.miniscope.miniscope_panel import SIDEBAR_WIDTH, PanelLayout
from mettagrid.renderer.miniscope.miniscope_state import (
    SELECT_MODE_KEY,
    MiniscopeState,
    RenderMode,
)

from .base import MiniscopeComponent


class ObjectInfoComponent(MiniscopeComponent):
    """Component for displaying object information at cursor position."""

    def __init__(
        self,
        env: MettaGridEnv,
        state: MiniscopeState,
        panels: PanelLayout,
    ):
        """Initialize the object info component."""
        super().__init__(env=env, state=state, panels=panels)
        self._set_panel(panels.get_sidebar_panel("object_info"))

    def _get_object_type_names(self) -> list[str]:
        """Get object type names from state."""
        return self.state.object_type_names if self.state else []

    def _get_resource_names(self) -> list[str]:
        """Get resource names from state."""
        return self.state.resource_names if self.state else []

    def update(self) -> None:
        """Render the object info panel using current environment and state."""
        if not self.state.is_sidebar_visible("object_info"):
            self._panel.clear()
            return

        if not self.env or not self.state:
            width = self._width if self._width else SIDEBAR_WIDTH
            lines = ["Object Info", "-" * min(width, 40), "Object info unavailable"]
            self._panel.set_content(lines)
            return

        if self.state.mode != RenderMode.SELECT:
            width = self._width if self._width else SIDEBAR_WIDTH
            select_hint = f"Switch to Select mode (press {SELECT_MODE_KEY})"
            lines = [
                "Object Info",
                "-" * min(width, 40),
                select_hint,
            ]
            self._panel.set_content(lines)
            return

        bbox = BoundingBox(
            min_row=0,
            max_row=self.env.map_height,
            min_col=0,
            max_col=self.env.map_width,
        )
        grid_objects = self.env.grid_objects(bbox)

        panel_height = self.state.viewport_height // 2 if self.state.viewport_height else 20

        lines = self._build_lines(
            grid_objects,
            self.state.cursor_row,
            self.state.cursor_col,
            panel_height,
        )
        self._panel.set_content(lines)

    def _build_lines(
        self,
        grid_objects: Dict[int, dict],
        cursor_row: int,
        cursor_col: int,
        panel_height: int,
    ) -> list[str]:
        """Build object information lines without color formatting."""
        width = self._width if self._width else SIDEBAR_WIDTH
        width = max(24, width)

        header = "Object Info"
        lines: list[str] = [header[:width].ljust(width), "-" * min(width, 40)]

        selected_obj = None
        for obj in grid_objects.values():
            if obj["r"] == cursor_row and obj["c"] == cursor_col:
                selected_obj = obj
                break

        if selected_obj is None:
            lines.append("Status: (empty space)".ljust(width))
            return lines

        # Get type name - prefer type_name field if available, otherwise look up
        if "type_name" in selected_obj:
            type_name = selected_obj["type_name"]
        else:
            object_type_names = self._get_object_type_names()
            type_name = object_type_names[selected_obj["type"]] if object_type_names else str(selected_obj["type"])
        lines.append(f"Type: {type_name}"[:width].ljust(width))
        lines.append(f"Cursor pos: ({cursor_row}, {cursor_col})"[:width].ljust(width))
        actual_r = selected_obj.get("r", "?")
        actual_c = selected_obj.get("c", "?")
        lines.append(f"Object pos: ({actual_r}, {actual_c})"[:width].ljust(width))

        max_property_rows = max(1, panel_height - 6)
        properties_added = 0

        current_recipe_inputs = selected_obj.get("current_recipe_inputs")
        has_recipes = "recipes" in selected_obj

        if current_recipe_inputs and has_recipes and isinstance(selected_obj["recipes"], list):
            for recipe in selected_obj["recipes"]:
                if isinstance(recipe, dict):
                    inputs = recipe.get("inputs", {})
                    if inputs == current_recipe_inputs:
                        outputs = recipe.get("outputs", {})
                        resource_names = self._get_resource_names()
                        if resource_names:
                            inputs_str = ", ".join(f"{resource_names[k]}:{v}" for k, v in inputs.items())
                            outputs_str = ", ".join(f"{resource_names[k]}:{v}" for k, v in outputs.items())
                        else:
                            inputs_str = ", ".join(f"{k}:{v}" for k, v in inputs.items())
                            outputs_str = ", ".join(f"{k}:{v}" for k, v in outputs.items())

                        lines.append("Recipe:"[:width].ljust(width))
                        lines.append(f"  {inputs_str} -> {outputs_str}"[:width].ljust(width))
                        properties_added += 2
                        break

        for key, value in sorted(selected_obj.items()):
            if properties_added >= max_property_rows:
                remaining = len(selected_obj) - properties_added - 4
                if has_recipes:
                    remaining -= 1
                if remaining > 0:
                    lines.append(f"... ({remaining} more)"[:width].ljust(width))
                break

            if key in ["r", "c", "type", "recipes", "current_recipe_inputs", "current_recipe_outputs"]:
                continue

            if isinstance(value, dict):
                if value:
                    lines.append(f"{key}: dict"[:width].ljust(width))
                    properties_added += 1
            elif isinstance(value, (int, float, bool, str)):
                lines.append(f"{key}: {value}"[:width].ljust(width))
                properties_added += 1

        if properties_added == 0:
            lines.append("Properties: (none)"[:width].ljust(width))

        return lines
