"""Object info panel component for miniscope renderer."""

from typing import Dict

from mettagrid.renderer.miniscope.components.base import MiniscopeComponent
from mettagrid.renderer.miniscope.miniscope_panel import SIDEBAR_WIDTH, PanelLayout
from mettagrid.renderer.miniscope.miniscope_state import MiniscopeState, RenderMode
from mettagrid.simulator.simulator import Simulation


class ObjectInfoComponent(MiniscopeComponent):
    """Component for displaying object information at cursor position."""

    def __init__(
        self,
        sim: Simulation,
        state: MiniscopeState,
        panels: PanelLayout,
    ):
        """Initialize the object info component.

        Args:
            sim: MettaGrid simulator reference
            state: Miniscope state reference
            panels: Panel layout containing all panels
        """
        super().__init__(sim=sim, state=state, panels=panels)
        sidebar_panel = panels.get_sidebar_panel("object_info")
        assert sidebar_panel is not None
        self._set_panel(sidebar_panel)

    def _get_resource_names(self) -> list[str]:
        """Get resource names from state."""
        resource_names = self.state.resource_names
        assert resource_names is not None
        return resource_names

    def update(self) -> None:
        """Render the object info panel using current environment and state."""
        panel = self._panel
        assert panel is not None
        if not self.state.is_sidebar_visible("object_info"):
            panel.clear()
            return

        if self.state.mode != RenderMode.SELECT:
            width = self._width if self._width else 40
            select_hint = "Switch to Select mode (press t)"
            lines = [
                "Object Info",
                "-" * min(width, 40),
                select_hint,
            ]
            panel.set_content(lines)
            return

        grid_objects = self._sim.grid_objects()

        panel_height = self.state.viewport_height // 2 if self.state.viewport_height else 20

        lines = self._build_lines(
            grid_objects,
            self.state.cursor_row,
            self.state.cursor_col,
            panel_height,
        )
        panel.set_content(lines)

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

        # Get type name (breaking: require type_name present)
        type_name = selected_obj.get("type_name")
        if type_name is None:
            type_name = "<missing type_name>"
        lines.append(f"Type: {type_name}"[:width].ljust(width))
        lines.append(f"Cursor pos: ({cursor_row}, {cursor_col})"[:width].ljust(width))
        actual_r = selected_obj.get("r", "?")
        actual_c = selected_obj.get("c", "?")
        lines.append(f"Object pos: ({actual_r}, {actual_c})"[:width].ljust(width))

        max_property_rows = max(1, panel_height - 6)
        properties_added = 0

        current_protocol_inputs = selected_obj.get("current_protocol_inputs")
        has_protocols = "protocols" in selected_obj

        if current_protocol_inputs and has_protocols and isinstance(selected_obj["protocols"], list):
            for protocol in selected_obj["protocols"]:
                if isinstance(protocol, dict):
                    inputs = protocol.get("inputs", {})
                    if inputs == current_protocol_inputs:
                        outputs = protocol.get("outputs", {})
                        resource_names = self._get_resource_names()
                        if resource_names:
                            inputs_str = ", ".join(f"{resource_names[k]}:{v}" for k, v in inputs.items())
                            outputs_str = ", ".join(f"{resource_names[k]}:{v}" for k, v in outputs.items())
                        else:
                            inputs_str = ", ".join(f"{k}:{v}" for k, v in inputs.items())
                            outputs_str = ", ".join(f"{k}:{v}" for k, v in outputs.items())

                        lines.append("Protocol:"[:width].ljust(width))
                        lines.append(f"  {inputs_str} -> {outputs_str}"[:width].ljust(width))
                        properties_added += 2
                        break

        for key, value in sorted(selected_obj.items()):
            if properties_added >= max_property_rows:
                remaining = len(selected_obj) - properties_added - 4
                if has_protocols:
                    remaining -= 1
                if remaining > 0:
                    lines.append(f"... ({remaining} more)"[:width].ljust(width))
                break

            if key in ["r", "c", "type", "protocols", "current_protocol_inputs", "current_protocol_outputs"]:
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
