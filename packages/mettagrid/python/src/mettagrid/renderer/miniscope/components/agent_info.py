"""Agent info panel component for miniscope renderer."""

from typing import Dict, Optional

import numpy as np

from mettagrid import MettaGridEnv
from mettagrid.core import BoundingBox
from mettagrid.renderer.miniscope.miniscope_panel import SIDEBAR_WIDTH, PanelLayout
from mettagrid.renderer.miniscope.miniscope_state import MiniscopeState
from mettagrid.renderer.miniscope.symbol import get_symbol_for_object

from .base import MiniscopeComponent


class AgentInfoComponent(MiniscopeComponent):
    """Component for displaying agent information."""

    def __init__(
        self,
        env: MettaGridEnv,
        state: MiniscopeState,
        panels: PanelLayout,
    ):
        """Initialize the agent info component.

        Args:
            env: MettaGrid environment reference
            state: Miniscope state reference
            panels: Panel layout containing all panels
        """
        super().__init__(env=env, state=state, panels=panels)
        self._set_panel(panels.get_sidebar_panel("agent_info"))

    def _get_object_type_names(self) -> list[str]:
        """Get object type names from state."""
        return self.state.object_type_names if self.state else []

    def _get_resource_names(self) -> list[str]:
        """Get resource names from state."""
        return self.state.resource_names if self.state else []

    def _get_symbol_map(self) -> dict[str, str]:
        """Get symbol map from state."""
        return self.state.symbol_map if self.state else {}

    def _get_glyphs(self) -> Optional[list[str]]:
        """Get glyphs from state."""
        return self.state.glyphs if self.state else None

    def update(self) -> None:
        """Render the agent info panel using current environment and state."""
        if not self.state.is_sidebar_visible("agent_info"):
            self._panel.clear()
            return

        if not self.env or not self.state:
            self._panel.set_content(["Agent info unavailable"])
            return

        bbox = BoundingBox(
            min_row=0,
            max_row=self.env.map_height,
            min_col=0,
            max_col=self.env.map_width,
        )
        grid_objects = self.env.grid_objects(bbox)

        lines = self._build_lines(
            grid_objects,
            self.state.selected_agent,
            self.state.total_rewards,
            self.state.manual_agents,
        )
        self._panel.set_content(lines)

    def _build_lines(
        self,
        grid_objects: Dict[int, dict],
        selected_agent: Optional[int],
        total_rewards: Optional[np.ndarray],
        manual_agents: set[int],
    ) -> list[str]:
        """Build fixed-width lines for agent info display."""
        width = self._width if self._width else SIDEBAR_WIDTH
        width = max(24, width)

        lines: list[str] = []
        header = "Agent Info"
        lines.append(header[:width].ljust(width))
        lines.append("-" * min(width, 40))

        label_width = min(18, max(8, width - 16))

        if selected_agent is None:
            lines.append(self._format_entry("Status", "No agent selected", width, label_width))
            return lines

        agent_obj = None
        for obj in grid_objects.values():
            if obj.get("agent_id") == selected_agent:
                agent_obj = obj
                break

        if agent_obj is None:
            lines.append(self._format_entry("Status", f"Agent {selected_agent} not found", width, label_width))
            return lines

        reward = 0.0
        if total_rewards is not None and selected_agent < len(total_rewards):
            reward = float(total_rewards[selected_agent])

        symbol_map = self._get_symbol_map()
        object_type_names = self._get_object_type_names()
        agent_symbol = ""
        if symbol_map and object_type_names:
            agent_symbol = get_symbol_for_object(agent_obj, object_type_names, symbol_map)

        glyphs = self._get_glyphs()
        glyph_id = agent_obj.get("glyph")
        glyph_text = ""
        if glyph_id is not None:
            if isinstance(glyph_id, int) and glyphs and 0 <= glyph_id < len(glyphs):
                glyph_text = f"{glyph_id} {glyphs[glyph_id]}"
            elif isinstance(glyph_id, str):
                glyph_text = glyph_id

        mode_text = "MANUAL" if selected_agent in manual_agents else "Policy"

        entries: list[tuple[str, str]] = []
        agent_value = f"{selected_agent} {agent_symbol}".strip()
        entries.append(("Agent", agent_value))
        entries.append(("Mode", mode_text))
        entries.append(("Reward", f"{reward:.2f}"))
        if glyph_text:
            entries.append(("Glyph", glyph_text))

        inventory = agent_obj.get("inventory", {}) if agent_obj else {}
        resource_names = self._get_resource_names()

        if not inventory or not isinstance(inventory, dict):
            entries.append(("Inventory", "(empty)"))
        else:
            first_resource = True
            for resource_id, amount in sorted(inventory.items()):
                if amount <= 0:
                    continue
                if resource_id >= len(resource_names):
                    resource_name = str(resource_id)
                else:
                    resource_name = resource_names[resource_id]
                value_text = f"{resource_name}: {amount}"
                if first_resource:
                    entries.append(("Inventory", value_text))
                    first_resource = False
                else:
                    entries.append(("", value_text))

            if first_resource:
                entries.append(("Inventory", "(empty)"))

        for label, value in entries:
            lines.append(self._format_entry(label, value, width, label_width))

        return lines

    def _format_entry(self, label: str, value: str, width: int, label_width: int) -> str:
        """Format a key/value entry with aligned value column."""
        label = label[:label_width]
        value_width = max(1, width - label_width - 2)
        trimmed_value = value[:value_width]
        if label:
            formatted = f"{label:<{label_width}}: {trimmed_value}"
        else:
            formatted = f"{' ':<{label_width}}  {trimmed_value}"
        return formatted[:width].ljust(width)
