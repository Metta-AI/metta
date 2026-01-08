"""Agent info panel component for miniscope renderer."""

from typing import Dict, Optional

import numpy as np

from mettagrid.renderer.miniscope.components.base import MiniscopeComponent
from mettagrid.renderer.miniscope.miniscope_panel import PanelLayout
from mettagrid.renderer.miniscope.miniscope_state import MiniscopeState
from mettagrid.renderer.miniscope.symbol import get_symbol_for_object
from mettagrid.simulator.simulator import Simulation


class AgentInfoComponent(MiniscopeComponent):
    """Component for displaying agent information."""

    def __init__(
        self,
        sim: Simulation,
        state: MiniscopeState,
        panels: PanelLayout,
    ):
        """Initialize the agent info component.

        Args:
            env: MettaGrid environment reference
            state: Miniscope state reference
            panels: Panel layout containing all panels
        """
        super().__init__(sim=sim, state=state, panels=panels)
        sidebar_panel = panels.get_sidebar_panel("agent_info")
        assert sidebar_panel is not None
        self._set_panel(sidebar_panel)

    def _get_resource_names(self) -> list[str]:
        """Get resource names from state."""
        resource_names = self.state.resource_names
        assert resource_names is not None
        return resource_names

    def _get_symbol_map(self) -> dict[str, str]:
        """Get symbol map from state."""
        symbol_map = self.state.symbol_map
        assert symbol_map is not None
        return symbol_map

    def _get_vibes(self) -> Optional[list[str]]:
        """Get vibes from state."""
        return self.state.vibes

    def update(self) -> None:
        """Render the agent info panel using current environment and state."""
        panel = self._panel
        assert panel is not None
        if not self.state.is_sidebar_visible("agent_info"):
            panel.clear()
            return

        grid_objects = self._sim.grid_objects()

        lines = self._build_lines(
            grid_objects,
            self.state.selected_agent,
            self.state.total_rewards,
            self.state.manual_agents,
        )
        panel.set_content(lines)

    def _build_lines(
        self,
        grid_objects: Dict[int, dict],
        selected_agent: Optional[int],
        total_rewards: Optional[np.ndarray],
        manual_agents: set[int],
    ) -> list[str]:
        """Build fixed-width lines for agent info display."""
        width = self._width if self._width else 40
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
        agent_symbol = ""
        if symbol_map:
            agent_symbol = get_symbol_for_object(agent_obj, symbol_map)

        vibes = self._get_vibes()
        vibe_id = agent_obj.get("vibe")
        vibe_text = ""
        if vibe_id is not None:
            if isinstance(vibe_id, int) and vibes and 0 <= vibe_id < len(vibes):
                vibe_text = f"{vibe_id} {vibes[vibe_id]}"
            elif isinstance(vibe_id, str):
                vibe_text = vibe_id

        mode_text = "MANUAL" if selected_agent in manual_agents else "Policy"

        entries: list[tuple[str, str]] = []
        agent_value = f"{selected_agent} {agent_symbol}".strip()
        entries.append(("Agent", agent_value))
        entries.append(("Mode", mode_text))
        entries.append(("Reward", f"{reward:.2f}"))
        if vibe_text:
            entries.append(("Vibe", vibe_text))

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
