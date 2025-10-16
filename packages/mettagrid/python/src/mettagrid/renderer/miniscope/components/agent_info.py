"""Agent info panel component for miniscope renderer."""

from typing import Dict, Optional

import numpy as np
from rich import box
from rich.table import Table
from rich.text import Text

from mettagrid import MettaGridEnv
from mettagrid.core import BoundingBox
from mettagrid.renderer.miniscope.miniscope_panel import PanelLayout
from mettagrid.renderer.miniscope.miniscope_state import MiniscopeState
from mettagrid.renderer.miniscope.styles import gradient_title, surface_panel
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
        self._set_panel(panels.sidebar)

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
        if not self.env or not self.state:
            return

        # Get grid objects from environment
        bbox = BoundingBox(
            min_row=0,
            max_row=self.env.map_height,
            min_col=0,
            max_col=self.env.map_width,
        )
        grid_objects = self.env.grid_objects(bbox)

        table = self._build_table(
            grid_objects,
            self.state.selected_agent,
            self.state.total_rewards,
            self.state.manual_agents,
        )
        panel = surface_panel(
            table,
            title=gradient_title("Agent Snapshot"),
            border_variant="alt",
            variant="alt",
        )
        self._panel.append_block(panel)

    def _build_table(
        self,
        grid_objects: Dict[int, dict],
        selected_agent: Optional[int],
        total_rewards: Optional[np.ndarray],
        manual_agents: set[int],
    ) -> Table:
        """Build the agent info table.

        Returns:
            Rich Table object
        """
        table = Table(
            title=gradient_title("Agent Info"),
            show_header=False,
            box=box.ROUNDED,
            padding=(0, 1),
            width=self._width,
            border_style="border.alt",
            row_styles=["surface.alt", "surface"],
        )
        table.add_column("Key", style="muted", no_wrap=True, width=12)
        table.add_column("Value", style="text")

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
                reward = (
                    total_rewards[selected_agent]
                    if total_rewards is not None and selected_agent < len(total_rewards)
                    else 0.0
                )

                agent_symbol = ""
                symbol_map = self._get_symbol_map()
                object_type_names = self._get_object_type_names()
                if symbol_map and object_type_names:
                    agent_symbol = get_symbol_for_object(agent_obj, object_type_names, symbol_map)
                    agent_symbol = f" {agent_symbol}"

                table.add_row("Agent", f"{selected_agent}{agent_symbol}")
                table.add_row("Reward", f"{reward:.1f}")

                if selected_agent in manual_agents:
                    table.add_row("Mode", Text("MANUAL", style="accent"))
                else:
                    table.add_row("Mode", "Policy")

                glyph_id = agent_obj.get("glyph")
                if glyph_id is None:
                    glyph_id = agent_obj.get("glyph_id")

                pending_override = None
                if selected_agent is not None:
                    pending_override = self.state.pending_glyphs.get(selected_agent)

                if glyph_id is None and pending_override is not None:
                    glyph_id = pending_override
                elif isinstance(glyph_id, int) and pending_override is not None and glyph_id == pending_override:
                    self.state.pending_glyphs.pop(selected_agent, None)

                glyphs = self._get_glyphs()
                if glyph_id is not None:
                    if isinstance(glyph_id, int) and glyphs and 0 <= glyph_id < len(glyphs):
                        glyph_symbol = glyphs[glyph_id]
                        table.add_row("Glyph", f"{glyph_id} {glyph_symbol}")
                    elif isinstance(glyph_id, str):
                        table.add_row("Glyph", glyph_id)

                inventory = agent_obj.get("inventory", {})
                if not inventory or not isinstance(inventory, dict):
                    table.add_row("Inventory", "(empty)")
                else:
                    has_items = False
                    for resource_id, amount in sorted(inventory.items()):
                        resource_names = self._get_resource_names()
                        if resource_id < len(resource_names) and amount > 0:
                            resource_name = resource_names[resource_id]
                            table.add_row(resource_name, str(amount))
                            has_items = True
                    if not has_items:
                        table.add_row("Inventory", "(empty)")

        return table
