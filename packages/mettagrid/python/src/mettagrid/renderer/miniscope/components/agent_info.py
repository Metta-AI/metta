"""Agent info panel component for miniscope renderer."""

from typing import Dict, List, Optional

import numpy as np
from rich import box
from rich.table import Table

from mettagrid import MettaGridEnv
from mettagrid.core import BoundingBox
from mettagrid.renderer.miniscope.miniscope_panel import PanelLayout
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

    def update(self) -> List[str]:
        """Render the agent info panel using current environment and state.

        Returns:
            List of strings representing the rendered panel
        """
        if not self.env or not self.state:
            return ["[Agent Info: No environment or state]"]

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
        return table

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
            title="Agent Info",
            show_header=False,
            box=box.ROUNDED,
            padding=(0, 1),
            width=self._width,
        )
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
                reward = (
                    total_rewards[selected_agent]
                    if total_rewards is not None and selected_agent < len(total_rewards)
                    else 0.0
                )

                # Get agent symbol
                agent_symbol = ""
                symbol_map = self._get_symbol_map()
                object_type_names = self._get_object_type_names()
                if symbol_map and object_type_names:
                    agent_symbol = get_symbol_for_object(agent_obj, object_type_names, symbol_map)
                    agent_symbol = f" {agent_symbol}"

                table.add_row("Agent", f"{selected_agent}{agent_symbol}")
                table.add_row("Reward", f"{reward:.1f}")

                # Show manual mode status
                if selected_agent in manual_agents:
                    table.add_row("Mode", "MANUAL")
                else:
                    table.add_row("Mode", "Policy")

                # Show glyph if available
                glyph_id = agent_obj.get("glyph")
                glyphs = self._get_glyphs()
                # glyph_id could be an int or a string emoji
                if glyph_id is not None:
                    if isinstance(glyph_id, int) and glyphs and 0 <= glyph_id < len(glyphs):
                        glyph_symbol = glyphs[glyph_id]
                        table.add_row("Glyph", f"{glyph_id} {glyph_symbol}")
                    elif isinstance(glyph_id, str):
                        # Direct emoji string
                        table.add_row("Glyph", glyph_id)

                inventory = agent_obj.get("inventory", {})
                if not inventory or not isinstance(inventory, dict):
                    table.add_row("Inventory", "(empty)")
                else:
                    # Show resources with amounts (inventory is dict of resource_id -> amount)
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
