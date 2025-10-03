"""Main miniscope renderer class."""

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import numpy as np

from mettagrid import MettaGridEnv
from mettagrid.renderer.renderer import Renderer

from .buffer import build_grid_buffer, compute_bounds, get_symbol_for_object
from .info_panels import build_agent_info_panel, build_object_info_panel
from .interactive import run_interactive_loop

if TYPE_CHECKING:
    from rich.table import Table

DEFAULT_SYMBOL_MAP = {
    # Terrain
    "wall": "⬛",
    "empty": "⬜",
    "block": "📦",
    # Agents
    "agent": "🤖",
    "agent.agent": "🤖",
    "agent.team_1": "🔵",
    "agent.team_2": "🔴",
    "agent.team_3": "🟢",
    "agent.team_4": "🟡",
    "agent.prey": "🐰",
    "agent.predator": "🦁",
    # UI elements
    "cursor": "🎯",
    "?": "❓",
}


class MiniscopeRenderer(Renderer):
    """Emoji-based renderer for MettaGridEnv with full-width emoji support."""

    def __init__(
        self,
        max_steps: Optional[int] = None,
        glyphs=None,
        target_fps: int = 4,
    ):
        """Initialize MiniscopeRenderer.

        Args:
            max_steps: Maximum steps for interactive mode
            glyphs: Optional glyphs for display
            target_fps: Target frames per second for interactive mode
        """
        # Store parameters
        self.max_steps = max_steps
        self.glyphs = glyphs
        self.target_fps = target_fps
        self.env: Optional[MettaGridEnv] = None
        self._should_continue = True
        self.result = None

        # Renderer state (initialized in on_episode_start)
        self._object_type_names = None
        self._resource_names = None
        self._symbol_map = None
        self._min_row = 0
        self._min_col = 0
        self._height = 0
        self._width = 0
        self._bounds_set = False
        self._step_count = 0

    def on_episode_start(self, env: "MettaGridEnv") -> None:
        """Initialize the renderer for a new episode."""
        self.env = env
        self._step_count = 0
        self._should_continue = True

        # Initialize miniscope-specific attributes
        self._object_type_names = env.object_type_names
        self._resource_names = env.resource_names
        self._symbol_map = DEFAULT_SYMBOL_MAP.copy()

        # Add custom symbols from game config if available
        if hasattr(env, "mg_config") and env.mg_config and env.mg_config.game:
            for obj in env.mg_config.game.objects.values():
                if hasattr(obj, "render_symbol"):
                    self._symbol_map[obj.name] = obj.render_symbol

        # Initialize bounds
        self._min_row = 0
        self._min_col = 0
        self._height = env.map_height
        self._width = env.map_width
        self._bounds_set = self._height > 0 and self._width > 0
        self._last_buffer = None

    def render(
        self,
        step: int,
        grid_objects: Dict[int, dict],
        viewport_center_row: int | None = None,
        viewport_center_col: int | None = None,
        viewport_height: int | None = None,
        viewport_width: int | None = None,
    ) -> str:
        """Render the environment buffer and print to screen.

        Args:
            step: Current step number
            grid_objects: Dictionary of grid objects to render
            viewport_center_row: Optional viewport center row
            viewport_center_col: Optional viewport center column
            viewport_height: Optional viewport height
            viewport_width: Optional viewport width

        Returns:
            The rendered buffer string
        """
        return self._render_simple(
            step, grid_objects, viewport_center_row, viewport_center_col, viewport_height, viewport_width
        )

    def on_step(
        self,
        current_step: int,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        infos: Dict[str, Any],
    ) -> None:
        """Update step count."""
        self._step_count = current_step

    def should_continue(self) -> bool:
        """Check if rendering should continue."""
        return self._should_continue

    def on_episode_end(self, infos: Dict[str, Any]) -> None:
        """Clean up renderer resources."""
        print("\033[?25h", end="")  # Show cursor
        self.env = None
        self._should_continue = True

    def _symbol_for(self, obj: dict) -> str:
        """Get the emoji symbol for an object (for backward compatibility with tests)."""
        return get_symbol_for_object(obj, self._object_type_names, self._symbol_map)

    def _compute_bounds(self, grid_objects: Dict[int, dict]):
        """Compute and update bounds (for backward compatibility with tests)."""
        self._min_row, self._min_col, self._height, self._width = compute_bounds(grid_objects, self._object_type_names)
        self._bounds_set = True

    def _build_buffer(
        self,
        grid_objects: Dict[int, dict],
        viewport_center_row: int | None = None,
        viewport_center_col: int | None = None,
        viewport_height: int | None = None,
        viewport_width: int | None = None,
        cursor_row: int | None = None,
        cursor_col: int | None = None,
    ) -> str:
        """Build buffer (for backward compatibility with tests)."""
        self._ensure_bounds(grid_objects)
        return build_grid_buffer(
            grid_objects,
            self._object_type_names,
            self._symbol_map,
            self._min_row,
            self._min_col,
            self._height,
            self._width,
            viewport_center_row,
            viewport_center_col,
            viewport_height,
            viewport_width,
            cursor_row,
            cursor_col,
        )

    def _build_info_panel(
        self,
        grid_objects: Dict[int, dict],
        selected_agent: Optional[int],
        resource_names: List[str],
        panel_height: int,
        total_rewards: np.ndarray,
        glyphs: list[str] | None = None,
    ) -> "Table":
        """Build info panel (for backward compatibility with tests)."""
        return build_agent_info_panel(
            grid_objects,
            self._object_type_names,
            selected_agent,
            resource_names,
            panel_height,
            total_rewards,
            glyphs,
            self._symbol_map,
            None,  # manual_agents not available in this context
        )

    def _build_object_info_panel(
        self,
        grid_objects: Dict[int, dict],
        cursor_row: int,
        cursor_col: int,
        panel_height: int,
    ) -> "Table":
        """Build object info panel (for backward compatibility with tests)."""
        return build_object_info_panel(grid_objects, self._object_type_names, cursor_row, cursor_col, panel_height)

    def _ensure_bounds(self, grid_objects: Dict[int, dict]):
        """Compute and cache bounds if not already set."""
        if not self._bounds_set:
            self._min_row, self._min_col, self._height, self._width = compute_bounds(
                grid_objects, self._object_type_names
            )
            self._bounds_set = True

    def _render_simple(
        self,
        step: int,
        grid_objects: Dict[int, dict],
        viewport_center_row: int | None = None,
        viewport_center_col: int | None = None,
        viewport_height: int | None = None,
        viewport_width: int | None = None,
    ) -> str:
        """Render the environment buffer and print to screen (simple mode)."""
        self._ensure_bounds(grid_objects)
        current_buffer = build_grid_buffer(
            grid_objects,
            self._object_type_names,
            self._symbol_map,
            self._min_row,
            self._min_col,
            self._height,
            self._width,
            viewport_center_row,
            viewport_center_col,
            viewport_height,
            viewport_width,
        )
        self._last_buffer = current_buffer
        header = f"🎮 Metta AI Miniscope - Step: {step} 🎮"
        separator = "═" * (self._width * 2)
        frame_buffer = f"\033[2J\033[H{header}\n{separator}\n{current_buffer}\n{separator}"
        print(frame_buffer, end="", flush=True)
        return current_buffer

    def get_buffer(
        self,
        grid_objects: Dict[int, dict],
        viewport_center_row: int | None = None,
        viewport_center_col: int | None = None,
        viewport_height: int | None = None,
        viewport_width: int | None = None,
    ) -> str:
        """Return emoji map buffer without side effects."""
        self._ensure_bounds(grid_objects)
        return build_grid_buffer(
            grid_objects,
            self._object_type_names,
            self._symbol_map,
            self._min_row,
            self._min_col,
            self._height,
            self._width,
            viewport_center_row,
            viewport_center_col,
            viewport_height,
            viewport_width,
        )

    def _run_interactive_loop(self):
        """Run the miniscope interactive loop."""
        if not self.env:
            return

        def get_actions_fn(
            obs: np.ndarray,
            selected_agent: Optional[int],
            manual_action: Optional[int | tuple],
            manual_agents: set[int],
        ) -> np.ndarray:
            """Get actions for all agents, with optional manual override."""
            if not self.env:
                return np.zeros((0, 2), dtype=np.int32)

            noop_action_id = self.env.action_names.index("noop") if "noop" in self.env.action_names else 0
            actions = np.zeros((self.env.num_agents, 2), dtype=np.int32)
            actions[:, 0] = noop_action_id

            # Apply manual action if provided
            if selected_agent is not None and manual_action is not None:
                if isinstance(manual_action, tuple):
                    actions[selected_agent] = list(manual_action)
                else:
                    move_action_id = self.env.action_names.index("move") if "move" in self.env.action_names else 0
                    actions[selected_agent] = [move_action_id, manual_action]

            return actions

        self.result = self.interactive_loop(
            self.env,
            get_actions_fn,
            max_steps=self.max_steps,
            target_fps=self.target_fps,
            glyphs=self.glyphs,
        )

    def get_result(self) -> Optional[Dict[str, Any]]:
        """Get the result from the interactive loop."""
        return self.result

    def interactive_loop(
        self,
        env,
        get_actions_fn: Callable[[np.ndarray, Optional[int], Optional[int | tuple], set[int]], np.ndarray],
        max_steps: Optional[int] = None,
        target_fps: int = 4,
        glyphs: list[str] | None = None,
    ) -> Dict[str, Any]:
        """Run interactive rendering loop with keyboard controls.

        Args:
            env: MettaGrid environment
            get_actions_fn: Function that takes (obs, selected_agent, manual_action_direction, manual_agents)
                           and returns actions for all agents
            max_steps: Maximum steps to run (None for unlimited)
            target_fps: Target frames per second when playing
            glyphs: Optional list of glyph symbols for display

        Returns:
            Dict with final statistics
        """
        return run_interactive_loop(
            env,
            self._object_type_names,
            self._symbol_map,
            get_actions_fn,
            max_steps,
            target_fps,
            glyphs,
            self._resource_names,
        )

    def __del__(self):
        """Show cursor when renderer is destroyed."""
        if hasattr(self, "_object_type_names"):  # Only if initialized
            print("\033[?25h", end="")
