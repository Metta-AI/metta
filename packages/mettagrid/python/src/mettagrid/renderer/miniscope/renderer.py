"""Main miniscope renderer class."""

from typing import Callable, Dict, List, Optional

import numpy as np

from .buffer import build_grid_buffer, compute_bounds
from .info_panels import build_agent_info_panel, build_object_info_panel
from .interactive import run_interactive_loop
from .symbols import get_symbol_for_object


class MiniscopeRenderer:
    """Emoji-based renderer for MettaGridEnv with full-width emoji support."""

    def __init__(self, object_type_names: List[str], map_height: int = 0, map_width: int = 0):
        self._object_type_names = object_type_names
        self._min_row = 0
        self._min_col = 0
        self._height = map_height if map_height > 0 else 0
        self._width = map_width if map_width > 0 else 0
        self._bounds_set = self._height > 0 and self._width > 0
        self._last_buffer = None
        # Clear screen and hide cursor on init
        print("\033[?25l", end="")  # Hide cursor
        print("\033[2J", end="")  # Clear screen
        print("\033[H", end="")  # Move to home position

    def __del__(self):
        # Show cursor when renderer is destroyed
        print("\033[?25h", end="")

    def _symbol_for(self, obj: dict) -> str:
        """Get the emoji symbol for an object (for backward compatibility with tests)."""
        return get_symbol_for_object(obj, self._object_type_names)

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
    ) -> List[str]:
        """Build info panel (for backward compatibility with tests)."""
        return build_agent_info_panel(
            grid_objects, self._object_type_names, selected_agent, resource_names, panel_height, total_rewards
        )

    def _build_object_info_panel(
        self,
        grid_objects: Dict[int, dict],
        cursor_row: int,
        cursor_col: int,
        panel_height: int,
    ) -> List[str]:
        """Build object info panel (for backward compatibility with tests)."""
        return build_object_info_panel(grid_objects, self._object_type_names, cursor_row, cursor_col, panel_height)

    def _ensure_bounds(self, grid_objects: Dict[int, dict]):
        """Compute and cache bounds if not already set."""
        if not self._bounds_set:
            self._min_row, self._min_col, self._height, self._width = compute_bounds(
                grid_objects, self._object_type_names
            )
            self._bounds_set = True

    def render(
        self,
        step: int,
        grid_objects: Dict[int, dict],
        viewport_center_row: int | None = None,
        viewport_center_col: int | None = None,
        viewport_height: int | None = None,
        viewport_width: int | None = None,
    ) -> str:
        """Render the environment buffer and print to screen."""
        self._ensure_bounds(grid_objects)
        current_buffer = build_grid_buffer(
            grid_objects,
            self._object_type_names,
            self._min_row,
            self._min_col,
            self._height,
            self._width,
            viewport_center_row,
            viewport_center_col,
            viewport_height,
            viewport_width,
        )
        header = f"🎮 Metta AI Miniscope - Step: {step} 🎮"
        separator = "═" * (self._width * 2)
        frame_buffer = f"\033[2J\033[H{header}\n{separator}\n{current_buffer}\n{separator}"
        print(frame_buffer, end="", flush=True)
        self._last_buffer = current_buffer
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
            self._min_row,
            self._min_col,
            self._height,
            self._width,
            viewport_center_row,
            viewport_center_col,
            viewport_height,
            viewport_width,
        )

    def interactive_loop(
        self,
        env,
        get_actions_fn: Callable[[np.ndarray, Optional[int], Optional[int | tuple]], np.ndarray],
        max_steps: Optional[int] = None,
        target_fps: int = 4,
    ) -> Dict[str, any]:
        """Run interactive rendering loop with keyboard controls.

        Args:
            env: MettaGrid environment
            get_actions_fn: Function that takes (obs, selected_agent, manual_action_direction)
                           and returns actions for all agents
            max_steps: Maximum steps to run (None for unlimited)
            target_fps: Target frames per second when playing

        Returns:
            Dict with final statistics
        """
        return run_interactive_loop(env, self._object_type_names, get_actions_fn, max_steps, target_fps)
