"""GUI renderer using mettascope."""

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from mettagrid import MettaGridEnv
from mettagrid.renderer.renderer import Renderer
from mettagrid.util.grid_object_formatter import format_grid_object


class MettascopeRenderer(Renderer):
    """Renderer for GUI mode using mettascope."""

    def __init__(self):
        nim_root, _nim_search_paths = _resolve_nim_root()
        nim_bindings_path = nim_root / "bindings" / "generated" if nim_root else None
        sys.path.insert(0, str(nim_bindings_path))
        import mettascope2

        self._mettascope = mettascope2
        self._should_continue = True
        self._env = None
        # Store the data directory for mettascope
        self._data_dir = str(nim_root / "data") if nim_root else "."
        # Store user actions persistently
        self._user_actions = {}  # Dict mapping agent_id -> (action_id, action_param)
        self._current_step = 0

    def on_episode_start(self, env: "MettaGridEnv") -> None:
        initial_replay = {
            "version": 2,
            "action_names": env.action_names,
            "item_names": env.resource_names,
            "type_names": env.object_type_names,
            "map_size": [env.map_width, env.map_height],
            "num_agents": env.num_agents,
            "max_steps": 0,
            "mg_config": env.mg_config.model_dump(mode="json"),
            "objects": [],
        }

        # mettascope2.init requires data_dir and replay arguments
        self.response = self._mettascope.init(self._data_dir, json.dumps(initial_replay))
        self._should_continue = not self.response.should_close
        self._env = env
        self._user_actions = {}
        self._current_step = 0

    def render(self) -> None:
        """Render current state and capture user input."""
        assert self._env is not None
        # Generate replay data for current state
        grid_objects = []
        total_rewards = self._env.get_episode_rewards()

        # Use zeros as placeholders for actions/rewards since we're rendering the current state
        placeholder_actions = np.zeros((self._env.num_agents, 2), dtype=np.int32)
        placeholder_rewards = np.zeros(self._env.num_agents)

        for grid_object in self._env.grid_objects().values():
            grid_objects.append(
                format_grid_object(
                    grid_object, placeholder_actions, self._env.action_success, placeholder_rewards, total_rewards
                )
            )

        step_replay = {"step": self._current_step, "objects": grid_objects}

        # Render and get user input
        self.response = self._mettascope.render(self._current_step, json.dumps(step_replay))
        if self.response.should_close:
            self._should_continue = False
            return

        # Store user actions to be applied in the next step
        if self.response.actions:
            for action in self.response.actions:
                self._user_actions[action.agent_id] = (action.action_id, action.argument)

        self._current_step += 1

    def get_user_actions(self) -> dict[int, tuple[int, int]]:
        """Get the current user actions for all agents.

        Returns:
            Dictionary mapping agent_id to (action_id, action_param)
        """
        return self._user_actions.copy()

    def on_step(
        self,
        current_step: int,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        infos: Dict[str, Any],
    ) -> None:
        pass

    def should_continue(self) -> bool:
        return self._should_continue

    def on_episode_end(self, infos: Dict[str, Any]) -> None:
        self._env = None


# Find the Nim bindings. Published wheels bundle the generated artifacts under
# ``mettagrid/nim/mettascope`` (because our PEP‑517 backend copies the Nim
# project into ``python/src`` during the build).  Editable installs, however,
# serve the package straight from the repository checkout where the canonical
# sources remain in ``packages/mettagrid/nim/mettascope`` and the copy step
# never runs.  To support both layouts we try the packaged location first and,
# if it is missing the bindings, walk upwards looking for the repository copy.
package_root = Path(__file__).resolve().parent


def _resolve_nim_root() -> tuple[Optional[Path], list[Path]]:
    search_paths: list[Path] = []

    packaged = package_root / "nim" / "mettascope"
    search_paths.append(packaged)
    if (packaged / "bindings" / "generated").exists():
        return packaged, search_paths

    current = package_root
    for _ in range(8):
        candidate = current / "packages" / "mettagrid" / "nim" / "mettascope"
        search_paths.append(candidate)
        if candidate.exists():
            return candidate, search_paths
        if current == current.parent:
            break
        current = current.parent
    return None, search_paths


# # Type stubs for static analysis
# if TYPE_CHECKING:
#     from typing import Any

#     def init(replay: Any) -> Any: ...
#     def render(step: int, replay_step: Any) -> Any: ...

#     class Mettascope2Error(Exception): ...
# else:
#     # Runtime import
#     if nim_bindings_path and nim_bindings_path.exists():
#         # Insert at the beginning to ensure it's found first
#         sys.path.insert(0, str(nim_bindings_path))

#         try:
#             # Import the mettascope2 module
#             import mettascope2

#             # Verify the module has the expected attributes
#             required_attrs = ["init", "render", "Mettascope2Error"]
#             missing_attrs = [attr for attr in required_attrs if not hasattr(mettascope2, attr)]

#             if missing_attrs:
#                 # List what attributes are actually available
#                 available_attrs = [attr for attr in dir(mettascope2) if not attr.startswith("_")]
#                 raise ImportError(
#                     f"mettascope2 module is missing required attributes: {missing_attrs}. "
#                     f"Available attributes: {available_attrs or 'none'}. "
#                     f"The Nim bindings may need to be regenerated."
#                 )

#             # Re-export the functions and classes
#             def init(replay):
#                 return mettascope2.init(data_dir=str(nim_root / "data"), replay=replay)

#             render = mettascope2.render
#             Mettascope2Error = mettascope2.Mettascope2Error

#         except ImportError as e:
#             raise ImportError(
#                 f"Failed to import mettascope2 from {nim_bindings_path}: {e}. "
#                 "Ensure the Nim bindings have been properly generated."
#             ) from e
#         finally:
#             # Remove the path from sys.path to avoid polluting it
#             if str(nim_bindings_path) in sys.path:
#                 sys.path.remove(str(nim_bindings_path))
#     else:
#         searched = ", ".join(str(path) for path in _nim_search_paths)
#         raise ImportError(
#             "Could not find mettascope bindings. "
#             f"Searched locations: {searched}. "
#             "Ensure the Nim bindings have been generated by running the appropriate build command."
#         )
