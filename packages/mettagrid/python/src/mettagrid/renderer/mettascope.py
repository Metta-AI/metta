"""GUI renderer using mettascope."""

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from mettagrid.renderer.renderer import Renderer
from mettagrid.simulator import Action
from mettagrid.util.grid_object_formatter import format_grid_object


class MettascopeRenderer(Renderer):
    """Renderer for GUI mode using mettascope."""

    def __init__(self):
        super().__init__()
        nim_root, _nim_search_paths = _resolve_nim_root()
        nim_bindings_path = nim_root / "bindings" / "generated" if nim_root else None
        sys.path.insert(0, str(nim_bindings_path))
        import mettascope

        self._mettascope = mettascope
        self._data_dir = str(nim_root / "data") if nim_root else "."

    def on_episode_start(self) -> None:
        assert self._sim is not None

        # Get the GameConfig from MettaGridConfig
        game_config = self._sim.config.game
        game_config_dict = game_config.model_dump(mode="json", exclude_none=True)

        # Convert snake_case to camelCase for Nim compatibility
        field_mappings = {
            "vibe_names": "vibeNames",
            "resource_names": "resourceNames",
            "num_agents": "numAgents",
            "max_steps": "maxSteps",
        }
        for snake, camel in field_mappings.items():
            if snake in game_config_dict:
                game_config_dict[camel] = game_config_dict.pop(snake)

        # Extract obs config to top level for Nim compatibility
        if "obs" in game_config_dict:
            obs_config = game_config_dict.pop("obs")
            if "width" in obs_config:
                game_config_dict["obsWidth"] = obs_config["width"]
            if "height" in obs_config:
                game_config_dict["obsHeight"] = obs_config["height"]

        initial_replay = {
            "version": 2,
            "action_names": list(self._sim.action_ids.keys()),
            "item_names": self._sim.resource_names,
            "type_names": self._sim.object_type_names,
            "map_size": [
                self._sim.map_width,
                self._sim.map_height,
            ],
            "num_agents": self._sim.num_agents,
            "max_steps": 0,
            "mg_config": {
                "label": "MettaGrid Replay",
                "game": game_config_dict,
            },
            "objects": [],
        }

        # mettascope.init requires data_dir and replay arguments
        json_str = json.dumps(initial_replay, allow_nan=False)
        self.response = self._mettascope.init(self._data_dir, json_str)

    def render(self) -> None:
        """Render current state and capture user input."""
        assert self._sim is not None

        # Generate replay data for current state
        grid_objects = []
        total_rewards = self._sim.episode_rewards

        # Use zeros as placeholders for actions/rewards since we're rendering the current state
        placeholder_actions = np.zeros((self._sim.num_agents, 2), dtype=np.int32)
        placeholder_rewards = np.zeros(self._sim.num_agents)

        for grid_object in self._sim.grid_objects().values():
            grid_objects.append(
                format_grid_object(
                    grid_object,
                    placeholder_actions,
                    self._sim.action_success,
                    placeholder_rewards,
                    total_rewards,
                )
            )

        step_replay = {"step": self._sim.current_step, "objects": grid_objects}

        # Render and get user input
        self.response = self._mettascope.render(self._sim.current_step, json.dumps(step_replay, allow_nan=False))
        if self.response.should_close:
            self._sim.end_episode()
            return

        # Apply user actions immediately (overriding any policy actions)
        if self.response.actions:
            for action in self.response.actions:
                # ctypes c_char_p returns bytes, we need to decode immediately
                # before the memory gets freed
                action_name_raw = action.action_name

                if isinstance(action_name_raw, bytes):
                    # Find null terminator and decode only up to there
                    null_idx = action_name_raw.find(b"\x00")
                    if null_idx > 0:
                        action_name = action_name_raw[:null_idx].decode("utf-8", errors="ignore")
                    else:
                        action_name = action_name_raw.decode("utf-8", errors="ignore")
                elif isinstance(action_name_raw, str):
                    action_name = action_name_raw
                else:
                    print(f"WARNING: Unexpected action_name type: {type(action_name_raw)}")
                    continue

                if not action_name:
                    continue

                print(f"DEBUG: Setting action '{action_name}' for agent {action.agent_id}")
                try:
                    self._sim.agent(action.agent_id).set_action(Action(name=action_name))
                except KeyError as e:
                    print(f"ERROR: Unknown action '{action_name}' - {e}")
                    available_actions = [a for a in self._sim.action_ids.keys() if "change_vibe" in a]
                    print(f"Available change_vibe actions: {available_actions}")
                    continue


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

#     class MettascopeError(Exception): ...
# else:
#     # Runtime import
#     if nim_bindings_path and nim_bindings_path.exists():
#         # Insert at the beginning to ensure it's found first
#         sys.path.insert(0, str(nim_bindings_path))

#         try:
#             # Import the mettascope module
#             import mettascope

#             # Verify the module has the expected attributes
#             required_attrs = ["init", "render", "MettascopeError"]
#             missing_attrs = [attr for attr in required_attrs if not hasattr(mettascope, attr)]

#             if missing_attrs:
#                 # List what attributes are actually available
#                 available_attrs = [attr for attr in dir(mettascope) if not attr.startswith("_")]
#                 raise ImportError(
#                     f"mettascope module is missing required attributes: {missing_attrs}. "
#                     f"Available attributes: {available_attrs or 'none'}. "
#                     f"The Nim bindings may need to be regenerated."
#                 )

#             # Re-export the functions and classes
#             def init(replay):
#                 return mettascope.init(data_dir=str(nim_root / "data"), replay=replay)

#             render = mettascope.render
#             MettascopeError = mettascope.MettascopeError

#         except ImportError as e:
#             raise ImportError(
#                 f"Failed to import mettascope from {nim_bindings_path}: {e}. "
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
