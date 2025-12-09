"""
Ultra-Fast Tribal Village Environment - Direct Buffer Interface.

Eliminates ALL conversion overhead by using direct numpy buffer communication.
"""

import ctypes
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from gymnasium import spaces

import pufferlib

ACTION_VERB_COUNT = 7
ACTION_ARGUMENT_COUNT = 8
ACTION_SPACE_SIZE = ACTION_VERB_COUNT * ACTION_ARGUMENT_COUNT


class TribalVillageEnv(pufferlib.PufferEnv):
    """
    Ultra-fast tribal village environment using direct buffer interface.

    Eliminates conversion overhead by using pre-allocated numpy buffers
    that Nim reads/writes directly.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, buf=None):
        self.config = config or {}
        self.max_steps = self.config.get("max_steps", 1_000)
        self._render_mode = self.config.get("render_mode", "rgb_array")

        # Load the optimized Nim library - cross-platform
        import platform

        if platform.system() == "Darwin":
            lib_name = "libtribal_village.dylib"
        elif platform.system() == "Windows":
            lib_name = "libtribal_village.dll"
        else:
            lib_name = "libtribal_village.so"

        package_dir = Path(__file__).resolve().parent
        candidate_paths = [
            package_dir.parent / lib_name,
            package_dir / lib_name,
        ]

        lib_path = next((path for path in candidate_paths if path.exists()), None)
        if lib_path is None:
            searched = ", ".join(str(path) for path in candidate_paths)
            raise FileNotFoundError(f"Nim library not found. Searched: {searched}")

        self.lib = ctypes.CDLL(str(lib_path))
        self._setup_ctypes_interface()

        # Get environment dimensions
        self.total_agents = self.lib.tribal_village_get_num_agents()
        self.obs_layers = self.lib.tribal_village_get_obs_layers()
        self.obs_width = self.lib.tribal_village_get_obs_width()
        self.obs_height = self.lib.tribal_village_get_obs_height()

        # Map dims for full-map render
        try:
            self.map_width = int(self.lib.tribal_village_get_map_width())
            self.map_height = int(self.lib.tribal_village_get_map_height())
            self.render_scale = max(1, int(self.config.get("render_scale", 4)))
            height = self.map_height * self.render_scale
            width = self.map_width * self.render_scale
            self._rgb_frame = np.zeros((height, width, 3), dtype=np.uint8)
        except Exception:
            self.map_width = None
            self.map_height = None
            self.render_scale = 1
            self._rgb_frame = None

        # PufferLib controls all agents
        self.num_agents = self.total_agents
        self.agents = [f"agent_{i}" for i in range(self.total_agents)]
        self.possible_agents = self.agents.copy()

        # Define spaces - use direct observation shape (no sparse tokens!)
        self.single_observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.obs_layers, self.obs_width, self.obs_height),
            dtype=np.uint8,
        )
        self.single_action_space = spaces.Discrete(ACTION_SPACE_SIZE)
        self.is_continuous = False

        super().__init__(buf)

        # Set up joint action space like metta does
        self.action_space = pufferlib.spaces.joint_space(self.single_action_space, self.num_agents)
        if hasattr(self, "actions"):
            self.actions = self.actions.astype(np.int32)

        # PufferLib will set these buffers - don't allocate our own!
        self.observations: np.ndarray
        self.terminals: np.ndarray
        self.truncations: np.ndarray
        self.rewards: np.ndarray

        # Only allocate actions buffer (input to environment)
        self.actions_buffer = np.zeros(self.total_agents, dtype=np.uint8)

        # Initialize environment
        self.env_ptr = self.lib.tribal_village_create()
        if not self.env_ptr:
            raise RuntimeError("Failed to create Nim environment")

        self.step_count = 0

    @property
    def render_mode(self):
        return self._render_mode

    @render_mode.setter
    def render_mode(self, value):
        self._render_mode = value

    def render(self):
        """Render via Nim, avoiding duplication in Python.

        - 'rgb_array': calls Nim RGB export and returns an HxWx3 uint8 array
          of the full map (uses tile colors from the engine).
        - 'ansi': calls Nim ASCII renderer and returns a string.
        - otherwise: falls back to 'ansi'.
        """
        mode = getattr(self, "_render_mode", "ansi")

        # Prefer native RGB if requested and available
        if mode == "rgb_array" and getattr(self, "_rgb_frame", None) is not None:
            ptr = self._rgb_frame.ctypes.data_as(ctypes.c_void_p)
            width = int(self._rgb_frame.shape[1])
            height = int(self._rgb_frame.shape[0])
            try:
                ok = self.lib.tribal_village_render_rgb(self.env_ptr, ptr, width, height)
            except AttributeError:
                ok = 0
            if ok:
                return self._rgb_frame
            # fall through to ansi if RGB export missing

        buf_size = int(self.config.get("ansi_buffer_size", 1_000_000))
        cbuf = ctypes.create_string_buffer(buf_size)
        try:
            n_written = self.lib.tribal_village_render_ansi(
                self.env_ptr, ctypes.cast(cbuf, ctypes.c_void_p), ctypes.c_int32(buf_size)
            )
        except AttributeError:
            return "(render not available in Nim build)"

        if n_written <= 0:
            return ""
        return cbuf.value.decode("utf-8", errors="replace")

    def _setup_ctypes_interface(self):
        """Setup ctypes for direct buffer functions."""

        # tribal_village_create() -> pointer
        self.lib.tribal_village_create.argtypes = []
        self.lib.tribal_village_create.restype = ctypes.c_void_p

        # tribal_village_reset_and_get_obs(env, obs_buf, rewards_buf, terminals_buf, truncations_buf) -> int32
        self.lib.tribal_village_reset_and_get_obs.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.lib.tribal_village_reset_and_get_obs.restype = ctypes.c_int32

        # tribal_village_step_with_pointers(env, actions_buf, obs_buf,
        #                                   rewards_buf, terminals_buf,
        #                                   truncations_buf) -> int32
        self.lib.tribal_village_step_with_pointers.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.lib.tribal_village_step_with_pointers.restype = ctypes.c_int32

        # tribal_village_destroy(env) -> void
        self.lib.tribal_village_destroy.argtypes = [ctypes.c_void_p]
        self.lib.tribal_village_destroy.restype = None

        # Map dimensions and RGB render
        try:
            self.lib.tribal_village_get_map_width.argtypes = []
            self.lib.tribal_village_get_map_width.restype = ctypes.c_int32
            self.lib.tribal_village_get_map_height.argtypes = []
            self.lib.tribal_village_get_map_height.restype = ctypes.c_int32
            self.lib.tribal_village_render_rgb.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int32,
                ctypes.c_int32,
            ]
            self.lib.tribal_village_render_rgb.restype = ctypes.c_int32
        except AttributeError:
            pass

        # tribal_village_render_ansi(env, out_buf, buf_len) -> int32
        try:
            self.lib.tribal_village_render_ansi.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32]
            self.lib.tribal_village_render_ansi.restype = ctypes.c_int32
        except AttributeError:
            pass

        # Dimension getters
        for func_name in [
            "tribal_village_get_num_agents",
            "tribal_village_get_obs_layers",
            "tribal_village_get_obs_width",
            "tribal_village_get_obs_height",
        ]:
            getattr(self.lib, func_name).argtypes = []
            getattr(self.lib, func_name).restype = ctypes.c_int32

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Ultra-fast reset using direct buffers."""
        self.step_count = 0

        # Get PufferLib managed buffer pointers
        obs_ptr = self.observations.ctypes.data_as(ctypes.c_void_p)
        rewards_ptr = self.rewards.ctypes.data_as(ctypes.c_void_p)
        terminals_ptr = self.terminals.ctypes.data_as(ctypes.c_void_p)
        truncations_ptr = self.truncations.ctypes.data_as(ctypes.c_void_p)

        # Direct buffer reset - no conversions
        success = self.lib.tribal_village_reset_and_get_obs(
            self.env_ptr, obs_ptr, rewards_ptr, terminals_ptr, truncations_ptr
        )
        if not success:
            raise RuntimeError("Failed to reset Nim environment")

        # Return observations as views of PufferLib buffers (no copying!)
        observations = {f"agent_{i}": self.observations[i] for i in range(self.num_agents)}
        info = {f"agent_{i}": {} for i in range(self.num_agents)}

        return observations, info

    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """Ultra-fast step using direct buffers."""
        self.step_count += 1

        # Clear actions buffer
        self.actions_buffer.fill(0)

        # Direct action setting (no dict overhead)
        for i in range(self.num_agents):
            agent_key = f"agent_{i}"
            if agent_key in actions:
                action_value = int(np.asarray(actions[agent_key]).reshape(()))
                if action_value < 0 or action_value >= self.single_action_space.n:
                    action_value = 0
                self.actions_buffer[i] = np.uint8(action_value)

        # Get PufferLib managed buffer pointers
        actions_ptr = self.actions_buffer.ctypes.data_as(ctypes.c_void_p)
        obs_ptr = self.observations.ctypes.data_as(ctypes.c_void_p)
        rewards_ptr = self.rewards.ctypes.data_as(ctypes.c_void_p)
        terminals_ptr = self.terminals.ctypes.data_as(ctypes.c_void_p)
        truncations_ptr = self.truncations.ctypes.data_as(ctypes.c_void_p)

        # Direct buffer step - no conversions
        success = self.lib.tribal_village_step_with_pointers(
            self.env_ptr,
            actions_ptr,
            obs_ptr,
            rewards_ptr,
            terminals_ptr,
            truncations_ptr,
        )
        if not success:
            raise RuntimeError("Failed to step Nim environment")

        # Return results as views of PufferLib buffers (no copying!)
        observations = {f"agent_{i}": self.observations[i] for i in range(self.num_agents)}
        rewards = {f"agent_{i}": float(self.rewards[i]) for i in range(self.num_agents)}
        terminated = {f"agent_{i}": bool(self.terminals[i]) for i in range(self.num_agents)}
        truncated = {
            f"agent_{i}": bool(self.truncations[i]) or (self.step_count >= self.max_steps)
            for i in range(self.num_agents)
        }
        infos = {f"agent_{i}": {} for i in range(self.num_agents)}

        return observations, rewards, terminated, truncated, infos

    def close(self):
        """Clean up the environment."""
        if hasattr(self, "env_ptr") and self.env_ptr:
            self.lib.tribal_village_destroy(self.env_ptr)
            self.env_ptr = None


def make_tribal_village_env(config: Optional[Dict[str, Any]] = None, **kwargs) -> TribalVillageEnv:
    """Factory function for ultra-fast tribal village environment."""
    if config is None:
        config = {}
    config.update(kwargs)
    return TribalVillageEnv(config=config)
