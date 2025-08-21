"""
MettaGridCore - Core Python wrapper for MettaGrid C++ environment.

This class provides the base functionality for all framework-specific adapters,
without any training-specific features or framework dependencies.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import gymnasium
import numpy as np
from gymnasium import spaces

from metta.mettagrid.level_builder import Level
from metta.mettagrid.mettagrid_c import MettaGrid as MettaGridCpp
from metta.mettagrid.mettagrid_c_config import from_mettagrid_config

logger = logging.getLogger("MettaGridCore")

ReadableFmt = Literal["json", "yaml"]


def _infer_fmt(path: Path, fmt: ReadableFmt | None) -> ReadableFmt:
    if fmt in ("json", "yaml"):
        return fmt
    suf = path.suffix.lower()
    if suf in {".yml", ".yaml"}:
        return "yaml"
    return "json"


def save_3d_array_readable(
    path: str | Path,
    data: Any,
    *,
    fmt: ReadableFmt | None = None,
    round_fp: int | None = None,
) -> Path:
    """Save a 3D array to a *human‑readable* text file (JSON or YAML).

    - Preserves shape & dtype
    - Uses nested lists for readability (depth -> rows -> cols)
    - Optionally rounds floating‑point values for smaller/cleaner files

    Parameters
    ----------
    path : str | Path
        Output file path. If no extension is given, defaults to .json.
    data : Any
        Array-like object convertible to a NumPy array.
    fmt : {"json", "yaml"} | None
        Force output format. If None, inferred from the file extension.
    round_fp : int | None
        If provided, round floats to this many decimals before saving.

    Returns
    -------
    Path
        The resolved output path.
    """
    arr = np.asarray(data)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got {arr.ndim}D")

    p = Path(path)
    chosen_fmt = _infer_fmt(p, fmt)
    if p.suffix == "":
        p = p.with_suffix(".json" if chosen_fmt == "json" else ".yaml")

    # Convert to nested lists for readability
    nested: Any = arr.tolist()

    if round_fp is not None:
        # Recursively round floats inside nested lists
        def _round(v: Any) -> Any:
            if isinstance(v, float):
                return round(v, round_fp)
            if isinstance(v, list):
                return [_round(x) for x in v]
            return v

        nested = _round(nested)

    payload = {
        "shape": list(arr.shape),
        "dtype": arr.dtype.name,
        "data": nested,
    }

    if chosen_fmt == "json":
        with p.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
    else:
        try:
            import yaml  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "PyYAML is required for YAML output. Install with `pip install pyyaml` or use fmt='json'."
            ) from e
        with p.open("w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)

    return p.resolve()


# --- Example ---
# x = np.arange(2*3*4, dtype=np.float32).reshape(2, 3, 4)
# save_3d_array_readable("array.yaml", x, fmt="yaml", round_fp=None)
# x2 = load_3d_array_readable("array.yaml")
# assert np.array_equal(x, x2)


class MettaGridCore:
    """
    Core MettaGrid functionality without any training features.

    This class provides pure C++ wrapper functionality without training-specific
    features like stats writing, replay writing, or curriculum management.
    Use MettaGridEnv for training functionality.
    """

    def __init__(
        self,
        level: Level,
        game_config_dict: Dict[str, Any],
        render_mode: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Initialize core MettaGrid functionality.

        Args:
            level: Level to use for the environment
            game_config_dict: Game configuration dictionary
            render_mode: Rendering mode (None, "human", "miniscope")
            **kwargs: Additional arguments passed to subclasses
        """
        self._render_mode = render_mode
        self._level = level
        self._renderer = None
        self._map_labels: List[str] = level.labels
        self._current_seed: int = 1337
        # Environment metadata
        self.labels: List[str] = []
        self._should_reset = False

        # Initialize renderer class if needed (before C++ env creation)
        if self._render_mode is not None:
            self._initialize_renderer()

        # Create C++ environment immediately
        self._c_env_instance: Optional[MettaGridCpp] = self._create_c_env(game_config_dict)

    @property
    def c_env(self) -> MettaGridCpp:
        """Get core environment instance, raising error if not initialized."""
        if self._c_env_instance is None:
            raise RuntimeError("Environment not initialized")
        return self._c_env_instance

    def _initialize_renderer(self) -> None:
        """Initialize renderer class based on render mode."""
        self._renderer = None
        self._renderer_class = None

        if self._render_mode == "human":
            from metta.mettagrid.renderer.nethack import NethackRenderer

            self._renderer_class = NethackRenderer
        elif self._render_mode == "miniscope":
            from metta.mettagrid.renderer.miniscope import MiniscopeRenderer

            self._renderer_class = MiniscopeRenderer

    def _create_c_env(self, game_config_dict: Dict[str, Any], seed: Optional[int] = None) -> MettaGridCpp:
        """
        Create a new MettaGridCpp instance.

        Args:
            game_config_dict: Game configuration dictionary
            seed: Random seed for environment

        Returns:
            New MettaGridCpp instance
        """
        level = self._level
        seed = 1337

        # Validate number of agents
        level_agents = np.count_nonzero(np.char.startswith(level.grid, "agent"))
        assert game_config_dict["num_agents"] == level_agents, (
            f"Number of agents {game_config_dict['num_agents']} does not match number of agents in map {level_agents}"
        )

        # Ensure we have a dict
        if not isinstance(game_config_dict, dict):
            raise ValueError(f"Expected dict for game config, got {type(game_config_dict)}")

        # Create C++ config
        try:
            c_cfg = from_mettagrid_config(game_config_dict)
        except Exception as e:
            logger.error(f"Error creating C++ config: {e}")
            logger.error(f"Game config: {game_config_dict}")
            raise e

        # Create C++ environment
        current_seed = seed if seed is not None else self._current_seed
        c_env = MettaGridCpp(c_cfg, level.grid.tolist(), current_seed)

        def rollout_fingerprint_multidiscrete(env_ctor, seed=123, T=50, use_fixed_actions=True):
            env = env_ctor(seed=seed)
            obs, info = env.reset()

            # infer N (#agents)
            N = getattr(env, "num_agents", None)
            if N is None:
                sample = obs["grid_obs"] if isinstance(obs, dict) else obs
                N = sample.shape[0]

            assert isinstance(env.action_space, gymnasium.spaces.multi_discrete.MultiDiscrete)
            nvec = np.asarray(env.action_space.nvec, dtype=np.int32)  # int32 for C++ binding
            K = int(nvec.shape[0])

            if use_fixed_actions:

                def next_action(t):
                    a = np.zeros((N, K), dtype=np.int32)
                    return np.ascontiguousarray(a)
            else:
                rng = np.random.default_rng(0)

                def next_action(t):
                    a = rng.integers(low=0, high=nvec, size=(N, K), endpoint=False, dtype=np.int32)
                    return np.ascontiguousarray(a)

            # hash helper
            def fp(x, i):
                arr = x["grid_obs"] if isinstance(x, dict) else x
                save_3d_array_readable(f"/Users/localmini/gridobs/grid_obs_{i}.txt", arr)
                return hashlib.sha256(arr.tobytes()).hexdigest()[:16]

            fps = [fp(obs, 0)]
            i = 1
            for t in range(T):
                a = next_action(t)  # shape (N, K), int32, C-contig
                obs, rew, done, trunc, info = env.step(a)
                fps.append(fp(obs, i))
                i += 1
                if np.asarray(done).ndim > 0 and np.asarray(done).any():
                    obs, info = env.reset()
            return fps

        #        f1 = rollout_fingerprint_multidiscrete(
        #            lambda seed: MettaGridCpp(c_cfg, level.grid.tolist(), 1337), seed=1337, T=900, use_fixed_actions=True
        #        )
        #        f2 = rollout_fingerprint_multidiscrete(
        #            lambda seed: MettaGridCpp(c_cfg, level.grid.tolist(), 1337), seed=1337, T=900, use_fixed_actions=True
        #        )
        #        print("trajectories identical:", f1 == f2)  #
        #        if f1 != f2:
        #            print("trajectories not identical:", f1, f2)

        # Initialize renderer if needed
        if (
            self._render_mode is not None
            and self._renderer is None
            and hasattr(self, "_renderer_class")
            and self._renderer_class is not None
        ):
            self._renderer = self._renderer_class(c_env.object_type_names())

        return c_env

    def reset(self, game_config_dict: Dict[str, Any], seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment.

        Args:
            game_config_dict: Game configuration dictionary
            seed: Random seed

        Returns:
            Tuple of (observations, info)
        """
        # Recreate C++ environment with new config
        self._c_env_instance = self._create_c_env(game_config_dict, seed)

        # Update seed
        self._current_seed = seed or 0

        # Reset flags
        self._should_reset = False

        # Get initial observations from core environment
        obs, infos = self._c_env_instance.reset()

        return obs, infos

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Execute one timestep of the environment dynamics with the given actions.

        Args:
            actions: A numpy array of shape (num_agents, 2) with dtype np.int32

        Returns:
            Tuple of (observations, rewards, terminals, truncations, infos)
        """
        # Execute step in core environment
        obs, rewards, terminals, truncations, _ = self._c_env_instance.step(actions)

        # Check for episode completion
        infos = {}
        if terminals.all() or truncations.all():
            self._should_reset = True

        return obs, rewards, terminals, truncations, infos

    def render(self) -> Optional[str]:
        """Render the environment."""
        if self._renderer is None or self._c_env_instance is None:
            return None

        return self._renderer.render(self._c_env_instance.current_step, self._c_env_instance.grid_objects())

    def close(self) -> None:
        """Close the environment."""
        if self._c_env_instance is not None:
            # Clean up any resources if needed
            self._c_env_instance = None

    # Properties that expose C++ environment functionality
    @property
    def done(self) -> bool:
        """Check if environment needs reset."""
        return self._should_reset

    @property
    def render_mode(self) -> Optional[str]:
        """Get render mode."""
        return self._render_mode

    @property
    def core_env(self) -> Optional[MettaGridCpp]:
        """Get core environment instance."""
        return self._c_env_instance

    # Properties that delegate to core environment
    @property
    def max_steps(self) -> int:
        return self._c_env_instance.max_steps

    @property
    def num_agents(self) -> int:
        return self._c_env_instance.num_agents

    @property
    def obs_width(self) -> int:
        return self._c_env_instance.obs_width

    @property
    def obs_height(self) -> int:
        return self._c_env_instance.obs_height

    @property
    def map_width(self) -> int:
        return self._c_env_instance.map_width

    @property
    def map_height(self) -> int:
        return self._c_env_instance.map_height

    @property
    def _observation_space(self) -> spaces.Box:
        """Internal observation space - use single_observation_space for PufferEnv compatibility."""
        return self._c_env_instance.observation_space

    @property
    def _action_space(self) -> spaces.MultiDiscrete:
        """Internal action space - use single_action_space for PufferEnv compatibility."""
        return self._c_env_instance.action_space

    @property
    def action_names(self) -> List[str]:
        return self._c_env_instance.action_names()

    @property
    def max_action_args(self) -> List[int]:
        action_args_array = self._c_env_instance.max_action_args()
        return [int(x) for x in action_args_array]

    @property
    def object_type_names(self) -> List[str]:
        return self._c_env_instance.object_type_names()

    @property
    def inventory_item_names(self) -> List[str]:
        return self._c_env_instance.inventory_item_names()

    @property
    def feature_normalizations(self) -> Dict[int, float]:
        """Get feature normalizations from C++ environment."""
        # Check if the C++ environment has the direct method
        if hasattr(self._c_env_instance, "feature_normalizations"):
            return self._c_env_instance.feature_normalizations()
        else:
            # Fallback to extracting from feature_spec (slower)
            feature_spec = self._c_env_instance.feature_spec()
            return {int(spec["id"]): float(spec["normalization"]) for spec in feature_spec.values()}

    @property
    def initial_grid_hash(self) -> int:
        return self._c_env_instance.initial_grid_hash

    @property
    def action_success(self) -> List[bool]:
        return self._c_env_instance.action_success()

    def get_observation_features(self) -> Dict[str, Dict]:
        """
        Build the features dictionary for initialize_to_environment.

        Returns:
            Dictionary mapping feature names to their properties
        """
        # Get feature spec from C++ environment
        feature_spec = self._c_env_instance.feature_spec()

        features = {}
        for feature_name, feature_info in feature_spec.items():
            feature_dict: Dict[str, Any] = {"id": feature_info["id"]}

            # Add normalization if present
            if "normalization" in feature_info:
                feature_dict["normalization"] = feature_info["normalization"]

            features[feature_name] = feature_dict

        return features

    @property
    def grid_objects(self) -> Dict[int, Dict[str, Any]]:
        """Get grid objects information."""
        return self._c_env_instance.grid_objects()
