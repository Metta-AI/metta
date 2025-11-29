"""
MettaGridPufferEnv - PufferLib integration for MettaGrid.

This class provides PufferLib compatibility for MettaGrid environments using
the Simulation class. This allows MettaGrid environments to be used
directly with PufferLib training infrastructure.

Provides:
 - Auto-reset on episode completion
 - Persistent buffers for re-use between resets

Architecture:
- MettaGridPufferEnv wraps Simulation and provides PufferEnv interface
- This enables MettaGridPufferEnv to work seamlessly with PufferLib training code

For users:
- Use MettaGridPufferEnv directly with PufferLib (it inherits PufferLib functionality)
- Alternatively, use PufferLib's MettaPuff wrapper for additional PufferLib features:
  https://github.com/PufferAI/PufferLib/blob/main/pufferlib/environments/metta/environment.py

This avoids double-wrapping while maintaining full PufferLib compatibility.
"""

from __future__ import annotations

import fcntl
import hashlib
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
from gymnasium.spaces import Box, Discrete
from typing_extensions import override

from mettagrid.config.mettagrid_config import EnvSupervisorConfig, MettaGridConfig
from mettagrid.mettagrid_c import (
    dtype_actions,
    dtype_masks,
    dtype_observations,
    dtype_rewards,
    dtype_terminals,
    dtype_truncations,
)
from mettagrid.policy.loader import initialize_or_load_policy
from mettagrid.policy.policy import MultiAgentPolicy, PolicySpec
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Simulation, Simulator
from mettagrid.simulator.simulator import Buffers
from pufferlib.pufferlib import PufferEnv  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


def _patch_agents_py_to_skip_compilation(agents_py: Path, nim_dir: Path) -> None:
    """Patch agents.py to check for existing compiled library before compiling.

    This prevents agents.py from recompiling when the library already exists,
    which avoids race conditions in multiprocessing environments.
    """
    # Use file locking to prevent multiple processes from patching simultaneously
    patch_lock_file = agents_py.parent / ".agents_patch.lock"
    lock_fd = None
    lock_acquired = False

    try:
        # Acquire lock before patching
        try:
            patch_lock_file.parent.mkdir(parents=True, exist_ok=True)
            lock_fd = open(patch_lock_file, "a+")
            try:
                fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                lock_acquired = True
            except BlockingIOError:
                # Another process is patching, wait for it
                logger.debug("Another process is patching agents.py, waiting...")
                fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)
                lock_acquired = True
            except OSError:
                # File locking not available, proceed without lock
                lock_acquired = False
        except Exception as e:
            logger.debug(f"Could not acquire patch lock: {e}")
            lock_acquired = False

        # Re-check if already patched (another process may have patched it)
        try:
            content = agents_py.read_text()
            if "_lib_exists" in content and "Skipping compilation" in content:
                return
        except Exception:
            return

        # Find the compilation section - look for the pattern where compilation happens
        # We'll insert a check before the compilation code
        lines = content.split("\n")
        new_lines = []
        i = 0
        found_compile = False

        while i < len(lines):
            line = lines[i]
            # Check if this is the start of compilation section
            if "# Compile the Nim code" in line:
                found_compile = True
                # Compute relative path from agents.py directory to nim_dir
                nim_rel_path = None
                nim_abs_path = None
                try:
                    agents_dir = agents_py.parent.resolve()
                    nim_dir_resolved = nim_dir.resolve()
                    if nim_dir_resolved.is_relative_to(agents_dir):
                        nim_rel_path = str(nim_dir_resolved.relative_to(agents_dir))
                    else:
                        # If nim_dir is not a subdirectory, store absolute path
                        nim_abs_path = str(nim_dir_resolved)
                except Exception:
                    # Fallback: try relative path
                    try:
                        if nim_dir.is_relative_to(agents_py.parent):
                            nim_rel_path = str(nim_dir.relative_to(agents_py.parent))
                        else:
                            nim_abs_path = str(nim_dir.resolve())
                    except Exception:
                        pass

                # Insert check before compilation
                # Check both in current_dir (where agents.py is) and nim_dir (where compilation happens)
                check_lines = [
                    "",
                    "# Check if library already exists before compiling",
                    "_lib_paths = [",
                    # Check in current_dir (extraction root)
                    '    os.path.join(current_dir, "bindings/generated/libdinky_agents.so"),',
                    '    os.path.join(current_dir, "bindings/generated/libnim_agents.so"),',
                    '    os.path.join(current_dir, "bindings/generated/libagents.so"),',
                    '    os.path.join(current_dir, "libdinky_agents.so"),',
                    '    os.path.join(current_dir, "libnim_agents.so"),',
                    '    os.path.join(current_dir, "libagents.so"),',
                ]

                # Add nim_dir paths (relative or absolute)
                if nim_rel_path:
                    lib_names = ["libdinky_agents.so", "libnim_agents.so", "libagents.so"]
                    for lib_name in lib_names:
                        check_lines.append(
                            f'    os.path.join(current_dir, "{nim_rel_path}", "bindings/generated/{lib_name}"),'
                        )
                        check_lines.append(f'    os.path.join(current_dir, "{nim_rel_path}", "{lib_name}"),')
                elif nim_abs_path:
                    check_lines.extend(
                        [
                            f'    os.path.join("{nim_abs_path}", "bindings/generated/libdinky_agents.so"),',
                            f'    os.path.join("{nim_abs_path}", "bindings/generated/libnim_agents.so"),',
                            f'    os.path.join("{nim_abs_path}", "bindings/generated/libagents.so"),',
                            f'    os.path.join("{nim_abs_path}", "libdinky_agents.so"),',
                            f'    os.path.join("{nim_abs_path}", "libnim_agents.so"),',
                            f'    os.path.join("{nim_abs_path}", "libagents.so"),',
                        ]
                    )

                check_lines.extend(
                    [
                        "]",
                        "_lib_exists = any(os.path.exists(p) and os.path.getsize(p) > 0 for p in _lib_paths)",
                        "",
                        "if not _lib_exists:",
                    ]
                )
                new_lines.extend(check_lines)
                # Add the compilation code with indentation
                while i < len(lines):
                    current_line = lines[i]
                    # Stop before os.chdir(pwd) which comes after compilation
                    if current_line.strip() == "os.chdir(pwd)":
                        break
                    # Indent compilation lines
                    if current_line.strip():
                        new_lines.append("    " + current_line)
                    else:
                        new_lines.append(current_line)
                    i += 1
                # Add else clause before os.chdir
                new_lines.append("else:")
                new_lines.append('    print("Skipping compilation - library already exists")')
                # Add os.chdir line without indentation
                if i < len(lines) and lines[i].strip() == "os.chdir(pwd)":
                    new_lines.append(lines[i])
                    i += 1
                continue
            new_lines.append(line)
            i += 1

        # Only write if we made changes
        if found_compile:
            new_content = "\n".join(new_lines)
            if new_content != content:
                agents_py.write_text(new_content)
                logger.info("Patched agents.py to skip compilation if library exists")
            else:
                logger.debug("agents.py already patched or no changes needed")
        else:
            logger.warning("Could not find '# Compile the Nim code' marker in agents.py to patch")
    except Exception as e:
        logger.warning(f"Failed to patch agents.py: {e}, compilation may happen multiple times")
    finally:
        # Release lock if we acquired it
        if lock_fd is not None:
            try:
                if lock_acquired:
                    fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            lock_fd.close()


# Type compatibility assertions - ensure C++ types match PufferLib expectations
# PufferLib expects particular datatypes - see pufferlib/vector.py
assert dtype_observations == np.dtype(np.uint8)
assert dtype_terminals == np.dtype(np.bool_)
assert dtype_truncations == np.dtype(np.bool_)
assert dtype_rewards == np.dtype(np.float32)
assert dtype_actions == np.dtype(np.int32)


class MettaGridPufferEnv(PufferEnv):
    """
    Wraps the Simulator class to provide PufferLib compatibility.

    Inherits from pufferlib.PufferEnv: High-performance vectorized environment interface
      https://github.com/PufferAI/PufferLib/blob/main/pufferlib/environments.py
    """

    def __init__(
        self,
        simulator: Simulator,
        cfg: MettaGridConfig,
        env_supervisor_cfg: Optional[EnvSupervisorConfig] = None,
        buf: Any = None,
        seed: int = 0,
    ):
        # Support both Simulation and MettaGridConfig for backwards compatibility
        self._simulator = simulator
        self._current_cfg = cfg
        self._current_seed = seed
        self._env_supervisor_cfg = env_supervisor_cfg or EnvSupervisorConfig()
        self._sim: Simulation | None = None

        # Initialize shared buffers FIRST (before super().__init__)
        # because PufferLib may access them during initialization

        policy_env_info = PolicyEnvInterface.from_mg_cfg(cfg)

        self._buffers: Buffers = Buffers(
            observations=np.zeros(
                (policy_env_info.num_agents, *policy_env_info.observation_space.shape),
                dtype=dtype_observations,
            ),
            terminals=np.zeros(policy_env_info.num_agents, dtype=dtype_terminals),
            truncations=np.zeros(policy_env_info.num_agents, dtype=dtype_truncations),
            rewards=np.zeros(policy_env_info.num_agents, dtype=dtype_rewards),
            masks=np.ones(policy_env_info.num_agents, dtype=dtype_masks),
            actions=np.zeros(policy_env_info.num_agents, dtype=dtype_actions),
            teacher_actions=np.zeros(policy_env_info.num_agents, dtype=dtype_actions),
        )

        # Set observation and action spaces BEFORE calling super().__init__()
        # PufferLib requires these to be set first
        self.single_observation_space: Box = policy_env_info.observation_space
        self.single_action_space: Discrete = policy_env_info.action_space

        self._env_supervisor: MultiAgentPolicy | None = None
        self._new_sim()
        sim = cast(Simulation, self._sim)
        self.num_agents = sim.num_agents

        super().__init__(buf=buf)

    @property
    def env_cfg(self) -> MettaGridConfig:
        """Get the environment configuration."""
        return self._current_cfg

    def set_mg_config(self, config: MettaGridConfig) -> None:
        self._current_cfg = config

    def get_episode_rewards(self) -> np.ndarray:
        return cast(Simulation, self._sim).episode_rewards

    @property
    def current_simulation(self) -> Simulation:
        return cast(Simulation, self._sim)

    def _new_sim(self) -> None:
        if self._sim is not None:
            self._sim.close()

        self._sim = self._simulator.new_simulation(self._current_cfg, self._current_seed, buffers=self._buffers)

        if self._env_supervisor_cfg.policy is not None:
            policy_spec = self._resolve_policy_spec()
            self._env_supervisor = initialize_or_load_policy(
                PolicyEnvInterface.from_mg_cfg(self._current_cfg),
                policy_spec,
            )

            self._compute_supervisor_actions()

    def _resolve_policy_spec(self) -> PolicySpec:
        """Resolve a policy spec from the supervisor config.

        Handles both URIs (s3://, file://) and class paths.
        Also handles submission archives (.zip files) by extracting them to a persistent cache.
        """
        policy = self._env_supervisor_cfg.policy
        assert policy is not None  # Caller should check this

        # Check if policy is a URI by checking for URI schemes
        # ParsedURI.parse treats everything as a file path, so we need to check explicitly
        uri_schemes = ("s3://", "file://", "mock://", "wandb://", "gdrive://", "https://", "http://")
        is_uri = any(policy.startswith(scheme) for scheme in uri_schemes)

        if is_uri:
            # Check if this is a submission archive (ends with .zip or contains .zip)
            # Handle cases like: s3://bucket/path/file.zip or s3://bucket/path/file.zip/
            policy_lower = policy.lower()
            if ".zip" in policy_lower and (
                policy_lower.endswith(".zip")
                or policy_lower.endswith(".zip/")
                or "/" in policy_lower
                and policy_lower.split("/")[-1].endswith(".zip")
            ):
                # This is a submission archive, extract it to a persistent cache location
                return self._resolve_submission_archive(policy)

            # This is a URI pointing to a checkpoint file, use CheckpointManager
            try:
                from metta.rl.checkpoint_manager import CheckpointManager
            except ImportError:
                raise ImportError(
                    f"Cannot load policy from URI {policy}: CheckpointManager not available. "
                    "Install metta package to use URI-based policies."
                ) from None

            return CheckpointManager.policy_spec_from_uri(policy, device="cpu")

        # Not a URI, use as class path
        return PolicySpec(
            class_path=policy,
            data_path=self._env_supervisor_cfg.policy_data_path,
        )

    def _resolve_submission_archive(self, s3_path: str) -> PolicySpec:
        """Extract a submission archive and return a PolicySpec.

        Extracts to a persistent cache directory so the extracted files remain available
        for the lifetime of the environment.
        """
        try:
            import json
            import zipfile

            from metta.common.util.file import local_copy
        except ImportError:
            raise ImportError(
                f"Cannot load policy from submission archive {s3_path}: "
                "Required modules not available. Install metta package."
            ) from None

        # Create a cache directory based on the URI hash to avoid re-extracting
        cache_dir = Path.home() / ".mettagrid" / "submission_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Use hash of URI as cache key
        uri_hash = hashlib.sha256(s3_path.encode()).hexdigest()[:16]
        extraction_root = cache_dir / uri_hash

        # Extract if not already cached
        if not extraction_root.exists() or not (extraction_root / "policy_spec.json").exists():
            extraction_root.mkdir(parents=True, exist_ok=True)
            with local_copy(s3_path) as local_archive:
                # Extract zip file
                with zipfile.ZipFile(local_archive, "r") as archive:
                    archive.extractall(extraction_root)

        # Load policy_spec.json
        policy_spec_path = extraction_root / "policy_spec.json"
        if not policy_spec_path.exists():
            raise FileNotFoundError(f"policy_spec.json not found in extracted submission: {extraction_root}")

        with policy_spec_path.open() as f:
            raw_spec = json.load(f)

        spec = PolicySpec.model_validate(raw_spec)

        # Resolve data_path relative to extraction root
        if spec.data_path:
            candidate = Path(spec.data_path)
            if candidate.is_absolute():
                if not candidate.exists():
                    raise FileNotFoundError(f"Policy data path does not exist: {candidate}")
                spec.data_path = str(candidate)
            else:
                resolved = extraction_root / candidate
                if not resolved.exists():
                    raise FileNotFoundError(
                        f"Policy data path '{spec.data_path}' not found in submission directory {extraction_root}"
                    )
                spec.data_path = str(resolved)

        # Check if this submission contains Nim code and ensure it's compiled
        # Look for common indicators: agents.py, *.nim files, nimby.lock
        has_nim_code = False
        nim_dir = None
        agents_py = None

        # Check for agents.py file (common in submissions with Nim code)
        # It might be in the root or in a subdirectory like dinky/
        agents_py_candidates = [extraction_root / "agents.py"]
        # Also check common subdirectories
        for subdir in ["dinky", "nim_agents", "agents"]:
            agents_py_candidates.append(extraction_root / subdir / "agents.py")

        for candidate_agents_py in agents_py_candidates:
            if candidate_agents_py.exists():
                agents_py = candidate_agents_py
                # Check if there's a corresponding Nim directory
                # Could be in the same directory as agents.py or a parent directory
                candidate_dirs = [
                    agents_py.parent,
                    extraction_root,
                    extraction_root / "nim_agents",
                    extraction_root / "dinky",
                ]
                for candidate_dir in candidate_dirs:
                    if (candidate_dir / "nimby.lock").exists() or list(candidate_dir.glob("*.nim")):
                        has_nim_code = True
                        nim_dir = candidate_dir
                        break
                if has_nim_code:
                    break

        # Also check for Nim files directly
        if not has_nim_code:
            nim_files = list(extraction_root.rglob("*.nim"))
            if nim_files:
                has_nim_code = True
                # Use the directory containing the first Nim file
                nim_dir = nim_files[0].parent
                # Also look for agents.py in the same directory
                if agents_py is None:
                    agents_py = nim_dir / "agents.py"
                    if not agents_py.exists():
                        agents_py = None

        # Patch agents.py FIRST (before compilation) to prevent it from compiling
        # This must happen before any imports
        if has_nim_code and nim_dir and agents_py and agents_py.exists():
            try:
                _patch_agents_py_to_skip_compilation(agents_py, nim_dir)
            except Exception as e:
                logger.debug(f"Failed to patch agents.py (non-fatal): {e}")

        # Ensure Nim code is compiled if present
        if has_nim_code and nim_dir:
            try:
                from mettagrid.policy.nim_build_cache import ensure_nim_compiled

                ensure_nim_compiled(nim_dir)
            except ImportError:
                # nim_build_cache not available, skip caching
                logger.warning("nim_build_cache not available, Nim compilation caching disabled")
            except Exception as e:
                logger.warning(f"Failed to ensure Nim compilation: {e}, will attempt on import")

        # Add extraction root to sys.path so custom policy code can be imported
        # This needs to persist for the lifetime of the environment
        sys_path_entry = str(extraction_root.resolve())
        if sys_path_entry not in sys.path:
            sys.path.insert(0, sys_path_entry)

        return spec

    @override
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self._current_seed = seed

        self._new_sim()

        return self._buffers.observations, {}

    @override
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        sim = cast(Simulation, self._sim)

        if sim._c_sim.terminals().all() or sim._c_sim.truncations().all():
            self._new_sim()
            sim = cast(Simulation, self._sim)

        # Gymnasium returns int64 arrays by default when sampling MultiDiscrete spaces,
        # so coerce here to keep callers simple while preserving strict bounds checking.
        actions_to_copy = actions if actions.dtype == dtype_actions else np.asarray(actions, dtype=dtype_actions)
        np.copyto(self._buffers.actions, actions_to_copy, casting="safe")

        sim.step()

        # Do this after step() so that the trainer can use it if needed
        if self._env_supervisor_cfg.policy is not None:
            self._compute_supervisor_actions()

        return (
            self._buffers.observations,
            self._buffers.rewards,
            self._buffers.terminals,
            self._buffers.truncations,
            sim._context.get("infos", {}),
        )

    def _compute_supervisor_actions(self) -> None:
        if self._env_supervisor is None:
            return

        teacher_actions = self._buffers.teacher_actions
        raw_observations = self._buffers.observations
        self._env_supervisor.step_batch(raw_observations, teacher_actions)

    @property
    def observations(self) -> np.ndarray:
        return self._buffers.observations

    @observations.setter
    def observations(self, observations: np.ndarray) -> None:
        self._buffers.observations = observations

    @property
    def rewards(self) -> np.ndarray:
        return self._buffers.rewards

    @rewards.setter
    def rewards(self, rewards: np.ndarray) -> None:
        self._buffers.rewards = rewards

    @property
    def terminals(self) -> np.ndarray:
        return self._buffers.terminals

    @terminals.setter
    def terminals(self, terminals: np.ndarray) -> None:
        self._buffers.terminals = terminals

    @property
    def truncations(self) -> np.ndarray:
        return self._buffers.truncations

    @truncations.setter
    def truncations(self, truncations: np.ndarray) -> None:
        self._buffers.truncations = truncations

    @property
    def masks(self) -> np.ndarray:
        return self._buffers.masks

    @masks.setter
    def masks(self, masks: np.ndarray) -> None:
        self._buffers.masks = masks

    @property
    def actions(self) -> np.ndarray:
        return self._buffers.actions

    @actions.setter
    def actions(self, actions: np.ndarray) -> None:
        self._buffers.actions = actions

    @property
    def teacher_actions(self) -> np.ndarray:
        return self._buffers.teacher_actions

    @teacher_actions.setter
    def teacher_actions(self, teacher_actions: np.ndarray) -> None:
        self._buffers.teacher_actions = teacher_actions

    @property
    def render_mode(self) -> str:
        """PufferLib render mode - returns 'ansi' for text-based rendering."""
        return "ansi"

    def render(self) -> str:
        """Render the current state as unicode text."""
        from mettagrid.renderer.miniscope.buffer import MapBuffer
        from mettagrid.renderer.miniscope.symbol import DEFAULT_SYMBOL_MAP

        sim = cast(Simulation, self._sim)

        symbol_map = DEFAULT_SYMBOL_MAP.copy()
        for obj in self._current_cfg.game.objects.values():
            if obj.render_name:
                symbol_map[obj.render_name] = obj.render_symbol
            symbol_map[obj.name] = obj.render_symbol

        return MapBuffer(
            symbol_map=symbol_map,
            initial_height=sim.map_height,
            initial_width=sim.map_width,
        ).render_full_map(sim._c_sim.grid_objects())

    def close(self) -> None:
        """Close the environment."""
        if self._sim is None:
            return

        self._sim.close()
        self._sim = None
