#!/usr/bin/env python3
"""
Doxascope Data Module

This module provides functionality for:
1. Logging LSTM memory vectors and agent positions during simulation using the DoxascopeLogger class.
2. Preprocessing the logged data for neural network training

"""

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tensordict import TensorDict

# from metta.agent.policy import Policy

logger = logging.getLogger(__name__)


class NoRecurrentStateError(Exception):
    """Raised when Doxascope tries to log data for a policy with no recurrent state."""

    pass


# Coordinate Conversion Utilities
@lru_cache(maxsize=128)
def get_positions_for_manhattan_distance(d: int) -> List[Tuple[int, int]]:
    """
    Returns a canonical, sorted list of all (dr, dc) positions
    within a given Manhattan distance.
    """
    d = abs(d)
    positions = []
    for dr in range(-d, d + 1):
        for dc in range(-d, d + 1):
            if abs(dr) + abs(dc) <= d:
                positions.append((dr, dc))
    # Sort by row, then column for a canonical order
    return sorted(positions)


@lru_cache(maxsize=128)
def get_num_classes_for_manhattan_distance(d: int) -> int:
    """Returns the number of reachable cells within a given Manhattan distance."""
    d = abs(d)
    return 2 * d * d + 2 * d + 1


@lru_cache(maxsize=128)
def get_pos_to_class_id_map(d: int) -> Dict[Tuple[int, int], int]:
    """Returns a mapping from (dr, dc) -> class_id for a given Manhattan distance."""
    positions = get_positions_for_manhattan_distance(d)
    return {pos: i for i, pos in enumerate(positions)}


@lru_cache(maxsize=128)
def get_class_id_to_pos_map(d: int) -> Dict[int, Tuple[int, int]]:
    """Returns a mapping from class_id -> (dr, dc) for a given Manhattan distance."""
    positions = get_positions_for_manhattan_distance(d)
    return {i: pos for i, pos in enumerate(positions)}


def pos_to_class_id(dr: int, dc: int, d: int) -> int:
    """Converts a relative position (dr, dc) to a class index for a given Manhattan distance."""
    mapping = get_pos_to_class_id_map(d)
    pos = (dr, dc)
    if pos not in mapping:
        raise ValueError(f"Position ({dr}, {dc}) is outside max Manhattan distance {d}")
    return mapping[pos]


def class_id_to_pos(class_id: int, d: int) -> Tuple[int, int]:
    """Converts a class index back to a relative position (dr, dc) for a given Manhattan distance."""
    mapping = get_class_id_to_pos_map(d)
    if class_id not in mapping:
        raise ValueError(f"Class ID {class_id} is out of bounds for Manhattan distance {d}")
    return mapping[class_id]


class DoxascopeLogger:
    """Logs memory vectors and position data for training doxascope networks."""

    def __init__(
        self,
        enabled: bool,
        simulation_id: str,
        output_dir: str = "./train_dir/doxascope/raw_data/",
    ):
        self.enabled = enabled
        if not self.enabled:
            return

        self.base_dir = Path(output_dir)
        self.simulation_id = simulation_id
        self.data: List = []
        self.timestep = 0
        self.agent_id_map: Optional[Dict[int, int]] = None
        self.agent_type_id: int = 0
        self.output_file: Optional[Path] = None
        self._warned_multi_env_mismatch = False
        self._last_td: Optional[TensorDict] = None

    def configure(
        self,
        policy_name: str,
        object_type_names: Optional[List[str]] = None,
    ):
        """Configure the logger with policy-specific information."""
        if not self.enabled:
            return

        self.output_dir = self.base_dir / policy_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = self.output_dir / f"doxascope_data_{self.simulation_id}.json"

        if object_type_names and "agent" in object_type_names:
            self.agent_type_id = object_type_names.index("agent")
        else:
            logger.warning("Could not find 'agent' in object_type_names, defaulting to type ID 0.")

        logger.info("Doxascope logging enabled.")

    def set_last_td(self, td: TensorDict) -> None:
        """Capture the most recent per-forward TensorDict to preserve agent batch ordering for logging."""
        if not self.enabled:
            return
        self._last_td = td

    def _build_agent_id_map(self, env_grid_objects: Dict) -> Dict[int, int]:
        """Builds a mapping from agent IDs to grid object IDs."""
        agent_id_map = {}
        for obj_id, obj in env_grid_objects.items():
            if obj.get("type") == self.agent_type_id:
                agent_id = obj.get("agent_id")
                if agent_id is not None:
                    agent_id_map[agent_id] = obj_id
        return agent_id_map

    def log_timestep(
        self,
        policy: Any,
        policy_idxs: torch.Tensor,
        env_grid_objects: Dict,
    ):
        """Log memory vectors and positions for policy agents at current timestep.

        Supports both single-env (env_grid_objects: Dict) and multi-env
        (env_grid_objects: List[Dict]) cases. For multi-env collection,
        provide agents_per_env to resolve (env_index, local_agent_id).
        """
        if not self.enabled:
            return

        self.timestep += 1

        # The policy passed in may be a wrapper. The actual model is inside.
        # Keep compatibility with wrapped policies if needed in future
        _ = getattr(policy, "policy", policy)

        # Try to extract recurrent state from multiple possible sources
        memory_matrix: Optional[torch.Tensor] = None
        batch_pos_to_global: Optional[Dict[int, int]] = None

        # 1) Preferred: from per-forward TensorDict (direct keys or nested)
        if isinstance(self._last_td, TensorDict):
            # Direct keys
            td_lstm_h = self._last_td.get("lstm_h") if "lstm_h" in self._last_td.keys() else None
            td_lstm_c = self._last_td.get("lstm_c") if "lstm_c" in self._last_td.keys() else None
            td_batch_idxs = self._last_td.get("__agent_indices") if "__agent_indices" in self._last_td.keys() else None

            # Or nested recurrent state
            recurrent_source: Any = None
            if "recurrent_state" in self._last_td.keys():
                recurrent_source = self._last_td.get("recurrent_state")

            lstm_h = (
                td_lstm_h if td_lstm_h is not None else (recurrent_source.get("lstm_h") if recurrent_source else None)
            )
            lstm_c = (
                td_lstm_c if td_lstm_c is not None else (recurrent_source.get("lstm_c") if recurrent_source else None)
            )
            batch_indices = (
                td_batch_idxs
                if td_batch_idxs is not None
                else (
                    recurrent_source.get("__agent_indices")
                    if recurrent_source and "__agent_indices" in recurrent_source.keys()
                    else None
                )
            )

            if (
                lstm_h is not None
                and lstm_c is not None
                and isinstance(lstm_h, torch.Tensor)
                and isinstance(lstm_c, torch.Tensor)
            ):
                if lstm_h.ndim == 3:
                    d0, d1, _ = lstm_h.shape
                    # Treat as [B, L, H] if middle dim small
                    if d1 <= 8 and d0 >= d1:
                        last_h = lstm_h[:, -1, :]
                        last_c = lstm_c[:, -1, :]
                        memory_matrix = torch.cat([last_h, last_c], dim=1)
                    # Treat as [L, B, H] if first dim small
                    elif d0 <= 8 and d1 >= d0:
                        last_h = lstm_h[-1, :, :]
                        last_c = lstm_c[-1, :, :]
                        memory_matrix = torch.cat([last_h, last_c], dim=1)
                elif lstm_h.ndim == 2 and lstm_c.ndim == 2 and lstm_h.shape[0] == lstm_c.shape[0]:
                    # Already [B, H]
                    memory_matrix = torch.cat([lstm_h, lstm_c], dim=1)

                if memory_matrix is not None and batch_indices is not None:
                    try:
                        batch_indices = batch_indices.reshape(-1).to(torch.long)
                        batch_pos_to_global = {int(i): int(g) for i, g in enumerate(batch_indices.tolist())}
                    except Exception:
                        batch_pos_to_global = None

        # 2) Fallback: from policy.state (supports LxBxH or LxAxH or AxH)
        if memory_matrix is None:
            st = getattr(policy, "state", None)
            h_attr = getattr(st, "lstm_h", None) if st is not None else None
            c_attr = getattr(st, "lstm_c", None) if st is not None else None
            if h_attr is not None and c_attr is not None:
                try:
                    if h_attr.ndim == 3:
                        # Assume [L, A, H] or [L, B, H] → take last layer
                        last_h = h_attr[-1, :, :]
                        last_c = c_attr[-1, :, :]
                    elif h_attr.ndim == 2:
                        # Already [A, H]
                        last_h = h_attr
                        last_c = c_attr
                    else:
                        last_h = None
                        last_c = None

                    if last_h is not None and last_c is not None:
                        mm = torch.cat([last_h, last_c], dim=1)  # [A, 2H]
                        # Reorder rows to match this step's policy agent ordering
                        try:
                            select_rows = policy_idxs.to(torch.long)
                            memory_matrix = mm.index_select(0, select_rows)
                        except Exception:
                            memory_matrix = mm
                        batch_pos_to_global = None
                except Exception:
                    pass

        # 3) Fallback: inspect policy components (handles ViT/LSTM and LSTMReset)
        if memory_matrix is None:
            try:
                # PolicyAutoBuilder exposes components on either `components` or `network.components`
                components_dict = getattr(policy, "components", None)
                if components_dict is None and hasattr(policy, "network"):
                    components_dict = getattr(policy.network, "components", None)

                if components_dict is not None:
                    # Try LSTMReset first (buffers `lstm_h`/`lstm_c` with shape [L, A, H])
                    for comp in components_dict.values():
                        h_buf = getattr(comp, "lstm_h", None)
                        c_buf = getattr(comp, "lstm_c", None)
                        if isinstance(h_buf, torch.Tensor) and isinstance(c_buf, torch.Tensor) and h_buf.ndim == 3:
                            # Take the last layer and map current agent batch rows
                            last_h = h_buf[-1, :, :]
                            last_c = c_buf[-1, :, :]
                            # Map to current policy batch ordering when possible
                            try:
                                select_rows = policy_idxs.to(torch.long)
                                # Guard against selecting beyond available rows
                                max_row = min(last_h.shape[0], int(select_rows.max().item() + 1))
                                clipped = torch.clamp(select_rows, 0, max_row - 1)
                                memory_matrix = torch.cat(
                                    [last_h.index_select(0, clipped), last_c.index_select(0, clipped)], dim=1
                                )
                            except Exception:
                                memory_matrix = torch.cat([last_h, last_c], dim=1)
                            batch_pos_to_global = None
                            break

                    # If not found, try LSTM component with dict memory
                    if memory_matrix is None:
                        for comp in components_dict.values():
                            get_mem = getattr(comp, "get_memory", None)
                            if callable(get_mem):
                                try:
                                    lstm_h_dict, lstm_c_dict = get_mem()
                                    if isinstance(lstm_h_dict, dict) and isinstance(lstm_c_dict, dict) and lstm_h_dict:
                                        # Use the first available key
                                        any_key = sorted(lstm_h_dict.keys())[0]
                                        h_t = lstm_h_dict[any_key]
                                        c_t = lstm_c_dict[any_key]
                                        if isinstance(h_t, torch.Tensor) and h_t.ndim == 3:
                                            last_h = h_t[-1, :, :]
                                            last_c = c_t[-1, :, :]
                                            try:
                                                select_rows = policy_idxs.to(torch.long)
                                                max_row = min(last_h.shape[0], int(select_rows.max().item() + 1))
                                                clipped = torch.clamp(select_rows, 0, max_row - 1)
                                                memory_matrix = torch.cat(
                                                    [last_h.index_select(0, clipped), last_c.index_select(0, clipped)],
                                                    dim=1,
                                                )
                                            except Exception:
                                                memory_matrix = torch.cat([last_h, last_c], dim=1)
                                            batch_pos_to_global = None
                                            break
                                except Exception:
                                    continue
            except Exception:
                pass

        # If we still don't have memory, disable after first step with a clear message
        if self.timestep == 2 and memory_matrix is None:
            logger.error(
                "Doxascope logging disabled: could not locate recurrent state (e.g., LSTM) "
                "in TensorDict or policy.state. Skipping data collection for this simulation."
            )
            self.enabled = False
            return

        # On the first logging step, check if we have memory vectors. If not, fail.
        if self.timestep == 2 and memory_matrix is None:
            logger.error(
                "Doxascope logging disabled: policy does not expose recurrent state (e.g., LSTM). "
                "Skipping data collection for this simulation."
            )
            self.enabled = False
            return

        # If memory vectors are still None after the first step, just skip logging
        if memory_matrix is None:
            return

        # Single environment only
        agent_map = self._build_agent_id_map(env_grid_objects)

        timestep_data = {"timestep": self.timestep, "agents": []}

        for i, agent_idx in enumerate(policy_idxs):
            flat_idx = int(agent_idx.item())
            # Determine batch row for this global index
            if "batch_pos_to_global" in locals() and batch_pos_to_global:
                # find batch position whose global id equals flat_idx
                try:
                    # Reverse map once lazily
                    if "global_to_batch" not in locals():
                        global_to_batch = {g: b for b, g in batch_pos_to_global.items()}
                    row = global_to_batch.get(flat_idx, None)
                except Exception:
                    row = i
            else:
                row = i
            if row is None or row < 0 or row >= memory_matrix.shape[0]:
                continue
            memory_vector = memory_matrix[row].flatten().detach().cpu()
            mv_np = memory_vector.numpy().astype(np.float32)

            # Single env; agent_idx corresponds directly to agent_id
            agent_id = flat_idx
            if agent_id not in agent_map:
                logger.warning(f"Agent {agent_id} not found in grid objects")
                continue
            grid_obj_id = agent_map[agent_id]
            grid_obj = env_grid_objects[grid_obj_id]  # type: ignore[index]
            position = (grid_obj["r"], grid_obj["c"])
            agent_id_record = {"agent_id": agent_id}

            record = {
                **agent_id_record,
                "memory_vector": mv_np.tolist(),
                "position": position,
            }
            timestep_data["agents"].append(record)

        self.data.append(timestep_data)

    def save(self):
        """Save logged data to JSON file."""
        if not self.enabled or self.output_file is None:
            return

        try:
            if not self.data:
                logger.warning("Doxascope: no data was logged for this simulation; nothing to save.")
                return
            # Write plain JSON
            with open(self.output_file, "w") as f:
                json.dump(self.data, f)
            file_size_bytes = self.output_file.stat().st_size
            file_size_mb = file_size_bytes / (1024 * 1024)
            logger.info(f"Doxascope data saved to {self.output_file.name} ({file_size_mb:.2f} MB)")
        except Exception as e:
            logger.error(f"Failed to save doxascope data: {e}")


def _extract_agent_trajectories(files: list) -> List[List[Tuple[np.ndarray, Tuple[int, int]]]]:
    """Load raw data (single-env) and return per-file, per-agent trajectories.

    Each trajectory is a list of (memory_vector, position) tuples from a single
    file for a single agent. Assumes single environment collection. Older logs
    that include an 'env' field are treated as single-env by ignoring that field.
    """
    trajectories: List[List[Tuple[np.ndarray, Tuple[int, int]]]] = []

    for json_file in files:
        with open(json_file, "r") as f:
            data = json.load(f)

        # Key by agent_id (single-env assumption)
        file_agent_trajs: Dict[int, List[Tuple[np.ndarray, Tuple[int, int]]]] = {}
        expected_dim = None

        for timestep_data in data:
            for agent_data in timestep_data["agents"]:
                agent_id = agent_data.get("agent_id")
                if agent_id is None:
                    continue

                memory = np.array(agent_data["memory_vector"], dtype=np.float32)

                if expected_dim is None:
                    expected_dim = memory.shape[0]
                elif memory.shape[0] != expected_dim:
                    logger.warning(
                        f"Skipping memory vector for agent {agent_id} in {json_file} with dimension "
                        f"{memory.shape[0]} (expected {expected_dim})"
                    )
                    continue

                position = tuple(agent_data["position"])
                key = int(agent_id)
                file_agent_trajs.setdefault(key, []).append((memory, position))

        # Append per-agent trajectories for this file
        for traj in file_agent_trajs.values():
            trajectories.append(traj)

    return trajectories


def _create_training_samples(
    trajectories: List[List[Tuple[np.ndarray, Tuple[int, int]]]], num_future: int, num_past: int
) -> Tuple[List[np.ndarray], List[List[int]]]:
    """Generate training samples (X, y) from per-agent trajectories.

    Args:
        trajectories: List of trajectories, each a list of (memory, position)
        num_future: Number of future timesteps to predict
        num_past: Number of past timesteps to predict
    """
    all_memory_vectors: List[np.ndarray] = []
    all_labels: List[List[int]] = []

    for trajectory in trajectories:
        if len(trajectory) < num_future + num_past + 1:
            continue

        for i in range(num_past, len(trajectory) - num_future):
            current_memory, current_pos = trajectory[i]
            timestep_labels: List[int] = []
            valid_sample = True

            # Create labels for past and future timesteps (Manhattan metric, no diagonals allowed)
            for k in list(range(-num_past, 0)) + list(range(1, num_future + 1)):
                pos_k = trajectory[i + k][1]
                dr, dc = pos_k[0] - current_pos[0], pos_k[1] - current_pos[1]
                max_dist = abs(k)

                if abs(dr) + abs(dc) > max_dist:
                    valid_sample = False
                    break

                label = pos_to_class_id(dr, dc, max_dist)
                timestep_labels.append(label)

            if valid_sample:
                all_labels.append(timestep_labels)
                all_memory_vectors.append(current_memory)

    return all_memory_vectors, all_labels


def preprocess_doxascope_data(
    json_files: list,
    preprocessed_dir: Path,
    output_filename: str = "training_data.npz",
    num_future_timesteps: int = 1,
    num_past_timesteps: int = 0,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Preprocesses raw doxascope JSON data to create training-ready NPZ files.
    """
    if not json_files:
        logger.warning("No JSON files provided for preprocessing.")
        return None, None

    logger.info(f"Processing {len(json_files)} simulation log(s)...")

    trajectories = _extract_agent_trajectories(json_files)
    if not trajectories:
        logger.warning("No valid agent trajectories found in the provided files.")
        return None, None

    all_memory_vectors, all_labels = _create_training_samples(trajectories, num_future_timesteps, num_past_timesteps)

    if not all_memory_vectors:
        logger.warning("No training samples could be created from the trajectories.")
        return None, None

    try:
        X = np.array(all_memory_vectors, dtype=np.float32)
        y = np.array(all_labels, dtype=np.int64)

        output_file = preprocessed_dir / output_filename
        preprocessed_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_file, X=X, y=y)

        logger.info(f"Successfully saved {len(X)} samples to {output_file}")
        return X, y
    except ValueError as e:
        logger.error(f"Failed to create NumPy arrays due to inconsistent shapes: {e}")
        unique_dims = {mv.shape for mv in all_memory_vectors}
        logger.error(f"Found memory vectors with the following shapes: {unique_dims}")
        return None, None
