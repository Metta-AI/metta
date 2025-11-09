"""
Doxascope Data Module

This module provides functionality for:
1. Logging LSTM memory vectors and agent positions during simulation using the DoxascopeLogger class.
2. Preprocessing the logged data for neural network training.

"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


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
    return sorted(positions)


def get_num_classes_for_manhattan_distance(d: int) -> int:
    """Returns the number of reachable cells within a given Manhattan distance."""
    d = abs(d)
    return 2 * d * d + 2 * d + 1


def get_pos_to_class_id_map(d: int) -> Dict[Tuple[int, int], int]:
    """Returns a mapping from (dr, dc) -> class_id for a given Manhattan distance."""
    positions = get_positions_for_manhattan_distance(d)
    return {pos: i for i, pos in enumerate(positions)}


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


def get_num_classes_for_quadrant_granularity(k: int) -> int:
    """Returns the number of classes for quadrant-based prediction."""
    return 5 if abs(k) == 1 else 9


def pos_to_quadrant_class_id(dr: int, dc: int) -> int:
    """Converts a relative position (dr, dc) to a quadrant class ID."""
    if dr == 0 and dc == 0:
        return 0  # Still
    if dr < 0 and dc == 0:
        return 1  # North
    if dr > 0 and dc == 0:
        return 2  # South
    if dr == 0 and dc < 0:
        return 3  # West
    if dr == 0 and dc > 0:
        return 4  # East
    if dr < 0 and dc < 0:
        return 5  # NW
    if dr < 0 and dc > 0:
        return 6  # NE
    if dr > 0 and dc < 0:
        return 7  # SW
    if dr > 0 and dc > 0:
        return 8  # SE
    raise ValueError(f"Invalid relative position for quadrant classification: ({dr}, {dc})")


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
        self.data: List[Dict] = []
        self.timestep = 0
        self.agent_id_map: Optional[Dict[int, int]] = None
        self.agent_type_id: int = 0
        self.output_file: Optional[Path] = None

    def configure(
        self,
        policy_uri: str,
        object_type_names: Optional[List[str]] = None,
    ):
        """Configure the logger with policy-specific information."""
        if not self.enabled:
            return

        stem = Path(policy_uri).stem
        if ":" in stem:
            base_name, version = stem.split(":", 1)
        else:
            base_name = stem
            version = None

        base_name = base_name.replace("/", "_")

        if version:
            self.output_dir = self.base_dir / base_name / version
        else:
            self.output_dir = self.base_dir / base_name

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = self.output_dir / f"doxascope_data_{self.simulation_id}.json"

        if object_type_names and "agent" in object_type_names:
            self.agent_type_id = object_type_names.index("agent")
        else:
            logger.warning(f"Could not find 'agent' in object_type_names {object_type_names}, defaulting to type ID 0.")

        logger.info("Doxascope logging enabled.")

    def _build_agent_id_map(self, env_grid_objects: Dict) -> Dict[int, int]:
        """Builds a mapping from agent IDs to grid object IDs."""
        agent_id_map = {}
        agent_obj_ids = []
        type_counts = {}
        for obj_id, obj in env_grid_objects.items():
            obj_type = obj.get("type_id")
            type_counts[obj_type] = type_counts.get(obj_type, 0) + 1
            if obj_type == self.agent_type_id:
                agent_obj_ids.append(obj_id)

        agent_obj_ids.sort()

        for agent_id, obj_id in enumerate(agent_obj_ids):
            agent_id_map[agent_id] = obj_id

        logger.debug(f"Type counts in grid objects: {type_counts}")
        logger.debug(
            f"Looking for agent_type_id={self.agent_type_id}, found "
            f"{len(agent_obj_ids)} agent-type objects, mapped {len(agent_id_map)} with sequential agent_ids"
        )
        return agent_id_map

    def _get_last_lstm_layer(
        self, h_tensor: torch.Tensor, c_tensor: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Extracts the last layer from LSTM state tensors, handling (L,B,H) and (B,L,H) formats.
        Heuristic to distinguish tensor formats: smaller dim is layers.
        - (L, B, H): layers first, batch second - typical PyTorch default
        - (B, L, H): batch first, layers second - when batch_first=True
        """
        if h_tensor.ndim == 2:
            return h_tensor, c_tensor
        if h_tensor.ndim != 3:
            return None, None

        d0, d1, _ = h_tensor.shape

        if d0 < d1:
            return h_tensor[-1, :, :], c_tensor[-1, :, :]
        elif d1 < d0:
            return h_tensor[:, -1, :], c_tensor[:, -1, :]
        else:
            logger.warning(
                f"Ambiguous LSTM state shape {h_tensor.shape} encountered. "
                f"Assuming default (L, B, H) format. If this is incorrect, "
                f"the logged memory vector will be wrong."
            )
            return h_tensor[-1, :, :], c_tensor[-1, :, :]

    def _extract_memory_from_policy_state(self, policy: Any, policy_idxs: torch.Tensor) -> Optional[torch.Tensor]:
        """Extract LSTM memory from policy.state (preferred method)."""
        try:
            st = getattr(policy, "state", None)
            h_attr = getattr(st, "lstm_h", None) if st is not None else None
            c_attr = getattr(st, "lstm_c", None) if st is not None else None
            if h_attr is not None and c_attr is not None:
                last_h, last_c = self._get_last_lstm_layer(h_attr, c_attr)
                if last_h is not None and last_c is not None:
                    mm = torch.cat([last_h, last_c], dim=1)
                    try:
                        select_rows = policy_idxs.to(torch.long)
                        return mm.index_select(0, select_rows)
                    except Exception:
                        return mm
        except Exception:
            pass
        return None

    def _extract_memory_from_components(self, policy: Any, policy_idxs: torch.Tensor) -> Optional[torch.Tensor]:
        """Extract LSTM memory directly from policy components."""
        try:
            components_dict = getattr(policy, "components", None)
            if components_dict is None and hasattr(policy, "network"):
                components_dict = getattr(policy.network, "components", None)

            if components_dict is not None:
                for comp in components_dict.values():
                    h_buf = getattr(comp, "lstm_h", None)
                    c_buf = getattr(comp, "lstm_c", None)

                    # Handle LSTMReset case (tensors)
                    if isinstance(h_buf, torch.Tensor) and isinstance(c_buf, torch.Tensor):
                        last_h, last_c = self._get_last_lstm_layer(h_buf, c_buf)
                        if last_h is not None and last_c is not None:
                            memory_matrix = self._select_agent_memories(last_h, last_c, policy_idxs)
                            if memory_matrix is not None:
                                return memory_matrix

                    # Handle regular LSTM case (dictionaries)
                    elif isinstance(h_buf, dict) and isinstance(c_buf, dict) and h_buf and c_buf:
                        env_id = max(h_buf.keys())
                        if env_id in c_buf:
                            h_tensor = h_buf[env_id]
                            c_tensor = c_buf[env_id]
                            last_h, last_c = self._get_last_lstm_layer(h_tensor, c_tensor)
                            if last_h is not None and last_c is not None:
                                memory_matrix = self._select_agent_memories(last_h, last_c, policy_idxs)
                                if memory_matrix is not None:
                                    return memory_matrix
        except Exception:
            pass
        return None

    def _extract_memory_from_get_memory(self, policy: Any, policy_idxs: torch.Tensor) -> Optional[torch.Tensor]:
        """Extract LSTM memory using component get_memory() methods."""
        try:
            components_dict = getattr(policy, "components", None)
            if components_dict is None and hasattr(policy, "network"):
                components_dict = getattr(policy.network, "components", None)

            if components_dict is not None:
                for comp in components_dict.values():
                    get_mem = getattr(comp, "get_memory", None)
                    if callable(get_mem):
                        lstm_h_dict, lstm_c_dict = get_mem()
                        if isinstance(lstm_h_dict, dict) and isinstance(lstm_c_dict, dict) and lstm_h_dict:
                            env_id = max(lstm_h_dict.keys())
                            h_t = lstm_h_dict[env_id]
                            c_t = lstm_c_dict[env_id]
                            if isinstance(h_t, torch.Tensor) and isinstance(c_t, torch.Tensor):
                                last_h, last_c = self._get_last_lstm_layer(h_t, c_t)
                                if last_h is not None and last_c is not None:
                                    memory_matrix = self._select_agent_memories(last_h, last_c, policy_idxs)
                                    if memory_matrix is not None:
                                        return memory_matrix
        except Exception:
            pass
        return None

    def _select_agent_memories(
        self, last_h: torch.Tensor, last_c: torch.Tensor, policy_idxs: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Select and concatenate memory vectors for the specified agent indices."""
        try:
            select_rows = policy_idxs.to(torch.long)
            max_row = min(last_h.shape[0], int(select_rows.max().item() + 1))
            clipped = torch.clamp(select_rows, 0, max_row - 1)
            if not torch.equal(clipped, select_rows):
                logger.warning(
                    f"Clamped out-of-bounds indices in LSTM extraction: original {select_rows}, clipped {clipped}"
                )
            return torch.cat([last_h.index_select(0, clipped), last_c.index_select(0, clipped)], dim=1)
        except (IndexError, ValueError) as e:
            logger.warning(f"Index selection failed: {e}, attempting fallback")
            try:
                # Validate dimensions before fallback
                if last_h.shape[0] == last_c.shape[0]:
                    return torch.cat([last_h, last_c], dim=1)
                else:
                    logger.error(
                        f"Cannot concatenate tensors with mismatched first dimensions: {last_h.shape} vs {last_c.shape}"
                    )
                    return None
            except Exception as e:
                logger.error(f"Fallback concatenation failed: {e}")
                return None

    def _extract_memory_cortex(self, policy: Any, policy_idxs: torch.Tensor, layer_n: int) -> Tensor:
        """
        General method to extract the memory tensors from the Cortex architecture.
        Will extract
        self._rollout_current_state (see agent/src/metta/agent/components/cortex.py) from the policy.
        """
        pass


    def log_timestep(
        self,
        policy: Any,
        policy_idxs: torch.Tensor,
        env_grid_objects: Dict,
    ):
        """Log memory vectors and positions for policy agents at current timestep.

        Note: Only supports single-environment logging. Multi-environment setups
        are not currently supported.

        Args:
            policy: The policy being evaluated
            policy_idxs: Indices of agents controlled by the policy
            env_grid_objects: Dictionary of grid objects from the environment
        """
        if not self.enabled:
            return

        self.timestep += 1

        memory_matrix: Optional[torch.Tensor] = None

        ###############################
        # TODO: Implement new general
        # cortex _extract_memory_general
        ###############################

        memory_matrix = _extract_memory_cortex(policy, policy_idxs)

        agent_map = self._build_agent_id_map(env_grid_objects)
        timestep_data: Dict[str, Any] = {"timestep": self.timestep, "agents": []}

        for i, agent_idx in enumerate(policy_idxs):
            flat_idx = int(agent_idx.item())
            row = i
            if row < 0 or row >= memory_matrix.shape[0]:
                continue
            memory_vector = memory_matrix[row].flatten().detach().cpu()
            mv_np = memory_vector.numpy().astype(np.float32)

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

        """
        def _detect_architecture(self, policy: Any) -> str:
            ""
            Detects the high-level config name (i.e. the name of the policy_architecture, such as ViTDefaultConfig
            or TransformerConfig. Then depending on that, attempts to go deeper to get an exact name for the config
            e.g. for GTrXL finds policy_architecture.transformer which returns GTrXLConfig.
            The high_level may be enough, but I wanted to get both to start with and can remove the exact later if needed.
            ""
            # You can actually just grab it from "transformerPolicy" you don't need to use the config here.
            high_level_config_name = type(policy.config).__name__

            if high_level_config_name == "TransformerPolicyConfig":
                exact_config_name = type(policy.config.transformer).__name__

            # More to be added as I check the exact structure of other architectures.

            return [high_level_config_name, exact_config_name]
        """
        """

        def _extract_memory_any_architecture(self, policy: Any, policy_idxs: torch.Tensor, layer_n: int) -> Optional[torch.Tensor]:
            high_level_config_name, exact_config_name = self._detect_architecture(policy)

            if high_level_config_name == "TransformerPolicyConfig": # Could be extracted into a helper function
                ""
                The TransformerConfig object stores memory in tensors defined as
                    self._memory_tensor: torch.Tensor | None = None  # [num_envs, layers, mem_len, hidden]
                (there is also self._memory which appears to be vestigial and unused.)
                The memory_tensor contains the current timestep, last layer at [:, num_layers, -1, :]
                ""
                try:
                    memory_tensor = getattr(policy, "_memory_tensor", None)
                    if memory_tensor is not None:
                        # Shape: [capacity, num_layers, memory_len, hidden_size]
                        if layer_n < 0:
                            layer_n = memory_tensor.size(1) + layer_n  # Support negative indexing other than -1

                        # Get agents specified by policy_idxs
                        select_rows = policy_idxs.to(torch.long)
                        agent_memories = memory_tensor.index_select(0, select_rows)

                        # Extract layer n, last memory timestep
                        layer_activations = agent_memories[:, layer_n, -1, :]  # [agents, hidden]
                except Exception:
                    logger.warning("No activations found in _memory_tensor on timestep %d. Disabling doxascope logging.", self.timestep)
                    return None
            else:
                logger.warning(f"Doxascope is not implemented for policies of configuration {high_level_config_name}. Disabling doxascope logging.")
                self.enabled = False
                return None

            return layer_activations
        """


        """
        def log_timestep_flexible(
            self,
            policy: Any,
            policy_idxs: torch.Tensor,
            env_grid_objects: Dict,
            layer_n: int = -1, # May want layer_n to be required, or default to -1.
        ):

            if not self.enabled:
                return

            self.timestep += 1

            memory_matrix = self._extract_memory_any_architecture(policy, policy_idxs, layer_n)
            if memory_matrix is None:
                return # error message already handled in _extract_memory_any_architecture

            agent_map = self._build_agent_id_map(env_grid_objects)
            timestep_data: Dict[str, Any] = {"timestep": self.timestep, "agents": []}

            for i, agent_idx in enumerate(policy_idxs):
                # Directly duplicated from previous logger.
                flat_idx = int(agent_idx.item())
                row = i
                if row < 0 or row >= memory_matrix.shape[0]:
                    continue

                # extracts memory vector
                # then flattens the memory vector for given agent
                # removes grad, and moves it to cpu, and changes it to numpy format
                memory_vector = memory_matrix[row].flatten().detach().cpu()
                mv_np = memory_vector.numpy().astype(np.float32)

                agent_id = flat_idx
                if agent_id not in agent_map:
                    logger.warning(f"Agent {agent_id} not found in grid objects")
                    continue

                # extracts agent's position on the grid
                grid_obj_id = agent_map[agent_id]
                grid_obj = env_grid_objects[grid_obj_id]  # type: ignore[index]
                position = (grid_obj["r"], grid_obj["c"])
                agent_id_record = {"agent_id": agent_id}

                # creates record vector
                record = {
                    **agent_id_record,
                    "memory_vector": mv_np.tolist(),
                    "position": position,
                }
                #appends record for agent #i to position i of timestep_data for this timestep
                # structure: timestep_data = [{..., memory_vector, position}, {..., memory_vector, position}] where timestep_data[n-1] contains the data for agent n
                timestep_data["agents"].append(record)

            self.data.append(timestep_data)
        """

    def save(self):
        """Save logged data to JSON file."""
        if not self.enabled or self.output_file is None:
            return

        try:
            if not self.data:
                logger.warning("Doxascope: no data was logged for this simulation; nothing to save.")
                return
            with open(self.output_file, "w") as f:
                json.dump(self.data, f)
            file_size_bytes = self.output_file.stat().st_size
            file_size_mb = file_size_bytes / (1024 * 1024)
            logger.info(f"Doxascope data saved to {self.output_file.name} ({file_size_mb:.2f} MB)")
        except Exception as e:
            logger.error(f"Failed to save doxascope data: {e}")


def _extract_agent_trajectories(files: list) -> List[List[Tuple[np.ndarray, Tuple[int, int]]]]:
    """
    Load raw data (single-env) and return per-file, per-agent trajectories.
    Each trajectory is a list of (memory_vector, position) tuples from a single
    file for a single agent.
    """
    trajectories: List[List[Tuple[np.ndarray, Tuple[int, int]]]] = []

    for json_file in files:
        with open(json_file, "r") as f:
            data = json.load(f)

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
    trajectories: List[List[Tuple[np.ndarray, Tuple[int, int]]]],
    num_future: int,
    num_past: int,
    granularity: str = "exact",
) -> Tuple[List[np.ndarray], List[List[int]]]:
    """Generate training samples (X, y) from per-agent trajectories.
    Args:
        trajectories: List of trajectories, each a list of (memory, position)
        num_future: Number of future timesteps to predict
        num_past: Number of past timesteps to predict
        granularity: The prediction granularity ("exact" or "quadrant")
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

            for k in list(range(-num_past, 0)) + list(range(1, num_future + 1)):
                pos_k = trajectory[i + k][1]
                dr, dc = pos_k[0] - current_pos[0], pos_k[1] - current_pos[1]
                max_dist = abs(k)

                if abs(dr) + abs(dc) > max_dist:
                    valid_sample = False
                    break

                if granularity == "exact":
                    label = pos_to_class_id(dr, dc, max_dist)
                elif granularity == "quadrant":
                    label = pos_to_quadrant_class_id(dr, dc)
                    # For k=1, agent can't move diagonally, so labels 5-8 are impossible.
                    if abs(k) == 1 and label > 4:
                        valid_sample = False
                        break
                else:
                    raise ValueError(f"Unknown granularity: {granularity}")

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
    granularity: str = "exact",
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Preprocesses raw doxascope JSON data to create training-ready NPZ files.
    """
    if not json_files:
        logger.warning("No JSON files provided for preprocessing.")
        return None, None

    logger.info(f"Processing {len(json_files)} simulation log(s) with '{granularity}' granularity...")

    trajectories = _extract_agent_trajectories(json_files)
    if not trajectories:
        logger.warning("No valid agent trajectories found in the provided files.")
        return None, None

    all_memory_vectors, all_labels = _create_training_samples(
        trajectories, num_future_timesteps, num_past_timesteps, granularity=granularity
    )

    if not all_memory_vectors:
        logger.warning("No training samples could be created from the trajectories.")
        return None, None

    try:
        X = np.array(all_memory_vectors, dtype=np.float32)
        y = np.array(all_labels, dtype=np.int64)

        output_file = preprocessed_dir / output_filename
        preprocessed_dir.mkdir(parents=True, exist_ok=True)
        # Save granularity along with the data
        np.savez_compressed(output_file, X=X, y=y, granularity=np.array(granularity))

        logger.info(f"Successfully saved {len(X)} samples to {output_file}")
        return X, y
    except ValueError as e:
        logger.error(f"Failed to create NumPy arrays due to inconsistent shapes: {e}")
        unique_dims = {mv.shape for mv in all_memory_vectors}
        logger.error(f"Found memory vectors with the following shapes: {unique_dims}")
        return None, None
