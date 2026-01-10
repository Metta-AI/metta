"""
Doxascope Data Module

This module provides functionality for:
1. Logging memory vectors and agent positions during simulation using the DoxascopeLogger class.
2. Preprocessing the logged data for neural network training.

"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from mettagrid.simulator.interface import SimulatorEventHandler

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
        self.resource_names: Optional[List[str]] = None

    def clone(self, simulation_id: str) -> "DoxascopeLogger":
        """Create a fresh logger instance with the same configuration but new simulation ID."""
        new_logger = DoxascopeLogger(
            enabled=self.enabled,
            simulation_id=simulation_id,
            output_dir=str(self.base_dir),
        )
        # Copy configured attributes
        if hasattr(self, "output_dir"):
            new_logger.output_dir = self.output_dir
            new_logger.output_file = new_logger.output_dir / f"doxascope_data_{simulation_id}.json"

        if hasattr(self, "object_type_names"):
            new_logger.object_type_names = self.object_type_names

        if self.resource_names is not None:
            new_logger.resource_names = self.resource_names

        return new_logger

    def configure(
        self,
        policy_uri: str,
        object_type_names: Optional[List[str]] = None,
        resource_names: Optional[List[str]] = None,
    ):
        """Configure the logger with policy-specific information."""
        if not self.enabled:
            return

        self.resource_names = resource_names

        # Strip URI prefixes if present, since we need a path not a URI
        if policy_uri.startswith("file://"):
            policy_uri = policy_uri[7:]  # Remove "file://"
        elif policy_uri.startswith("s3://"):
            policy_uri = policy_uri[5:]  # Remove "s3://"
        elif policy_uri.startswith("metta://policy/"):
            policy_uri = policy_uri[15:]  # Remove "metta://policy/"
        elif policy_uri.startswith("metta://"):
            policy_uri = policy_uri[8:]  # Remove "metta://"

        stem = Path(policy_uri).stem
        if ":" in stem:
            base_name = stem.split(":", 1)[0]
            version = stem.split(":", 1)[1]
        elif ":" in policy_uri:
            parts = policy_uri.split(":")
            if len(parts) > 1:
                version = parts[-1]
                path_parts = parts[-2].split("/")
                base_name = path_parts[-1] if path_parts else "unknown_policy"
            else:
                base_name = stem
                version = None
        else:
            base_name = stem
            version = None

        if version:
            self.output_dir = self.base_dir / base_name / version
        else:
            self.output_dir = self.base_dir / base_name

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = self.output_dir / f"doxascope_data_{self.simulation_id}.json"

    def _build_agent_id_map(self, env_grid_objects: Dict) -> Dict[int, int]:
        """Builds a mapping from agent IDs to grid object IDs.

        Note: grid_objects() returns objects with "type_name" not "type_id".
        We match against type_name starting with "agent".
        """
        agent_id_map = {}
        agent_obj_ids = []
        type_counts = {}

        # Get the agent type name from object_type_names if available
        agent_type_name = None
        if hasattr(self, "object_type_names") and self.object_type_names:
            if self.agent_type_id < len(self.object_type_names):
                agent_type_name = self.object_type_names[self.agent_type_id]

        # If we don't have object_type_names, try common agent type names
        if agent_type_name is None:
            agent_type_name = "agent"

        for obj_id, obj in env_grid_objects.items():
            # grid_objects() returns "type_name" not "type_id"
            obj_type_name = obj.get("type_name")
            type_counts[obj_type_name] = type_counts.get(obj_type_name, 0) + 1

            # Match objects whose type_name starts with "agent" (handles "agent", "agent.agent", etc.)
            if obj_type_name and obj_type_name.startswith("agent"):
                agent_obj_ids.append(obj_id)

        agent_obj_ids.sort()

        for agent_id, obj_id in enumerate(agent_obj_ids):
            agent_id_map[agent_id] = obj_id

        logger.debug(f"Type counts in grid objects: {type_counts}")
        logger.debug(
            f"Looking for agent type_name starting with 'agent', found "
            f"{len(agent_obj_ids)} agent-type objects, mapped {len(agent_id_map)} with sequential agent_ids"
        )
        return agent_id_map

    def log_timestep(
        self,
        policies: list,
        env_grid_objects: Dict,
        object_type_names: Optional[List[str]] = None,
    ):
        """Log memory vectors and positions for policy agents at current timestep.

        This method is designed for cogames evaluations where each agent has its own
        AgentPolicy instance. It now supports multi-policy scenarios where different
        agents may use different policies (e.g., Candidate vs Thinky vs Ladybug).

        Each agent's policy is called independently with batch_size=1, so each policy
        maintains its own Cortex state. We iterate through all agents and extract
        memory from each agent's specific policy.

        WARNING: This only works when there's one ML agent and multiple scripted
        "policies". A refactor is required to make this work in the case of multiple
        ML policies. Right now, if there are multiple different ML policies they will
        both get logged in the same file and the training won't run correctly.
        (Scripted agents log no actual data, which is why this is fine.)
        (This will be fixed soon.)

        Args:
            policies: List of AgentPolicy instances, one per agent
            env_grid_objects: Dictionary of grid objects from the environment
            object_type_names: List of object type names for agent discovery
        """
        if not self.enabled:
            return

        self.timestep += 1

        if self.timestep == 1:
            # Store object_type_names for use in _build_agent_id_map
            self.object_type_names = object_type_names or []

            # Discover the agent type ID on the first timestep
            # Try to find "agent" in the type names list
            if object_type_names:
                # Find any type name that starts with "agent"
                agent_type_idx = None
                for idx, type_name in enumerate(object_type_names):
                    if type_name and type_name.startswith("agent"):
                        agent_type_idx = idx
                        break

                if agent_type_idx is not None:
                    self.agent_type_id = agent_type_idx
                else:
                    logger.warning(
                        f"Could not find 'agent' type in object_type_names {object_type_names}, "
                        "defaulting to type ID 0."
                    )
                    self.agent_type_id = 0
            else:
                logger.warning("No object_type_names provided, defaulting to type ID 0 for agent discovery.")
                self.agent_type_id = 0

        if not policies:
            return

        # Build agent map from grid objects
        agent_map = self._build_agent_id_map(env_grid_objects)
        timestep_data: Dict[str, Any] = {"timestep": self.timestep, "agents": []}

        # Track unique policies we've seen and warn about them once
        seen_policy_types = set()

        # On first timestep, check for multiple CortexTD policies
        if self.timestep == 1:
            cortex_policies = set()  # Track unique CortexTD policies

            for _, agent_policy in enumerate(policies):
                # Extract the underlying policy for this agent
                underlying_policy = None
                if hasattr(agent_policy, "_policy"):
                    underlying_policy = agent_policy._policy
                elif hasattr(agent_policy, "_base_policy"):
                    base = agent_policy._base_policy
                    if hasattr(base, "_policy"):
                        underlying_policy = base._policy
                    else:
                        underlying_policy = base
                elif hasattr(agent_policy, "_parent"):
                    underlying_policy = agent_policy._parent

                if underlying_policy:
                    has_cortex = self._find_cortex_component(underlying_policy) is not None

                    # Track unique policies with CortexTD
                    if has_cortex:
                        cortex_policies.add(id(underlying_policy))

            # Warn if multiple different CortexTD policies detected
            if len(cortex_policies) > 1:
                logger.warning(
                    "\n" + "=" * 80 + "\n"
                    "WARNING: Multiple different CortexTD policies detected!\n"
                    f"Found {len(cortex_policies)} unique policies with CortexTD components.\n"
                    "Doxascope will log data from all agents into a SINGLE file.\n"
                    "This will mix data from different policies, making training incorrect.\n"
                    "TODO: Implement per-policy logging (separate files per policy).\n"
                    "=" * 80
                )

        # Extract memory for each agent individually
        # Each agent may use a different policy, so we need to extract from their specific policy
        for agent_idx, agent_policy in enumerate(policies):
            # Use agent_idx as agent_id (should match the agent's actual ID in the simulation)
            agent_id = agent_idx

            if agent_id not in agent_map:
                if self.timestep == 1:
                    logger.warning(
                        "Agent ID %d not found in agent_map (available: %s)",
                        agent_id,
                        list(agent_map.keys()),
                    )
                continue

            # Extract the underlying policy for this specific agent
            underlying_policy = None
            if hasattr(agent_policy, "_policy"):
                # Old _SingleAgentAdapter style
                underlying_policy = agent_policy._policy
            elif hasattr(agent_policy, "_base_policy"):
                # New StatefulAgentPolicy style - get the base policy implementation
                base = agent_policy._base_policy
                # The base_policy might wrap a Policy (has _policy attr), or be the policy itself
                if hasattr(base, "_policy"):
                    underlying_policy = base._policy
                else:
                    # For scripted/LSTM policies, _base_policy IS the implementation
                    underlying_policy = base
            elif hasattr(agent_policy, "_parent"):
                # NimAgentPolicy style - get the parent NimMultiAgentPolicy
                underlying_policy = agent_policy._parent

            if underlying_policy is None:
                policy_type = type(agent_policy)
                if self.timestep == 1 and policy_type not in seen_policy_types:
                    logger.warning(
                        "Could not extract underlying policy from agent %d with type %s "
                        "(expected _policy, _base_policy, or _parent attribute)",
                        agent_id,
                        policy_type,
                    )
                    seen_policy_types.add(policy_type)
                continue

            # Find the CortexTD component for this agent's policy
            cortex_component = self._find_cortex_component(underlying_policy)
            if cortex_component is None:
                policy_type = type(underlying_policy)
                if self.timestep == 1 and policy_type not in seen_policy_types:
                    logger.info(
                        "No CortexTD component found in policy for agent %d (type: %s) - skipping",
                        agent_id,
                        policy_type.__name__,
                    )
                    seen_policy_types.add(policy_type)
                continue

            # Get the agent's state from Cortex.
            # The agent_id is used as env_id, so we need to look it up in the rollout store.
            # First, check if the agent's state is in _rollout_current_state (most recent)
            rollout_state = cortex_component._rollout_current_state
            rollout_env_ids = cortex_component._rollout_current_env_ids

            memory_vector = None

            # Check if this agent's state is in the current rollout state
            if rollout_state is not None and rollout_env_ids is not None and rollout_env_ids.numel() > 0:
                # Find the batch position for this agent's env_id
                env_id_matches = (rollout_env_ids == agent_id).nonzero(as_tuple=False)
                if env_id_matches.numel() > 0:
                    batch_pos = int(env_id_matches[0].item())
                    memory_vector = self._extract_cortex_memory_at_position(rollout_state, batch_pos)

            # If not in current state, try to get from stored state via _rollout_id2slot
            if memory_vector is None:
                memory_vector = self._extract_cortex_memory_from_store(cortex_component, agent_id)

            if memory_vector is None:
                if self.timestep == 1:
                    logger.warning("Failed to extract memory for agent %d (state may not be initialized yet)", agent_id)
                continue

            # Get position from grid objects
            grid_obj_id = agent_map[agent_id]
            grid_obj = env_grid_objects[grid_obj_id]
            position = (grid_obj["r"], grid_obj["c"])

            inventory_raw = grid_obj.get("inventory", {})
            if inventory_raw and self.resource_names:
                inventory = {
                    self.resource_names[int(resource_id)]: int(quantity)
                    for resource_id, quantity in inventory_raw.items()
                    if 0 <= int(resource_id) < len(self.resource_names)
                }
            elif inventory_raw:
                inventory = {str(k): int(v) for k, v in inventory_raw.items()}
            else:
                inventory = {}

            record = {
                "agent_id": agent_id,
                "memory_vector": memory_vector.detach().cpu().numpy().astype(np.float32).tolist(),
                "position": position,
                "inventory": inventory,
                "policy_type": type(underlying_policy).__name__,
            }
            timestep_data["agents"].append(record)

        # Only append if we have at least one agent's data
        if timestep_data["agents"]:
            self.data.append(timestep_data)
        else:
            if self.timestep == 1:
                logger.warning(f"Doxascope: No agent data collected at timestep {self.timestep}")

    def _find_cortex_component(self, policy: Any) -> Optional[Any]:
        """Find the CortexTD component in a policy's component list.

        Args:
            policy: The underlying Policy object (e.g., PolicyAutoBuilder)

        Returns:
            The CortexTD component if found, None otherwise
        """
        if not hasattr(policy, "components"):
            return None

        # Search through components for CortexTD
        for comp in policy.components.values():
            # Check by class name to avoid import dependencies
            if comp.__class__.__name__ in ("CortexTD", "CortexStack"):
                return comp

        return None

    def _extract_cortex_memory_at_position(
        self,
        rollout_state: Any,  # TensorDict
        batch_pos: int,
    ) -> Optional[torch.Tensor]:
        """Extract and flatten memory tensors for a specific batch position.

        Args:
            rollout_state: The TensorDict containing Cortex state
            batch_pos: Position in the batch dimension

        Returns:
            Flattened memory vector for the agent at batch_pos
        """
        try:
            # Import optree for flattening the TensorDict tree
            import optree

            # Flatten the entire state tree and extract at batch_pos
            leaves, _ = optree.tree_flatten(rollout_state, namespace="torch")

            # Extract the batch_pos slice from each leaf and concatenate
            memory_parts = []
            for leaf in leaves:
                if isinstance(leaf, torch.Tensor) and leaf.shape[0] > batch_pos:
                    # Extract the slice for this batch position and flatten
                    memory_parts.append(leaf[batch_pos].flatten())

            if not memory_parts:
                return None

            # Concatenate all memory components into one vector
            return torch.cat(memory_parts, dim=0)

        except Exception as e:
            logger.warning(f"Failed to extract Cortex memory at position {batch_pos}: {e}")
            return None

    def _extract_cortex_memory_from_store(self, cortex_component: Any, agent_id: int) -> Optional[torch.Tensor]:
        """Extract memory from Cortex's stored state using agent_id as env_id.

        Args:
            cortex_component: The CortexTD component
            agent_id: The agent ID (used as env_id)

        Returns:
            Flattened memory vector for the agent, or None if not found
        """
        try:
            # Check if agent_id is mapped to a slot
            if not hasattr(cortex_component, "_rollout_id2slot"):
                return None

            slot = cortex_component._rollout_id2slot.get(agent_id)
            if slot is None:
                return None

            # Get the stored state leaves
            if not hasattr(cortex_component, "_rollout_store_leaves") or not cortex_component._rollout_store_leaves:
                return None

            # Import optree for tree operations

            # Reconstruct state from stored leaves at the slot position
            # The leaves are stored as [capacity, *leaf_shape] tensors
            memory_parts = []
            for leaf in cortex_component._rollout_store_leaves:
                if isinstance(leaf, torch.Tensor) and leaf.shape[0] > slot:
                    # Extract the slice for this slot and flatten
                    memory_parts.append(leaf[slot].flatten())

            if not memory_parts:
                return None

            # Concatenate all memory components into one vector
            return torch.cat(memory_parts, dim=0)

        except Exception as e:
            logger.debug(f"Failed to extract Cortex memory from store for agent {agent_id}: {e}")
            return None

    def save(self):
        """Save logged data to JSON file with metadata."""
        if not self.enabled or self.output_file is None:
            return

        try:
            if not self.data:
                # This warning is usually suppressed in eval.py if data is empty, but keeping it here for completeness
                logger.warning("Doxascope: no data was logged for this simulation; nothing to save.")
                return

            # Wrap data with metadata for resource names and other info
            output = {
                "metadata": {
                    "resource_names": self.resource_names,
                    "simulation_id": self.simulation_id,
                },
                "timesteps": self.data,
            }

            with open(self.output_file, "w") as f:
                json.dump(output, f)
            file_size_bytes = self.output_file.stat().st_size
            file_size_mb = file_size_bytes / (1024 * 1024)
            logger.info(f"Doxascope data saved to {self.output_file.name} ({file_size_mb:.2f} MB)")
        except Exception as e:
            logger.error(f"Failed to save doxascope data: {e}")


class DoxascopeEventHandler(SimulatorEventHandler):
    # needs to be passed the Logger object
    def __init__(self, logger: DoxascopeLogger):
        super().__init__()
        self._logger = logger

    def on_step(self):
        super().on_step()
        if not self._logger.enabled:
            return
        env_grid_objects = self._sim.grid_objects()
        policies = self._sim._context.get("policies", [])

        # Set resource_names on first step if not already configured
        if self._logger.resource_names is None:
            self._logger.resource_names = self._sim.resource_names

        self._logger.log_timestep(
            policies=policies,
            env_grid_objects=env_grid_objects,
            object_type_names=self._sim.object_type_names,
        )


@dataclass
class TrajectoryData:
    """Container for extracted trajectory data from raw files."""

    trajectories: List[List[Tuple[np.ndarray, Tuple[int, int], Dict[str, int]]]]
    resource_names: List[str]


def _extract_agent_trajectories(files: list) -> TrajectoryData:
    """
    Load raw data (single-env) and return per-file, per-agent trajectories.
    Each trajectory is a list of (memory_vector, position, inventory) tuples from a single
    file for a single agent.

    Also extracts and unions all resource names across files.
    """
    trajectories: List[List[Tuple[np.ndarray, Tuple[int, int], Dict[str, int]]]] = []
    all_resource_names: set = set()

    for json_file in files:
        with open(json_file, "r") as f:
            raw_data = json.load(f)

        # Handle both old format (list of timesteps) and new format (with metadata)
        if isinstance(raw_data, dict) and "timesteps" in raw_data:
            data = raw_data["timesteps"]
            metadata = raw_data.get("metadata", {})
            file_resource_names = metadata.get("resource_names")
            if file_resource_names:
                all_resource_names.update(file_resource_names)
        else:
            # Old format: raw_data is directly the list of timesteps
            data = raw_data

        file_agent_trajs: Dict[int, List[Tuple[np.ndarray, Tuple[int, int], Dict[str, int]]]] = {}
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
                inventory = agent_data.get("inventory", {})
                # Ensure inventory values are integers
                inventory = {str(k): int(v) for k, v in inventory.items()}

                # Track resource names from inventory keys
                all_resource_names.update(inventory.keys())

                key = int(agent_id)
                file_agent_trajs.setdefault(key, []).append((memory, position, inventory))

        # Append per-agent trajectories for this file
        for traj in file_agent_trajs.values():
            trajectories.append(traj)

    # Sort resource names for consistent ordering
    resource_names = sorted(all_resource_names)

    return TrajectoryData(trajectories=trajectories, resource_names=resource_names)


def _find_items_that_change(
    trajectories: List[List[Tuple[np.ndarray, Tuple[int, int], Dict[str, int]]]],
    all_resource_names: List[str],
) -> List[str]:
    """
    Scan all trajectories to find which items actually change at least once.

    Args:
        trajectories: List of agent trajectories
        all_resource_names: Full list of possible resource names

    Returns:
        List of resource names that actually change (sorted)
    """
    changing_items: set = set()

    for trajectory in trajectories:
        if len(trajectory) < 2:
            continue

        for i in range(len(trajectory) - 1):
            current_inventory = trajectory[i][2]
            next_inventory = trajectory[i + 1][2]

            for item_name in all_resource_names:
                current_val = current_inventory.get(item_name, 0)
                next_val = next_inventory.get(item_name, 0)
                if current_val != next_val:
                    changing_items.add(item_name)

    return sorted(changing_items)


def _find_next_changing_items(
    trajectory: List[Tuple[np.ndarray, Tuple[int, int], Dict[str, int]]],
    current_idx: int,
    resource_names: List[str],
) -> Optional[List[Tuple[int, int]]]:
    """
    From current_idx, find the next timestep where inventory changes and return
    which items change.

    Args:
        trajectory: List of (memory, position, inventory) tuples
        current_idx: Current timestep index in the trajectory
        resource_names: Ordered list of resource names (only items that actually change)

    Returns:
        List of (item_index, time_to_change) tuples for each item that changes,
        or None if no change found. If multiple items change at the same timestep,
        returns one entry per item (all with same time_to_change).
    """
    if not resource_names:
        return None

    current_inventory = trajectory[current_idx][2]

    # Look ahead for the next inventory change
    for future_idx in range(current_idx + 1, len(trajectory)):
        future_inventory = trajectory[future_idx][2]

        # Find all items that changed at this timestep
        changing_items = []
        for i, item_name in enumerate(resource_names):
            current_val = current_inventory.get(item_name, 0)
            future_val = future_inventory.get(item_name, 0)
            if current_val != future_val:
                time_to_change = future_idx - current_idx
                changing_items.append((i, time_to_change))

        if changing_items:
            return changing_items

    # No inventory change found in the rest of the trajectory
    return None


@dataclass
class TrainingSamples:
    """Container for training samples with location and inventory data."""

    memory_vectors: List[np.ndarray]
    location_labels: List[List[int]]
    # inventory_labels: single class index (which item changes next), not multi-hot
    inventory_labels: Optional[List[int]] = None
    time_to_change: Optional[List[int]] = None


def _create_training_samples(
    trajectories: List[List[Tuple[np.ndarray, Tuple[int, int], Dict[str, int]]]],
    num_future: int,
    num_past: int,
    granularity: str = "exact",
    include_inventory: bool = True,
    resource_names: Optional[List[str]] = None,
) -> TrainingSamples:
    """Generate training samples (X, y) from per-agent trajectories.

    Args:
        trajectories: List of trajectories, each a list of (memory, position, inventory)
        num_future: Number of future timesteps to predict
        num_past: Number of past timesteps to predict
        granularity: The prediction granularity ("exact" or "quadrant")
        include_inventory: Whether to include inventory prediction labels
        resource_names: Ordered list of resource names for inventory encoding

    Returns:
        TrainingSamples with memory vectors, location labels, and optionally inventory data.
        For inventory prediction: one sample per item that changes (if multiple items change
        at the same timestep, multiple samples are created with the same memory vector).
    """
    all_memory_vectors: List[np.ndarray] = []
    all_location_labels: List[List[int]] = []
    all_inventory_labels: List[int] = [] if include_inventory else None
    all_time_to_change: List[int] = [] if include_inventory else None

    for trajectory in trajectories:
        if len(trajectory) < num_future + num_past + 1:
            continue

        for i in range(num_past, len(trajectory) - num_future):
            current_memory, current_pos, _ = trajectory[i]
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

            if not valid_sample:
                continue

            # Handle inventory prediction
            if include_inventory and resource_names:
                changing_items = _find_next_changing_items(trajectory, i, resource_names)
                if changing_items is None:
                    # No inventory change found - skip this sample entirely
                    continue

                # Create one sample per item that changes
                # (if multiple items change at same timestep, all are valid "next items")
                for item_idx, ttc in changing_items:
                    all_memory_vectors.append(current_memory)
                    all_location_labels.append(timestep_labels)
                    all_inventory_labels.append(item_idx)
                    all_time_to_change.append(ttc)
            else:
                # Location-only mode
                all_location_labels.append(timestep_labels)
                all_memory_vectors.append(current_memory)

    return TrainingSamples(
        memory_vectors=all_memory_vectors,
        location_labels=all_location_labels,
        inventory_labels=all_inventory_labels if all_inventory_labels else None,
        time_to_change=all_time_to_change if all_time_to_change else None,
    )


@dataclass
class PreprocessedData:
    """Container for preprocessed training data."""

    X: np.ndarray
    y_location: np.ndarray
    y_inventory: Optional[np.ndarray] = None
    time_to_change: Optional[np.ndarray] = None
    resource_names: Optional[List[str]] = None


def find_changing_items_across_files(
    json_files: list,
    exclude_items: Optional[List[str]] = None,
) -> List[str]:
    """
    Scans all JSON files to find which items change in any trajectory.

    This should be called once on ALL files before splitting into train/val/test
    to ensure consistent item lists across all splits.

    Args:
        json_files: List of JSON files to scan
        exclude_items: Optional list of item names to exclude from prediction
            (e.g., ["energy"] to exclude passive resource consumption)

    Returns:
        Sorted list of resource names that change at least once (minus excluded items)
    """
    trajectory_data = _extract_agent_trajectories(json_files)
    if not trajectory_data.trajectories or not trajectory_data.resource_names:
        return []
    changing_items = _find_items_that_change(trajectory_data.trajectories, trajectory_data.resource_names)

    # Filter out excluded items
    if exclude_items:
        excluded_set = set(exclude_items)
        changing_items = [item for item in changing_items if item not in excluded_set]

    return changing_items


def preprocess_doxascope_data(
    json_files: list,
    preprocessed_dir: Path,
    output_filename: str = "training_data.npz",
    num_future_timesteps: int = 1,
    num_past_timesteps: int = 0,
    granularity: str = "exact",
    include_inventory: bool = True,
    resource_names_override: Optional[List[str]] = None,
) -> Optional[PreprocessedData]:
    """
    Preprocesses raw doxascope JSON data to create training-ready NPZ files.

    Args:
        json_files: List of JSON files to process
        preprocessed_dir: Directory to save preprocessed data
        output_filename: Name of the output NPZ file
        num_future_timesteps: Number of future timesteps to predict
        num_past_timesteps: Number of past timesteps to predict
        granularity: Prediction granularity ("exact" or "quadrant")
        include_inventory: Whether to include inventory prediction data
        resource_names_override: If provided, use this list of items instead of auto-detecting.
            This ensures consistent item lists across train/val/test splits.

    Returns:
        PreprocessedData container or None if preprocessing fails
    """
    if not json_files:
        logger.warning("No JSON files provided for preprocessing.")
        return None

    logger.info(f"Processing {len(json_files)} simulation log(s) with '{granularity}' granularity...")

    trajectory_data = _extract_agent_trajectories(json_files)
    if not trajectory_data.trajectories:
        logger.warning("No valid agent trajectories found in the provided files.")
        return None

    resource_names = None
    if include_inventory:
        if resource_names_override:
            # Use provided list (ensures consistency across splits)
            resource_names = resource_names_override
            logger.info(f"Using provided resource list with {len(resource_names)} items: {resource_names}")
        else:
            # Auto-detect changing items (only for single-file processing)
            all_resource_names = trajectory_data.resource_names
            if all_resource_names:
                resource_names = _find_items_that_change(trajectory_data.trajectories, all_resource_names)
                if resource_names:
                    logger.info(
                        f"Found {len(resource_names)} resource types that change: {resource_names} "
                        f"(filtered from {len(all_resource_names)} total)"
                    )
                else:
                    logger.warning("No items change in any trajectory; inventory prediction will be skipped.")
                    include_inventory = False
            else:
                logger.warning("No resource names found in data; inventory prediction will be skipped.")
                include_inventory = False

    samples = _create_training_samples(
        trajectory_data.trajectories,
        num_future_timesteps,
        num_past_timesteps,
        granularity=granularity,
        include_inventory=include_inventory,
        resource_names=resource_names,
    )

    if not samples.memory_vectors:
        logger.warning("No training samples could be created from the trajectories.")
        return None

    try:
        X = np.array(samples.memory_vectors, dtype=np.float32)
        y_location = np.array(samples.location_labels, dtype=np.int64)

        # Prepare data dict for saving
        save_dict = {
            "X": X,
            "y": y_location,  # Keep 'y' for backward compatibility
            "granularity": np.array(granularity),
        }

        # Add inventory data if available
        # y_inventory is now class indices (int64), not multi-hot vectors
        y_inventory = None
        time_to_change = None
        if include_inventory and samples.inventory_labels:
            y_inventory = np.array(samples.inventory_labels, dtype=np.int64)
            time_to_change = np.array(samples.time_to_change, dtype=np.int64)
            save_dict["y_inventory"] = y_inventory
            save_dict["time_to_change"] = time_to_change
            save_dict["resource_names"] = np.array(resource_names, dtype=object)
            logger.info(f"Included inventory labels for {len(resource_names)} resource types")

        output_file = preprocessed_dir / output_filename
        preprocessed_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_file, **save_dict)

        logger.info(f"Successfully saved {len(X)} samples to {output_file}")

        return PreprocessedData(
            X=X,
            y_location=y_location,
            y_inventory=y_inventory,
            time_to_change=time_to_change,
            resource_names=resource_names,
        )
    except ValueError as e:
        logger.error(f"Failed to create NumPy arrays due to inconsistent shapes: {e}")
        unique_dims = {mv.shape for mv in samples.memory_vectors}
        logger.error(f"Found memory vectors with the following shapes: {unique_dims}")
        return None
