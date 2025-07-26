"""
Simple registry system for mapping environment names to neural network indices.
Used for both observation features and actions to ensure consistent handling.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class Registry:
    """Maps names to indices with support for saving/loading and environment changes."""

    def __init__(self, name: str, unknown_id: int = 255):
        self.name = name
        self.unknown_id = unknown_id
        self.name_to_id: Dict[str, int] = {}
        self.next_id = 0

    def initialize(self, names: List[str], device: torch.device) -> torch.Tensor:
        """Initialize registry with environment names and return active indices.

        Args:
            names: List of names from the environment
            device: Device to place the indices tensor on

        Returns:
            Tensor of indices for the active names
        """
        indices = []
        new_names = []

        for name in names:
            if name not in self.name_to_id:
                # Assign new ID
                self.name_to_id[name] = self.next_id
                self.next_id += 1
                new_names.append(name)
            indices.append(self.name_to_id[name])

        if new_names:
            logger.info(f"{self.name}: Added {len(new_names)} new entries: {new_names[:5]}...")

        return torch.tensor(indices, device=device, dtype=torch.long)

    def get_remapping(self, current_names: List[str], is_training: bool) -> Optional[Dict[int, int]]:
        """Get remapping from current environment IDs to stored IDs.

        Args:
            current_names: Names with their positions as implicit IDs
            is_training: Whether we're in training mode

        Returns:
            Dict mapping current position -> stored ID, or None if no remapping needed
        """
        remap = {}
        unknown_names = []

        for current_id, name in enumerate(current_names):
            if name in self.name_to_id:
                stored_id = self.name_to_id[name]
                if stored_id != current_id:
                    remap[current_id] = stored_id
            elif not is_training:
                # In eval mode, map unknown entries to unknown_id
                remap[current_id] = self.unknown_id
                unknown_names.append(name)
            else:
                # In training mode, learn new entries
                self.name_to_id[name] = self.next_id
                self.next_id += 1
                if self.next_id != current_id:
                    remap[current_id] = self.next_id - 1

        if unknown_names:
            logger.info(f"{self.name}: {len(unknown_names)} unknown entries mapped to ID {self.unknown_id}")

        return remap if remap else None

    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary for saving."""
        return {
            "name_to_id": self.name_to_id,
            "next_id": self.next_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, any], name: str, unknown_id: int = 255) -> "Registry":
        """Create from saved dictionary."""
        registry = cls(name, unknown_id)
        registry.name_to_id = data["name_to_id"]
        registry.next_id = data["next_id"]
        return registry


class FeatureRegistry(Registry):
    """Registry for observation features with normalization support."""

    def __init__(self):
        super().__init__("features", unknown_id=255)
        self.normalizations: Dict[int, float] = {}

    @classmethod
    def from_dict(cls, data: Dict[str, any], name: str = "features", unknown_id: int = 255) -> "FeatureRegistry":
        """Create from saved dictionary."""
        registry = cls()
        registry.name_to_id = data["name_to_id"]
        registry.next_id = data["next_id"]
        if "normalizations" in data:
            registry.normalizations = data["normalizations"]
        return registry

    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary for saving."""
        data = super().to_dict()
        data["normalizations"] = self.normalizations
        return data

    def initialize_with_features(
        self, features: Dict[str, Dict], device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize with feature dictionary from environment.

        Args:
            features: Dict mapping feature names to properties including 'id' and optional 'normalization'
            device: Device for tensors

        Returns:
            Tuple of (remapping tensor, normalization tensor)
        """
        # Build remapping from environment IDs to our stored IDs
        remap_tensor = torch.arange(256, dtype=torch.uint8, device=device)
        norm_tensor = torch.ones(256, dtype=torch.float32, device=device)

        for name, props in features.items():
            env_id = props["id"]

            if name not in self.name_to_id:
                # New feature - assign next ID
                self.name_to_id[name] = self.next_id
                self.next_id += 1

            stored_id = self.name_to_id[name]
            remap_tensor[env_id] = stored_id

            # Store normalization
            if "normalization" in props:
                self.normalizations[stored_id] = props["normalization"]
                norm_tensor[stored_id] = props["normalization"]

        return remap_tensor, norm_tensor


class ActionRegistry(Registry):
    """Registry for actions with automatic embedding expansion."""

    def __init__(self):
        super().__init__("actions", unknown_id=None)  # No unknown actions

    @classmethod
    def from_dict(cls, data: Dict[str, any], name: str = "actions", unknown_id: int = None) -> "ActionRegistry":
        """Create from saved dictionary."""
        registry = cls()
        registry.name_to_id = data["name_to_id"]
        registry.next_id = data["next_id"]
        return registry

    def initialize_with_actions(
        self, action_names: List[str], action_max_params: List[int], device: torch.device
    ) -> torch.Tensor:
        """Initialize with action names and parameters.

        Args:
            action_names: List of action type names
            action_max_params: List of max parameters per action
            device: Device for tensors

        Returns:
            Tensor of indices for all (action, param) pairs
        """
        full_names = []
        for action_name, max_param in zip(action_names, action_max_params, strict=False):
            for i in range(max_param + 1):
                full_names.append(f"{action_name}_{i}")

        return self.initialize(full_names, device)

    def ensure_embedding_size(self, embedding_layer: torch.nn.Embedding) -> torch.nn.Embedding:
        """Ensure embedding layer is large enough for all registered actions.

        Args:
            embedding_layer: Current embedding layer

        Returns:
            Same layer or new expanded layer if needed
        """
        required_size = self.next_id + 1  # +1 for 0-based indexing

        if required_size <= embedding_layer.num_embeddings:
            return embedding_layer

        # Need to expand
        logger.info(f"Expanding action embeddings from {embedding_layer.num_embeddings} to {required_size}")

        new_embed = torch.nn.Embedding(
            required_size, embedding_layer.embedding_dim, device=embedding_layer.weight.device
        )

        # Initialize new weights
        torch.nn.init.orthogonal_(new_embed.weight)
        with torch.no_grad():
            max_abs = torch.max(torch.abs(new_embed.weight))
            new_embed.weight.mul_(0.1 / max_abs)

            # Copy existing weights
            new_embed.weight[: embedding_layer.num_embeddings] = embedding_layer.weight

        return new_embed
