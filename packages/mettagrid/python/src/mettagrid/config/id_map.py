"""Observation feature ID mapping for MettaGrid.

This module provides the IdMap class which manages observation feature IDs
and their mappings, along with the ObservationFeatureSpec class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from mettagrid.config.mettagrid_config import GameConfig, MettaGridConfig


class ObservationFeatureSpec(BaseModel):
    """Specification for an observation feature."""

    model_config = ConfigDict(protected_namespaces=())

    id: int
    name: str
    normalization: float


class IdMap:
    """Manages observation feature IDs and mappings for a MettaGrid configuration."""

    def __init__(self, config: MettaGridConfig):
        self._config = config
        self._features_list: list[ObservationFeatureSpec] | None = None

    def features(self) -> list[ObservationFeatureSpec]:
        """Get the list of observation features, computing them once on first access."""
        if self._features_list is None:
            self._features_list = self._compute_features()
        return self._features_list

    def feature_id(self, name: str) -> int:
        """Get the ID for a named feature."""
        feature_ids = self._feature_ids_map()
        if name not in feature_ids:
            raise KeyError(f"Unknown observation feature: {name}")
        return feature_ids[name]

    def feature(self, name: str) -> ObservationFeatureSpec:
        """Get the feature spec for a named feature."""
        for feat in self.features():
            if feat.name == name:
                return feat
        raise KeyError(f"Unknown observation feature: {name}")

    def _feature_ids_map(self) -> dict[str, int]:
        """Get mapping of feature names to IDs."""
        return {feature.name: feature.id for feature in self.features()}

    def tag_names(self) -> dict[int, str]:
        """Get mapping of tag IDs to tag names.

        Returns a dictionary mapping tag ID (int) to tag name (str).
        Tags are sorted alphabetically and assigned consecutive IDs starting from 0.
        """
        # Collect all unique tags from all objects
        all_tags = set()
        game_cfg = self._config.game

        # Collect tags from objects
        for obj_config in game_cfg.objects.values():
            all_tags.update(obj_config.tags)

        # Collect tags from agents
        for agent_config in game_cfg.agents:
            all_tags.update(agent_config.tags)

        # Sort tags alphabetically and create mapping
        sorted_tags = sorted(all_tags)
        tag_id_to_name = {i: tag for i, tag in enumerate(sorted_tags)}

        return tag_id_to_name

    @staticmethod
    def assign_type_ids(game_config: GameConfig) -> None:
        """Auto-assign type_ids for objects that don't have one, filling gaps deterministically."""
        # Access objects directly via __dict__ to avoid recursion
        objects = object.__getattribute__(game_config, "objects")
        if not objects:
            return

        # Collect explicit type_ids
        explicit_type_ids = {obj.type_id for obj in objects.values() if obj.type_id is not None}

        # Sort objects by name for deterministic assignment
        sorted_objects = sorted(objects.items())

        # Start type_id assignment at 1 (0 is reserved for agents)
        next_type_id = 1
        for object_name, object_config in sorted_objects:
            if object_config.type_id is None:
                # Find next available type_id, skipping explicit ones
                while next_type_id in explicit_type_ids:
                    next_type_id += 1

                # Check if we exceed uint8 range
                if next_type_id > 255:
                    raise ValueError(
                        f"auto-generated type_id exceeds uint8 range (255). "
                        f"Object '{object_name}' would be assigned type_id {next_type_id}. "
                        f"Reduce the number of objects or use explicit type_ids to manage the range."
                    )

                object_config.type_id = next_type_id
                next_type_id += 1

    def _compute_features(self) -> list[ObservationFeatureSpec]:
        """Compute observation features from the game configuration."""
        game_cfg = self._config.game

        features: list[ObservationFeatureSpec] = []
        feature_id = 0

        # Core features (fixed set)
        core_features = [
            ("type_id", 1.0),
            ("agent:group", 10.0),
            ("agent:frozen", 1.0),
            ("agent:orientation", 1.0),
            ("agent:reserved_for_future_use", 255.0),
            ("converting", 1.0),
            ("swappable", 1.0),
        ]

        for name, normalization in core_features:
            features.append(ObservationFeatureSpec(id=feature_id, normalization=normalization, name=name))
            feature_id += 1

        # Global observation features (always included for feature_ids, config controls if populated)
        features.append(ObservationFeatureSpec(id=feature_id, normalization=255.0, name="episode_completion_pct"))
        feature_id += 1

        features.append(ObservationFeatureSpec(id=feature_id, normalization=10.0, name="last_action"))
        feature_id += 1
        features.append(ObservationFeatureSpec(id=feature_id, normalization=10.0, name="last_action_arg"))
        feature_id += 1

        features.append(ObservationFeatureSpec(id=feature_id, normalization=100.0, name="last_reward"))
        feature_id += 1

        # Agent-specific features
        features.append(ObservationFeatureSpec(id=feature_id, normalization=255.0, name="vibe"))
        feature_id += 1

        # Visitation counts (always include in feature list, even if disabled)
        features.append(ObservationFeatureSpec(id=feature_id, normalization=1000.0, name="agent:visitation_counts"))
        feature_id += 1

        # Tag feature (always included)
        features.append(ObservationFeatureSpec(id=feature_id, normalization=10.0, name="tag"))
        feature_id += 1

        # Object features
        features.append(ObservationFeatureSpec(id=feature_id, normalization=255.0, name="cooldown_remaining"))
        feature_id += 1

        features.append(ObservationFeatureSpec(id=feature_id, normalization=1.0, name="clipped"))
        feature_id += 1

        features.append(ObservationFeatureSpec(id=feature_id, normalization=255.0, name="remaining_uses"))
        feature_id += 1

        # Inventory features (one per resource)
        for resource_name in game_cfg.resource_names:
            features.append(ObservationFeatureSpec(id=feature_id, normalization=100.0, name=f"inv:{resource_name}"))
            feature_id += 1

        # Protocol details features (if enabled)
        if game_cfg.protocol_details_obs:
            for resource_name in game_cfg.resource_names:
                features.append(
                    ObservationFeatureSpec(id=feature_id, normalization=100.0, name=f"protocol_input:{resource_name}")
                )
                feature_id += 1

            for resource_name in game_cfg.resource_names:
                features.append(
                    ObservationFeatureSpec(id=feature_id, normalization=100.0, name=f"protocol_output:{resource_name}")
                )
                feature_id += 1

        return features
