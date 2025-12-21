"""Observation feature ID mapping for MettaGrid.

This module provides the IdMap class which manages observation feature IDs
and their mappings, along with the ObservationFeatureSpec class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from mettagrid.config.mettagrid_config import GameConfig


class ObservationFeatureSpec(BaseModel):
    """Specification for an observation feature."""

    model_config = ConfigDict(protected_namespaces=())

    id: int
    name: str
    normalization: float


class IdMap:
    """Manages observation feature IDs and mappings for a MettaGrid configuration."""

    def __init__(self, config: GameConfig):
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

    def tag_names(self) -> list[str]:
        """Get all tag names in alphabetical order."""

        result = sorted(
            set(
                [tag for obj_config in self._config.objects.values() for tag in obj_config.tags]
                + [tag for agent_config in self._config.agents for tag in agent_config.tags]
                + self._config.agent.tags
            )
        )
        return result

    def _compute_features(self) -> list[ObservationFeatureSpec]:
        """Compute observation features from the game configuration."""

        features: list[ObservationFeatureSpec] = []
        feature_id = 0

        # Core features (fixed set)
        core_features = [
            ("agent:group", 10.0),
            ("agent:frozen", 1.0),
        ]

        for name, normalization in core_features:
            features.append(ObservationFeatureSpec(id=feature_id, normalization=normalization, name=name))
            feature_id += 1

        # Global observation features (always included for feature_ids, config controls if populated)
        features.append(ObservationFeatureSpec(id=feature_id, normalization=255.0, name="episode_completion_pct"))
        feature_id += 1

        features.append(ObservationFeatureSpec(id=feature_id, normalization=10.0, name="last_action"))
        feature_id += 1

        features.append(ObservationFeatureSpec(id=feature_id, normalization=100.0, name="last_reward"))
        feature_id += 1

        # Goal feature (for indicating rewarding resources)
        features.append(ObservationFeatureSpec(id=feature_id, normalization=100.0, name="goal"))
        feature_id += 1

        # Agent-specific features
        features.append(ObservationFeatureSpec(id=feature_id, normalization=255.0, name="vibe"))
        feature_id += 1

        # Compass direction toward assembler
        features.append(ObservationFeatureSpec(id=feature_id, normalization=1.0, name="agent:compass"))
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
        for resource_name in self._config.resource_names:
            features.append(ObservationFeatureSpec(id=feature_id, normalization=100.0, name=f"inv:{resource_name}"))
            feature_id += 1

        # Protocol details features (if enabled)
        if self._config.protocol_details_obs:
            for resource_name in self._config.resource_names:
                features.append(
                    ObservationFeatureSpec(id=feature_id, normalization=100.0, name=f"protocol_input:{resource_name}")
                )
                feature_id += 1

            for resource_name in self._config.resource_names:
                features.append(
                    ObservationFeatureSpec(id=feature_id, normalization=100.0, name=f"protocol_output:{resource_name}")
                )
                feature_id += 1

        return features
