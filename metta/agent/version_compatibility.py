"""
Version compatibility checking for MettaAgent checkpoints.

This module provides utilities to check compatibility between:
- Checkpoint format versions
- Observation space versions
- Action space versions
- Layer architecture versions
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class CompatibilityLevel(Enum):
    """Levels of compatibility between versions."""

    FULL = "full"  # Fully compatible, no issues expected
    PARTIAL = "partial"  # May work with warnings
    INCOMPATIBLE = "incompatible"  # Will not work
    UNKNOWN = "unknown"  # Cannot determine


@dataclass
class VersionInfo:
    """Container for version information."""

    checkpoint_format: Optional[int] = None
    observation_space: Optional[str] = None
    action_space: Optional[str] = None
    layer_architecture: Optional[str] = None

    @classmethod
    def from_checkpoint(cls, checkpoint: Dict) -> "VersionInfo":
        """Extract version info from a checkpoint."""
        return cls(
            checkpoint_format=checkpoint.get("checkpoint_format_version"),
            observation_space=checkpoint.get("observation_space_version"),
            action_space=checkpoint.get("action_space_version"),
            layer_architecture=checkpoint.get("layer_version"),
        )


@dataclass
class CompatibilityReport:
    """Report of compatibility between two versions."""

    level: CompatibilityLevel
    warnings: List[str]
    errors: List[str]
    recommendations: List[str]

    def is_compatible(self) -> bool:
        """Check if versions are compatible enough to proceed."""
        return self.level in (CompatibilityLevel.FULL, CompatibilityLevel.PARTIAL)

    def __str__(self) -> str:
        """Format the report as a string."""
        lines = [f"Compatibility Level: {self.level.value}"]

        if self.errors:
            lines.append("\nErrors:")
            for error in self.errors:
                lines.append(f"  - {error}")

        if self.warnings:
            lines.append("\nWarnings:")
            for warning in self.warnings:
                lines.append(f"  - {warning}")

        if self.recommendations:
            lines.append("\nRecommendations:")
            for rec in self.recommendations:
                lines.append(f"  - {rec}")

        return "\n".join(lines)


class VersionCompatibilityChecker:
    """Checks compatibility between different versions of MettaAgent components."""

    # Define compatibility matrices
    CHECKPOINT_FORMAT_COMPATIBILITY = {
        # (from_version, to_version): CompatibilityLevel
        (1, 2): CompatibilityLevel.FULL,  # v1 can be loaded by v2
        (2, 1): CompatibilityLevel.INCOMPATIBLE,  # v2 cannot be loaded by v1
        (None, 2): CompatibilityLevel.PARTIAL,  # Unknown version, try anyway
    }

    # Known observation space versions and their compatibility
    OBSERVATION_SPACE_VERSIONS = {
        "v1": {
            "description": "Original 128-channel tokenized observation",
            "compatible_with": ["v1"],
        },
        "v2": {
            "description": "Extended observation with additional channels",
            "compatible_with": ["v1", "v2"],  # v2 can handle v1
        },
    }

    # Known action space versions
    ACTION_SPACE_VERSIONS = {
        "v1": {
            "description": "9 actions with up to 10 parameters",
            "action_count": 9,
            "max_params": 10,
        },
        "v2": {
            "description": "Extended action space with new actions",
            "action_count": 12,
            "max_params": 15,
        },
    }

    def check_compatibility(self, checkpoint_version: VersionInfo, runtime_version: VersionInfo) -> CompatibilityReport:
        """Check compatibility between checkpoint and runtime versions."""
        warnings = []
        errors = []
        recommendations = []

        # Check checkpoint format compatibility
        format_compat = self._check_format_compatibility(
            checkpoint_version.checkpoint_format, runtime_version.checkpoint_format
        )

        if format_compat == CompatibilityLevel.INCOMPATIBLE:
            errors.append(
                f"Checkpoint format v{checkpoint_version.checkpoint_format} "
                f"is not compatible with runtime v{runtime_version.checkpoint_format}"
            )
            recommendations.append(
                "Use the checkpoint migration tool: python -m metta.agent.migrate_checkpoints <checkpoint_path>"
            )
        elif format_compat == CompatibilityLevel.PARTIAL:
            warnings.append("Checkpoint format version is unknown, loading may fail")

        # Check observation space compatibility
        if checkpoint_version.observation_space and runtime_version.observation_space:
            obs_compat = self._check_observation_compatibility(
                checkpoint_version.observation_space, runtime_version.observation_space
            )
            if not obs_compat:
                errors.append(
                    f"Observation space mismatch: checkpoint uses "
                    f"{checkpoint_version.observation_space}, runtime expects "
                    f"{runtime_version.observation_space}"
                )

        # Check action space compatibility
        if checkpoint_version.action_space and runtime_version.action_space:
            action_compat = self._check_action_compatibility(
                checkpoint_version.action_space, runtime_version.action_space
            )
            if not action_compat:
                warnings.append(
                    f"Action space version mismatch: checkpoint uses "
                    f"{checkpoint_version.action_space}, runtime uses "
                    f"{runtime_version.action_space}. "
                    "Some actions may not work correctly."
                )

        # Check layer architecture compatibility
        if checkpoint_version.layer_architecture != runtime_version.layer_architecture:
            if checkpoint_version.layer_architecture and runtime_version.layer_architecture:
                warnings.append(
                    f"Layer architecture version mismatch: checkpoint uses "
                    f"{checkpoint_version.layer_architecture}, runtime uses "
                    f"{runtime_version.layer_architecture}"
                )

        # Determine overall compatibility level
        if errors:
            level = CompatibilityLevel.INCOMPATIBLE
        elif warnings:
            level = CompatibilityLevel.PARTIAL
        else:
            level = CompatibilityLevel.FULL

        return CompatibilityReport(level=level, warnings=warnings, errors=errors, recommendations=recommendations)

    def _check_format_compatibility(self, from_version: Optional[int], to_version: Optional[int]) -> CompatibilityLevel:
        """Check checkpoint format compatibility."""
        if from_version == to_version:
            return CompatibilityLevel.FULL

        key = (from_version, to_version)
        if key in self.CHECKPOINT_FORMAT_COMPATIBILITY:
            return self.CHECKPOINT_FORMAT_COMPATIBILITY[key]

        # Default cases
        if from_version is None:
            return CompatibilityLevel.PARTIAL
        if to_version is None:
            return CompatibilityLevel.UNKNOWN
        if from_version > to_version:
            return CompatibilityLevel.INCOMPATIBLE

        return CompatibilityLevel.PARTIAL

    def _check_observation_compatibility(self, checkpoint_version: str, runtime_version: str) -> bool:
        """Check if observation spaces are compatible."""
        if checkpoint_version == runtime_version:
            return True

        runtime_info = self.OBSERVATION_SPACE_VERSIONS.get(runtime_version, {})
        compatible_versions = runtime_info.get("compatible_with", [])

        return checkpoint_version in compatible_versions

    def _check_action_compatibility(self, checkpoint_version: str, runtime_version: str) -> bool:
        """Check if action spaces are compatible."""
        if checkpoint_version == runtime_version:
            return True

        # For now, action spaces must match exactly
        # Future versions could handle subset compatibility
        return False

    def get_version_info(self, version_type: str, version: str) -> Optional[Dict]:
        """Get information about a specific version."""
        if version_type == "observation_space":
            return self.OBSERVATION_SPACE_VERSIONS.get(version)
        elif version_type == "action_space":
            return self.ACTION_SPACE_VERSIONS.get(version)
        return None


# Global compatibility checker instance
compatibility_checker = VersionCompatibilityChecker()


def check_checkpoint_compatibility(
    checkpoint_path: str, runtime_version: Optional[VersionInfo] = None
) -> CompatibilityReport:
    """
    Convenience function to check checkpoint compatibility.

    Args:
        checkpoint_path: Path to the checkpoint file
        runtime_version: Current runtime version info (if None, uses current)

    Returns:
        CompatibilityReport with details about compatibility
    """
    import torch

    # Load checkpoint header (minimal data)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_version = VersionInfo.from_checkpoint(checkpoint)

    # Get runtime version if not provided
    if runtime_version is None:
        runtime_version = VersionInfo(
            checkpoint_format=2,  # Current format version
            observation_space="v1",  # Would be dynamically determined
            action_space="v1",  # Would be dynamically determined
            layer_architecture="v1",  # Would be dynamically determined
        )

    return compatibility_checker.check_compatibility(checkpoint_version, runtime_version)
