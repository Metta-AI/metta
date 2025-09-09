"""Type mapping utilities for converting between object names and type IDs.

This module provides utilities for building bidirectional mappings between
object names and type IDs based on GameConfig definitions.
"""

from typing import Dict, List, Optional

from metta.mettagrid.mettagrid_config import GameConfig
from metta.mettagrid.object_types import ObjectTypes


class TypeMapping:
    """Bidirectional mapping between object names and type IDs."""

    def __init__(self, game_config: Optional[GameConfig] = None):
        """Initialize type mapping.

        Args:
            game_config: Optional GameConfig to build mapping from. If None, uses standard mappings.
        """
        self.name_to_type_id: Dict[str, int] = {}
        self.type_id_to_name: Dict[int, str] = {}
        self.decoder_key: List[str] = []

        if game_config:
            self.build_from_game_config(game_config)
        else:
            self.build_standard_mapping()

    def build_standard_mapping(self):
        """Build mapping using standard ObjectTypes constants."""
        self.name_to_type_id = ObjectTypes.get_standard_mappings()
        self.type_id_to_name = ObjectTypes.get_reverse_mappings()
        self._build_decoder_key()

    def build_from_game_config(self, game_config: GameConfig):
        """Build mapping from GameConfig object definitions.

        Args:
            game_config: The game configuration to build mapping from

        Raises:
            ValueError: If there are type_id conflicts or missing type_ids
        """
        self.name_to_type_id = {"empty": ObjectTypes.EMPTY}
        self.type_id_to_name = {ObjectTypes.EMPTY: "empty"}

        # Track used type_ids to detect conflicts
        used_type_ids = {ObjectTypes.EMPTY}

        # Process objects from GameConfig
        for obj_name, obj_config in game_config.objects.items():
            if not hasattr(obj_config, "type_id"):
                raise ValueError(f"Object {obj_name} missing type_id")

            type_id = obj_config.type_id

            if type_id in used_type_ids:
                existing_name = self.type_id_to_name.get(type_id, "unknown")
                raise ValueError(f"Type ID conflict: {obj_name} and {existing_name} both use type_id {type_id}")

            self.name_to_type_id[obj_name] = type_id
            self.type_id_to_name[type_id] = obj_name
            used_type_ids.add(type_id)

        # Add standard agent aliases if agents are defined
        self._add_agent_aliases(game_config)
        self._build_decoder_key()

    def _add_agent_aliases(self, game_config: GameConfig):
        """Add standard agent aliases based on team configurations."""
        # Add team-based aliases for agents that exist in the config
        team_names = {0: "red", 1: "blue", 2: "green", 3: "yellow", 4: "purple", 5: "orange"}

        for obj_name, obj_config in game_config.objects.items():
            if obj_name.startswith("agent."):
                type_id = obj_config.type_id

                # Add team_X alias if this is a team agent
                for team_id, team_name in team_names.items():
                    if obj_name == f"agent.{team_name}":
                        alias = f"agent.team_{team_id}"
                        self.name_to_type_id[alias] = type_id
                        break

                # Add agent.agent alias for the default agent
                if obj_name == "agent.red" or obj_name == "agent.default":
                    self.name_to_type_id["agent.agent"] = type_id

    def _build_decoder_key(self):
        """Build decoder key list for converting type_ids back to names."""
        if not self.type_id_to_name:
            return

        max_type_id = max(self.type_id_to_name.keys())
        self.decoder_key = [""] * (max_type_id + 1)

        for type_id, name in self.type_id_to_name.items():
            self.decoder_key[type_id] = name

    def get_type_id(self, obj_name: str) -> int:
        """Get type ID for object name.

        Args:
            obj_name: The object name to look up

        Returns:
            The type ID for the object

        Raises:
            KeyError: If object name is not found
        """
        if obj_name not in self.name_to_type_id:
            raise KeyError(f"Unknown object name: {obj_name}")
        return self.name_to_type_id[obj_name]

    def get_name(self, type_id: int) -> str:
        """Get object name for type ID.

        Args:
            type_id: The type ID to look up

        Returns:
            The object name for the type ID

        Raises:
            KeyError: If type ID is not found
        """
        if type_id not in self.type_id_to_name:
            raise KeyError(f"Unknown type ID: {type_id}")
        return self.type_id_to_name[type_id]

    def has_name(self, obj_name: str) -> bool:
        """Check if object name exists in mapping.

        Args:
            obj_name: The object name to check

        Returns:
            True if the object name exists
        """
        return obj_name in self.name_to_type_id

    def has_type_id(self, type_id: int) -> bool:
        """Check if type ID exists in mapping.

        Args:
            type_id: The type ID to check

        Returns:
            True if the type ID exists
        """
        return type_id in self.type_id_to_name

    def get_decoder_key(self) -> List[str]:
        """Get decoder key for converting type_id arrays back to names.

        Returns:
            List where index=type_id and value=object_name
        """
        return self.decoder_key.copy()

    def validate_config(self, game_config: GameConfig) -> List[str]:
        """Validate that all objects in GameConfig have proper type_id assignments.

        Args:
            game_config: The game configuration to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        used_type_ids = set()

        for obj_name, obj_config in game_config.objects.items():
            if not hasattr(obj_config, "type_id"):
                errors.append(f"Object {obj_name} missing type_id")
                continue

            type_id = obj_config.type_id

            if type_id in used_type_ids:
                errors.append(f"Duplicate type_id {type_id} used by {obj_name}")
            else:
                used_type_ids.add(type_id)

            if type_id < 0 or type_id > 255:
                errors.append(f"Object {obj_name} has invalid type_id {type_id} (must be 0-255)")

        return errors
