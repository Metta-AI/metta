"""Object type constants for map building.

This module provides constants for object type IDs to avoid string literals
and enable int-based map storage. Type IDs should match the type_id values
in GameConfig object definitions.
"""

from typing import Dict, Set


class ObjectTypes:
    """Constants for object type IDs to avoid string literals."""

    # Core terrain types (0-9)
    EMPTY = 0
    WALL = 1
    BLOCK = 2

    # Agent types (10-99) - expandable range for different groups
    AGENT_BASE = 10
    AGENT_DEFAULT = 10  # agent.agent, agent.default
    AGENT_RED = 10  # agent.red, agent.team_0
    AGENT_BLUE = 11  # agent.blue, agent.team_1
    AGENT_GREEN = 12  # agent.green, agent.team_2
    AGENT_YELLOW = 13  # agent.yellow, agent.team_3
    AGENT_PURPLE = 14  # agent.purple, agent.team_4
    AGENT_ORANGE = 15  # agent.orange, agent.team_5
    AGENT_PREY = 20  # agent.prey
    AGENT_PREDATOR = 21  # agent.predator

    # Mines (100-119)
    MINE_RED = 100
    MINE_BLUE = 101
    MINE_GREEN = 102

    # Generators (120-139)
    GENERATOR_RED = 120
    GENERATOR_BLUE = 121
    GENERATOR_GREEN = 122

    # Buildings/Structures (140-199)
    ALTAR = 140
    ARMORY = 141
    LASERY = 142
    LAB = 143
    FACTORY = 144
    TEMPLE = 145
    CONVERTER = 146

    @classmethod
    def agent_type_id(cls, group_id: int) -> int:
        """Get type ID for agent with specific group ID.

        Args:
            group_id: The group/team ID (0-5 map to standard team colors, higher numbers get sequential IDs)

        Returns:
            The type ID for the agent group
        """
        if group_id <= 5:
            return cls.AGENT_BASE + group_id
        else:
            return cls.AGENT_BASE + 6 + group_id  # Skip special agent types

    @classmethod
    def is_agent(cls, type_id: int) -> bool:
        """Check if type ID represents an agent.

        Args:
            type_id: The type ID to check

        Returns:
            True if the type ID represents any kind of agent
        """
        return cls.AGENT_BASE <= type_id < 100

    @classmethod
    def get_standard_mappings(cls) -> Dict[str, int]:
        """Get standard object name to type_id mappings.

        Returns:
            Dictionary mapping standard object names to their type IDs
        """
        return {
            # Core terrain
            "empty": cls.EMPTY,
            "wall": cls.WALL,
            "block": cls.BLOCK,
            # Agent aliases
            "agent.agent": cls.AGENT_DEFAULT,
            "agent.default": cls.AGENT_DEFAULT,
            "agent.red": cls.AGENT_RED,
            "agent.team_0": cls.AGENT_RED,
            "agent.blue": cls.AGENT_BLUE,
            "agent.team_1": cls.AGENT_BLUE,
            "agent.green": cls.AGENT_GREEN,
            "agent.team_2": cls.AGENT_GREEN,
            "agent.yellow": cls.AGENT_YELLOW,
            "agent.team_3": cls.AGENT_YELLOW,
            "agent.purple": cls.AGENT_PURPLE,
            "agent.team_4": cls.AGENT_PURPLE,
            "agent.orange": cls.AGENT_ORANGE,
            "agent.team_5": cls.AGENT_ORANGE,
            "agent.prey": cls.AGENT_PREY,
            "agent.predator": cls.AGENT_PREDATOR,
            # Mines
            "mine_red": cls.MINE_RED,
            "mine_blue": cls.MINE_BLUE,
            "mine_green": cls.MINE_GREEN,
            # Generators
            "generator_red": cls.GENERATOR_RED,
            "generator_blue": cls.GENERATOR_BLUE,
            "generator_green": cls.GENERATOR_GREEN,
            # Buildings
            "altar": cls.ALTAR,
            "armory": cls.ARMORY,
            "lasery": cls.LASERY,
            "lab": cls.LAB,
            "factory": cls.FACTORY,
            "temple": cls.TEMPLE,
            "converter": cls.CONVERTER,
        }

    @classmethod
    def get_reverse_mappings(cls) -> Dict[int, str]:
        """Get type_id to standard object name mappings.

        Returns:
            Dictionary mapping type IDs to their primary object names
        """
        # Use primary names (first alias for each type)
        return {
            cls.EMPTY: "empty",
            cls.WALL: "wall",
            cls.BLOCK: "block",
            cls.AGENT_DEFAULT: "agent.agent",
            cls.AGENT_BLUE: "agent.blue",
            cls.AGENT_GREEN: "agent.green",
            cls.AGENT_YELLOW: "agent.yellow",
            cls.AGENT_PURPLE: "agent.purple",
            cls.AGENT_ORANGE: "agent.orange",
            cls.AGENT_PREY: "agent.prey",
            cls.AGENT_PREDATOR: "agent.predator",
            cls.MINE_RED: "mine_red",
            cls.MINE_BLUE: "mine_blue",
            cls.MINE_GREEN: "mine_green",
            cls.GENERATOR_RED: "generator_red",
            cls.GENERATOR_BLUE: "generator_blue",
            cls.GENERATOR_GREEN: "generator_green",
            cls.ALTAR: "altar",
            cls.ARMORY: "armory",
            cls.LASERY: "lasery",
            cls.LAB: "lab",
            cls.FACTORY: "factory",
            cls.TEMPLE: "temple",
            cls.CONVERTER: "converter",
        }

    @classmethod
    def get_agent_type_ids(cls) -> Set[int]:
        """Get all type IDs that represent agents.

        Returns:
            Set of all agent type IDs
        """
        return {
            cls.AGENT_DEFAULT,
            cls.AGENT_BLUE,
            cls.AGENT_GREEN,
            cls.AGENT_YELLOW,
            cls.AGENT_PURPLE,
            cls.AGENT_ORANGE,
            cls.AGENT_PREY,
            cls.AGENT_PREDATOR,
        }
