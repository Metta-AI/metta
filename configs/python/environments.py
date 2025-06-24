"""Environment configurations defined in Python.

These replace the complex YAML environment configs with Python functions.
"""

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class EnvConfig:
    """Base configuration for an environment."""

    name: str
    max_steps: int = 1000
    num_agents: int = 1
    width: int = 35
    height: int = 35

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format expected by the environment."""
        return {
            "game": {
                "max_steps": self.max_steps,
                "map_builder": self.get_map_builder(),
            }
        }

    def get_map_builder(self) -> Dict[str, Any]:
        """Override this to define the map builder configuration."""
        raise NotImplementedError


# Navigation environments
class NavigationEmpty(EnvConfig):
    """Empty space navigation environment."""

    name: str = "navigation/empty"

    def get_map_builder(self) -> Dict[str, Any]:
        return {
            "_target_": "mettagrid.room.mean_distance.MeanDistance",
            "width": self.width,
            "height": self.height,
            "mean_distance": 25,
            "border_width": 3,
            "agents": self.num_agents,
            "objects": {"altar": 3},
        }


class NavigationWalls(EnvConfig):
    """Navigation with walls."""

    name: str = "navigation/walls"

    def get_map_builder(self) -> Dict[str, Any]:
        return {
            "_target_": "mettagrid.room.mean_distance.MeanDistance",
            "width": self.width,
            "height": self.height,
            "mean_distance": 25,
            "border_width": 3,
            "agents": self.num_agents,
            "objects": {"altar": 3, "wall": 12},
        }


class NavigationObstacles(EnvConfig):
    """Navigation with various obstacles."""

    name: str = "navigation/obstacles"
    difficulty: int = 0

    def get_map_builder(self) -> Dict[str, Any]:
        obstacle_counts = [
            {"wall": 5, "altar": 3},
            {"wall": 10, "altar": 3, "mine": 2},
            {"wall": 15, "altar": 3, "mine": 4},
            {"wall": 20, "altar": 3, "mine": 6, "armory": 2},
        ]
        return {
            "_target_": "mettagrid.room.mean_distance.MeanDistance",
            "width": self.width,
            "height": self.height,
            "mean_distance": 25,
            "border_width": 3,
            "agents": self.num_agents,
            "objects": obstacle_counts[min(self.difficulty, 3)],
        }


class NavigationMaze(EnvConfig):
    """Maze navigation environment."""

    name: str = "navigation/maze"
    complexity: float = 0.5

    def get_map_builder(self) -> Dict[str, Any]:
        return {
            "_target_": "mettagrid.room.maze.Maze",
            "width": self.width,
            "height": self.height,
            "complexity": self.complexity,
            "density": 0.5,
            "agents": self.num_agents,
            "objects": {"altar": 3},
        }


# Memory environments
class MemorySequence(EnvConfig):
    """Memory sequence task."""

    name: str = "memory/sequence"
    sequence_length: int = 3

    def get_map_builder(self) -> Dict[str, Any]:
        return {
            "_target_": "mettagrid.room.memory_sequence.MemorySequence",
            "width": self.width,
            "height": self.height,
            "sequence_length": self.sequence_length,
            "agents": self.num_agents,
        }


class MemoryLandmarks(EnvConfig):
    """Memory task with landmarks."""

    name: str = "memory/landmarks"
    num_landmarks: int = 4

    def get_map_builder(self) -> Dict[str, Any]:
        return {
            "_target_": "mettagrid.room.memory_landmarks.MemoryLandmarks",
            "width": self.width,
            "height": self.height,
            "num_landmarks": self.num_landmarks,
            "agents": self.num_agents,
        }


# Object use environments
class ObjectUseArmory(EnvConfig):
    """Armory object use task."""

    name: str = "objectuse/armory"

    def get_map_builder(self) -> Dict[str, Any]:
        return {
            "_target_": "mettagrid.room.object_use.ObjectUseRoom",
            "width": 25,
            "height": 25,
            "object_type": "armory",
            "agents": self.num_agents,
            "spawn_distance": 10,
        }


class ObjectUseGenerator(EnvConfig):
    """Generator object use task."""

    name: str = "objectuse/generator"

    def get_map_builder(self) -> Dict[str, Any]:
        return {
            "_target_": "mettagrid.room.object_use.ObjectUseRoom",
            "width": 25,
            "height": 25,
            "object_type": "generator",
            "agents": self.num_agents,
            "spawn_distance": 10,
        }


# Multi-agent environments
class MultiAgentCooperation(EnvConfig):
    """Multi-agent cooperation task."""

    name: str = "multiagent/cooperation"
    num_agents: int = 2

    def get_map_builder(self) -> Dict[str, Any]:
        return {
            "_target_": "mettagrid.room.cooperation.CooperationRoom",
            "width": 40,
            "height": 40,
            "agents": self.num_agents,
            "num_targets": 4,
            "require_cooperation": True,
        }


class MultiAgentCompetition(EnvConfig):
    """Multi-agent competition task."""

    name: str = "multiagent/competition"
    num_agents: int = 4

    def get_map_builder(self) -> Dict[str, Any]:
        return {
            "_target_": "mettagrid.room.competition.CompetitionRoom",
            "width": 50,
            "height": 50,
            "agents": self.num_agents,
            "resources": {"altar": 10, "mine": 5},
            "respawn_resources": True,
        }


# Environment suites for evaluation
def navigation_eval_suite() -> List[EnvConfig]:
    """Get navigation evaluation environments."""
    return [
        NavigationEmpty(name="navigation/empty_simple"),
        NavigationWalls(name="navigation/walls_sparse"),
        NavigationWalls(name="navigation/walls_dense", width=40, height=40),
        NavigationObstacles(name="navigation/obstacles_easy", difficulty=0),
        NavigationObstacles(name="navigation/obstacles_medium", difficulty=1),
        NavigationObstacles(name="navigation/obstacles_hard", difficulty=2),
        NavigationMaze(name="navigation/maze_simple", complexity=0.3),
        NavigationMaze(name="navigation/maze_complex", complexity=0.7),
    ]


def memory_eval_suite() -> List[EnvConfig]:
    """Get memory evaluation environments."""
    return [
        MemorySequence(name="memory/sequence_easy", sequence_length=2),
        MemorySequence(name="memory/sequence_medium", sequence_length=4),
        MemorySequence(name="memory/sequence_hard", sequence_length=6),
        MemoryLandmarks(name="memory/landmarks_few", num_landmarks=3),
        MemoryLandmarks(name="memory/landmarks_many", num_landmarks=8),
    ]


def objectuse_eval_suite() -> List[EnvConfig]:
    """Get object use evaluation environments."""
    return [
        ObjectUseArmory(name="objectuse/armory"),
        ObjectUseGenerator(name="objectuse/generator"),
        ObjectUseArmory(name="objectuse/armory_far", width=40, height=40),
        ObjectUseGenerator(name="objectuse/generator_far", width=40, height=40),
    ]


def all_eval_suite() -> List[EnvConfig]:
    """Get all evaluation environments."""
    return navigation_eval_suite() + memory_eval_suite() + objectuse_eval_suite()
