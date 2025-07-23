"""Python-based curriculum definitions.

This module contains curriculum configurations as Python code,
allowing for better type safety and easier programmatic access.
"""

from metta.mettagrid.curriculum import CurriculumConfig
from metta.mettagrid.curriculum.curriculum_algorithm import DiscreteRandomHypers
from metta.mettagrid.curriculum.learning_progress import LearningProgressHypers
from metta.mettagrid.curriculum.prioritize_regressed import PrioritizeRegressedHypers


# Arena curricula
ARENA_TASKS = [
    "/env/mettagrid/arena/basic",
    "/env/mettagrid/arena/basic_easy",
    "/env/mettagrid/arena/basic_easy_shaped",
    "/env/mettagrid/arena/basic_poor",
    "/env/mettagrid/arena/combat",
    "/env/mettagrid/arena/combat_easy",
    "/env/mettagrid/arena/combat_easy_shaped",
    "/env/mettagrid/arena/combat_poor",
    "/env/mettagrid/arena/advanced",
    "/env/mettagrid/arena/advanced_easy",
    "/env/mettagrid/arena/advanced_easy_shaped",
    "/env/mettagrid/arena/advanced_poor",
    "/env/mettagrid/arena/tag",
    "/env/mettagrid/arena/tag_easy",
    "/env/mettagrid/arena/tag_easy_shaped",
]

arena_random = CurriculumConfig(
    name="arena_random",
    algorithm=DiscreteRandomHypers(),
    env_paths=ARENA_TASKS,
)

arena_learning_progress = CurriculumConfig(
    name="arena_learning_progress",
    algorithm=LearningProgressHypers(),
    env_paths=ARENA_TASKS,
)

arena_prioritize_regressed = CurriculumConfig(
    name="arena_prioritize_regressed",
    algorithm=PrioritizeRegressedHypers(),
    env_paths=ARENA_TASKS,
)


# Navigation curricula
from metta.mettagrid.curriculum.curriculum_config import ParameterRange

navigation_bucketed = CurriculumConfig(
    name="navigation_bucketed",
    algorithm=DiscreteRandomHypers(),
    env_paths=["/env/mettagrid/navigation/training/terrain_from_numpy"],
    parameters={
        "game.map_builder.room.dir": ParameterRange(
            values=[
                "terrain_maps_nohearts",
                "varied_terrain/balanced_large",
                "varied_terrain/balanced_medium",
                "varied_terrain/balanced_small",
                "varied_terrain/sparse_large",
                "varied_terrain/sparse_medium",
                "varied_terrain/sparse_small",
                "varied_terrain/dense_large",
                "varied_terrain/dense_medium",
                "varied_terrain/dense_small",
                "varied_terrain/maze_large",
                "varied_terrain/maze_medium",
                "varied_terrain/maze_small",
                "varied_terrain/cylinder-world_large",
                "varied_terrain/cylinder-world_medium",
                "varied_terrain/cylinder-world_small",
            ]
        ),
        "game.map_builder.room.objects.altar": ParameterRange(
            range=(2, 18),
            bins=4,
        ),
    },
    env_overrides={"sampling": 0},
)

navigation_learning_progress = CurriculumConfig(
    name="navigation_learning_progress",
    algorithm=LearningProgressHypers(),
    env_paths=navigation_bucketed.env_paths,
    parameters=navigation_bucketed.parameters,
    env_overrides=navigation_bucketed.env_overrides,
)

navigation_prioritize_regressed = CurriculumConfig(
    name="navigation_prioritize_regressed",
    algorithm=PrioritizeRegressedHypers(),
    env_paths=navigation_bucketed.env_paths,
    parameters=navigation_bucketed.parameters,
    env_overrides=navigation_bucketed.env_overrides,
)


# Register all curricula in the store
from metta.mettagrid.curriculum.curriculum_store import curriculum_store

curriculum_store.register("arena_random", arena_random)
curriculum_store.register("arena_learning_progress", arena_learning_progress)
curriculum_store.register("arena_prioritize_regressed", arena_prioritize_regressed)
curriculum_store.register("navigation_bucketed", navigation_bucketed)
curriculum_store.register("navigation_learning_progress", navigation_learning_progress)
curriculum_store.register("navigation_prioritize_regressed", navigation_prioritize_regressed)

# Export all curriculum configs for easy access
CURRICULUM_CONFIGS = {
    "arena/random": arena_random,
    "arena/learning_progress": arena_learning_progress,
    "arena/prioritize_regressed": arena_prioritize_regressed,
    "navigation/bucketed": navigation_bucketed,
    "navigation/learning_progress": navigation_learning_progress,
    "navigation/prioritize_regressed": navigation_prioritize_regressed,
}