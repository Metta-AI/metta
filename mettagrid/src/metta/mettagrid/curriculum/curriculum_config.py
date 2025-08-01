# metta/mettagrid/curriculum/curriculum_config.py
from typing import Any, Dict, List, Optional, Tuple

from omegaconf import DictConfig, OmegaConf
from pydantic import Field, field_validator, model_validator

from metta.common.util.config import config_from_path
from metta.common.util.typed_config import BaseModelWithForbidExtra
from metta.mettagrid.curriculum.curriculum import Curriculum
from metta.mettagrid.curriculum.curriculum_algorithm import (
    CurriculumAlgorithmHypers,
    DiscreteRandomHypers,
)
from metta.mettagrid.curriculum.curriculum_builder import task_set
from metta.mettagrid.curriculum.learning_progress import LearningProgressHypers
from metta.mettagrid.curriculum.prioritize_regressed import PrioritizeRegressedHypers
from metta.mettagrid.curriculum.progressive import ProgressiveHypers, SimpleProgressiveHypers


class ParameterRange(BaseModelWithForbidExtra):
    """Specification for a parameter range in curriculum generation.

    Can specify either:
    - A list of discrete values
    - A continuous range with number of bins
    """

    values: Optional[List[Any]] = None
    range: Optional[Tuple[float, float]] = None
    bins: Optional[int] = None

    @model_validator(mode="after")
    def validate_range_spec(self):
        """Ensure exactly one range specification type is provided."""
        has_values = self.values is not None
        has_range = self.range is not None

        if has_values and has_range:
            raise ValueError("Cannot specify both 'values' and 'range'")

        if not has_values and not has_range:
            raise ValueError("Must specify either 'values' or 'range'")

        # bins is now optional for ranges (missing bins = continuous range)
        # but if specified, must be >= 2
        if has_range and self.bins is not None and self.bins < 2:
            raise ValueError("'bins' must be >= 2 when specified. For continuous ranges, omit the 'bins' field.")

        return self


class CurriculumConfig(BaseModelWithForbidExtra):
    """Configuration for creating a Curriculum.

    This configuration can create either:
    1. A flat task set (when env_paths is provided)
    2. A hierarchical task tree (when children are provided)

    For flat task sets:
    - Provide env_paths list for one or more environment configs
    - Add parameters to generate a parameter grid from all env_paths

    Parameters create a cartesian product with ALL provided configs.
    """

    name: str

    # List of environment config paths (can be a single item)
    env_paths: Optional[List[str]] = None

    # Parameter ranges for grid generation (applied to all configs)
    parameters: Optional[Dict[str, ParameterRange]] = None

    # Child task sets for hierarchical structure
    children: Optional[List["CurriculumConfig"]] = None

    # Curriculum algorithm configuration - stores the actual hypers object
    algorithm: Optional[CurriculumAlgorithmHypers] = None

    # Environment configuration overrides applied to all tasks
    env_overrides: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("children", mode="before")
    def parse_children(cls, v):
        """Convert child specifications to CurriculumConfig objects."""
        if v is None:
            return None

        result = []
        for child in v:
            if isinstance(child, str):
                # Load config from path
                child_cfg = config_from_path(child, None)
                child_dict = OmegaConf.to_container(child_cfg, resolve=True)
                child_config = cls.model_validate(child_dict)
                result.append(child_config)
            elif isinstance(child, dict):
                # Parse dict as CurriculumConfig
                child_config = cls.model_validate(child)
                result.append(child_config)
            elif isinstance(child, cls):
                # Already a CurriculumConfig
                result.append(child)
            else:
                raise ValueError(f"Child must be a string path, dict, or CurriculumConfig. Got: {type(child)}")

        return result

    @field_validator("algorithm", mode="before")
    def parse_algorithm(cls, v):
        """Convert algorithm specification to actual hypers object."""
        if v is None:
            return None

        # If it's already a hypers object, return it
        if isinstance(v, CurriculumAlgorithmHypers):
            return v

        # Handle string shorthand
        if isinstance(v, str):
            algorithm_type = v
            params = {}
        # Handle dict with type and params
        elif isinstance(v, dict) and "type" in v:
            algorithm_type = v["type"]
            params = v.get("params", {})
        else:
            raise ValueError(
                "Algorithm must be a string type, a dict with 'type' field, "
                f"or a CurriculumAlgorithmHypers instance. Got: {type(v)}"
            )

        # Create the appropriate hypers object
        algorithm_map = {
            "random": DiscreteRandomHypers,
            "discrete_random": DiscreteRandomHypers,  # alias
            "learning_progress": LearningProgressHypers,
            "prioritize_regressed": PrioritizeRegressedHypers,
            "progressive": ProgressiveHypers,
            "simple_progressive": SimpleProgressiveHypers,
        }

        hypers_class = algorithm_map.get(algorithm_type)
        if hypers_class is None:
            raise ValueError(
                f"Unknown algorithm type: {algorithm_type}. Valid types are: {', '.join(algorithm_map.keys())}"
            )

        return hypers_class(**params)

    @model_validator(mode="after")
    def validate_config_type(self):
        """Ensure configuration is either a task set or hierarchical."""
        is_task_set = self.env_paths is not None
        is_hierarchical = self.children is not None

        if is_task_set and is_hierarchical:
            raise ValueError(
                "Cannot specify both env_paths and children. Use either a flat task set or a hierarchical structure."
            )

        if not is_task_set and not is_hierarchical:
            raise ValueError("Must specify either env_paths or children.")

        return self

    def to_serializable_dict(self) -> dict:
        """Convert to a dict suitable for serialization, handling algorithm field specially."""
        data = self.model_dump(exclude={"algorithm"})

        # Handle algorithm serialization
        if self.algorithm is not None:
            # Use the algorithm_type() method for clean string representation
            data["algorithm"] = self.algorithm.algorithm_type()

        return data

    def create(self, env_overrides: DictConfig | None = None) -> Curriculum:
        """Create the Curriculum from this configuration.

        Args:
            env_overrides: Additional environment overrides to apply
        """
        # Merge external overrides with config overrides
        if env_overrides:
            merged_overrides = OmegaConf.merge(
                OmegaConf.create(self.env_overrides) if self.env_overrides else OmegaConf.create({}), env_overrides
            )
        else:
            merged_overrides = OmegaConf.create(self.env_overrides) if self.env_overrides else None

        # Route to appropriate creation method
        if self.children:
            return self._create_hierarchical()
        else:
            return self._create_task_set(merged_overrides)

    def _create_task_set(self, env_overrides_cfg) -> Curriculum:
        """Create a flat task set using the unified task_set API."""
        # Collect all env configs
        env_configs = []
        for path in self.env_paths:
            env_configs.append((self._name_from_path(path), path))

        # Convert parameter ranges if provided
        parameter_ranges = None
        if self.parameters:
            parameter_ranges = {}
            for param_name, param_range in self.parameters.items():
                if param_range.values is not None:
                    parameter_ranges[param_name] = {"values": param_range.values}
                else:
                    parameter_ranges[param_name] = {"range": list(param_range.range), "bins": param_range.bins}

        # Load actual configs from paths
        loaded_env_configs = [(name, config_from_path(path, None)) for name, path in env_configs]

        # Use the unified task_set API
        # parameter_ranges will create a cartesian product with ALL env_configs
        return task_set(
            name=self.name,
            env_configs=loaded_env_configs,
            env_overrides=env_overrides_cfg,
            curriculum_hypers=self.algorithm,
            parameter_ranges=parameter_ranges,
        )

    def _create_hierarchical(self) -> Curriculum:
        """Create a hierarchical task tree."""
        # Children are now always CurriculumConfig objects thanks to the validator
        child_trees = [child.create() for child in self.children]

        # Default to random algorithm if not specified
        algorithm_hypers = self.algorithm or DiscreteRandomHypers()

        algorithm = algorithm_hypers.create(len(child_trees))
        return Curriculum(self.name, algorithm, child_trees)

    @staticmethod
    def _name_from_path(path: str) -> str:
        """Extract a short name from a config path.

        Examples:
        - "/env/mettagrid/arena/basic" -> "basic"
        - "/env/mettagrid/navigation/training/small.yaml" -> "small"
        - "configs/env/combat" -> "combat"
        """
        # Remove file extension if present
        path_without_ext = path.rsplit(".", 1)[0] if "." in path else path

        # Get the last component of the path
        return path_without_ext.split("/")[-1]


# Enable recursive type references
CurriculumConfig.model_rebuild()
