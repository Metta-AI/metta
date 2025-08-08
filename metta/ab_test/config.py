"""
Configuration classes for A/B testing framework.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class ABVariant:
    """Represents a single variant in an A/B test."""

    name: str
    description: str
    overrides: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate variant configuration."""
        if not self.name:
            raise ValueError("Variant name cannot be empty")
        if not self.description:
            raise ValueError("Variant description cannot be empty")


@dataclass
class ABExperiment:
    """Represents an A/B test experiment configuration."""

    name: str
    description: str
    date: Optional[str] = None
    variants: Dict[str, ABVariant] = field(default_factory=dict)
    runs_per_variant: int = 5
    base_config: Dict[str, Any] = field(default_factory=dict)
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None

    def __post_init__(self):
        """Validate experiment configuration."""
        if not self.name:
            raise ValueError("Experiment name cannot be empty")
        if not self.description:
            raise ValueError("Experiment description cannot be empty")
        if len(self.variants) < 2:
            raise ValueError("Experiment must have at least 2 variants")
        if self.runs_per_variant < 1:
            raise ValueError("runs_per_variant must be at least 1")

        # Auto-generate date if not provided
        if self.date is None:
            self.date = datetime.now().strftime("%Y-%m-%d")

        # Auto-generate wandb project name if not provided
        if self.wandb_project is None:
            self.wandb_project = f"ab_test_{self.name}_{self.date}"

    def add_variant(self, variant: ABVariant) -> None:
        """Add a variant to the experiment."""
        self.variants[variant.name] = variant

    def get_variant(self, name: str) -> ABVariant:
        """Get a variant by name."""
        if name not in self.variants:
            raise ValueError(f"Variant '{name}' not found in experiment")
        return self.variants[name]


@dataclass
class ABTestConfig:
    """Configuration for running A/B tests."""

    experiment: ABExperiment
    output_dir: str = "ab_test_results"
    parallel_runs: bool = False
    max_parallel_runs: int = 4
    retry_failed_runs: bool = True
    max_retries: int = 3

    def __post_init__(self):
        """Validate configuration."""
        if self.max_parallel_runs < 1:
            raise ValueError("max_parallel_runs must be at least 1")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")


class ExperimentBuilder:
    """Helper class for building experiments with a fluent interface."""

    def __init__(self, name: str, description: str):
        self.experiment = ABExperiment(name=name, description=description)

    def add_variant(self, name: str, description: str, **overrides) -> "ExperimentBuilder":
        """Add a variant to the experiment."""
        variant = ABVariant(name=name, description=description, overrides=overrides)
        self.experiment.add_variant(variant)
        return self

    def set_runs_per_variant(self, runs: int) -> "ExperimentBuilder":
        """Set the number of runs per variant."""
        self.experiment.runs_per_variant = runs
        return self

    def set_base_config(self, **config) -> "ExperimentBuilder":
        """Set the base configuration."""
        self.experiment.base_config.update(config)
        return self

    def set_wandb_config(self, project: Optional[str] = None, entity: Optional[str] = None) -> "ExperimentBuilder":
        """Set WandB configuration."""
        if project:
            self.experiment.wandb_project = project
        if entity:
            self.experiment.wandb_entity = entity
        return self

    def build(self) -> ABExperiment:
        """Build and return the experiment."""
        return self.experiment


def create_experiment(name: str, description: str) -> ExperimentBuilder:
    """Create a new experiment builder."""
    return ExperimentBuilder(name, description)
