"""ExperimentSpec: Complete configuration for reproducible experiments."""

from typing import Any, Callable, Literal, Optional

from pydantic import BaseModel, Field


class AxiomControls(BaseModel):
    """Control variables for deterministic experiment execution."""
    
    seed: int = Field(default=42, description="Master seed for all RNGs")
    enforce_determinism: bool = Field(default=True, description="Enable CUDA/cuDNN determinism")
    single_factor_enforce: bool = Field(default=False, description="Reject multi-change variants")
    capture_env_vars: list[str] = Field(
        default_factory=lambda: [
            "CUDA_VISIBLE_DEVICES",
            "CUBLAS_WORKSPACE_CONFIG",
            "OMP_NUM_THREADS",
            "PYTHONHASHSEED",
        ],
        description="Environment variables to capture",
    )


class ExperimentSpec(BaseModel):
    """Complete specification for an experiment.
    
    The ExperimentSpec contains ALL configuration needed to:
    1. Build the main pipeline (via tool configs)
    2. Configure sub-pipelines 
    3. Set control variables (seeds, determinism)
    4. Define variation points (which joins to expose)
    
    This is the single source of truth for experiment configuration.
    Everything needed to reproduce an experiment should be in the spec.
    """
    
    # Experiment identity
    name: str = Field(description="Unique experiment name")
    description: str = Field(default="", description="Human-readable description")
    tags: list[str] = Field(default_factory=list, description="Tags for organization")
    
    # Control variables
    controls: AxiomControls = Field(
        default_factory=AxiomControls,
        description="Seeds, determinism settings, etc."
    )
    
    # Main pipeline configuration
    # This could be TrainJobPipeline config, or any Tool config
    pipeline_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration for main pipeline tool"
    )
    pipeline_type: str = Field(
        default="training",
        description="Type of pipeline to create (training, sweep, eval, etc.)"
    )
    
    # Sub-pipeline configurations
    # Maps join point names to their configurations
    join_configs: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Configurations for join point implementations"
    )
    
    # Exposed join points
    exposed_joins: list[str] = Field(
        default_factory=list,
        description="Join points to expose for variation"
    )
    
    # Provided join implementations
    # Maps join names to pipeline factory names
    provided_joins: dict[str, str] = Field(
        default_factory=dict,
        description="Which implementations to use for joins"
    )
    
    # Runtime settings
    run_dir: str = Field(
        default="./experiments",
        description="Directory for experiment artifacts"
    )
    
    # Fingerprinting callables (serialized as strings for now)
    dataset_hasher: Optional[str] = Field(
        default=None,
        description="Name of dataset hashing function"
    )
    env_hasher: Optional[str] = Field(
        default=None,
        description="Name of environment hashing function"  
    )
    
    class Config:
        """Pydantic configuration."""
        extra = "forbid"  # Fail on unknown fields


class TrainingExperimentSpec(ExperimentSpec):
    """Specification for training experiments."""
    
    pipeline_type: Literal["training"] = Field(default="training")
    
    # Training-specific exposed joins
    exposed_joins: list[str] = Field(
        default_factory=lambda: [
            "trainer",
            "trainer.optimizer", 
            "trainer.advantage",
            "trainer.rollout"
        ]
    )
    
    # Example training pipeline config
    pipeline_config: dict[str, Any] = Field(
        default_factory=lambda: {
            "run": "training_experiment",
            "trainer": {
                "total_timesteps": 100000,
                "num_workers": 4,
                "batch_size": 64,
            },
            "optimizer": {
                "learning_rate": 1e-3,
                "betas": [0.9, 0.999],
            },
            "wandb": {
                "project": "experiments",
                "entity": "my-team",
            }
        }
    )


class SweepExperimentSpec(ExperimentSpec):
    """Specification for hyperparameter sweep experiments."""
    
    pipeline_type: Literal["sweep"] = Field(default="sweep")
    
    # Sweep-specific configuration
    num_trials: int = Field(default=10, description="Number of sweep trials")
    
    search_space: dict[str, Any] = Field(
        default_factory=dict,
        description="Hyperparameter search space"
    )
    
    metric: str = Field(
        default="eval_score",
        description="Metric to optimize"
    )
    
    # Sweep typically exposes the training pipeline
    exposed_joins: list[str] = Field(
        default_factory=lambda: ["trial_pipeline"]
    )


class ComparisonExperimentSpec(ExperimentSpec):
    """Specification for A/B comparison experiments."""
    
    pipeline_type: Literal["comparison"] = Field(default="comparison")
    
    # Comparison settings
    baseline_spec: dict[str, Any] = Field(
        default_factory=dict,
        description="Baseline configuration"
    )
    
    variants: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Variant configurations to compare"
    )
    
    # Single factor enforcement
    controls: AxiomControls = Field(
        default_factory=lambda: AxiomControls(single_factor_enforce=True)
    )


def load_experiment_spec(path: str) -> ExperimentSpec:
    """Load experiment spec from file.
    
    Args:
        path: Path to spec file (JSON or YAML)
    
    Returns:
        Loaded ExperimentSpec
    """
    import json
    from pathlib import Path
    
    spec_path = Path(path)
    
    if spec_path.suffix == ".json":
        with open(spec_path) as f:
            data = json.load(f)
    elif spec_path.suffix in [".yaml", ".yml"]:
        import yaml
        with open(spec_path) as f:
            data = yaml.safe_load(f)
    else:
        raise ValueError(f"Unknown spec format: {spec_path.suffix}")
    
    # Determine spec type from data
    pipeline_type = data.get("pipeline_type", "training")
    
    if pipeline_type == "training":
        return TrainingExperimentSpec(**data)
    elif pipeline_type == "sweep":
        return SweepExperimentSpec(**data)
    elif pipeline_type == "comparison":
        return ComparisonExperimentSpec(**data)
    else:
        return ExperimentSpec(**data)


def save_experiment_spec(spec: ExperimentSpec, path: str) -> None:
    """Save experiment spec to file.
    
    Args:
        spec: ExperimentSpec to save
        path: Path to save to
    """
    import json
    from pathlib import Path
    
    spec_path = Path(path)
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    
    if spec_path.suffix == ".json":
        with open(spec_path, "w") as f:
            json.dump(spec.model_dump(), f, indent=2)
    elif spec_path.suffix in [".yaml", ".yml"]:
        import yaml
        with open(spec_path, "w") as f:
            yaml.dump(spec.model_dump(), f, default_flow_style=False)
    else:
        raise ValueError(f"Unknown spec format: {spec_path.suffix}")