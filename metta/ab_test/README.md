# A/B Testing Framework

A Python-based framework for defining and running A/B comparison experiments in Metta.

## Overview

This framework allows you to define A/B test experiments as Python classes instead of YAML files, providing more flexibility and power. Each experiment can compare multiple variants across a single axis of variation.

## Quick Start

### 1. Define an Experiment

Create a Python file defining your experiment:

```python
# my_experiment.py
from metta.ab_test.config import create_experiment

def create_my_experiment():
    return (
        create_experiment(
            name="my_experiment",
            description="My A/B test experiment"
        )
        .add_variant(
            name="variant_a",
            description="First variant",
            trainer__learning_rate=0.001,
            run="variant_a"
        )
        .add_variant(
            name="variant_b",
            description="Second variant",
            trainer__learning_rate=0.0001,
            run="variant_b"
        )
        .set_runs_per_variant(5)
        .set_base_config(
            defaults=["/common", "/agent/fast", "/trainer/trainer"],
            total_timesteps=1_000_000_000
        )
        .build()
    )

# Alternative: define as a variable
experiment = create_my_experiment()
```

### 2. Run the Experiment

```bash
# Run the experiment
python tools/run_ab_test.py my_experiment.py

# Dry run to see configuration
python tools/run_ab_test.py my_experiment.py --dry-run

# Run with custom output directory
python tools/run_ab_test.py my_experiment.py --output-dir results/exp1
```

## Example: Curriculum Comparison

Here's a complete example comparing learning progress vs prioritized regressed curriculum:

```python
# curriculum_comparison.py
from metta.ab_test.config import create_experiment

def create_curriculum_comparison_experiment():
    return (
        create_experiment(
            name="curriculum_comparison",
            description="Learning Progress vs Prioritized Regressed Curriculum"
        )
        .add_variant(
            name="learning_progress",
            description="Learning progress curriculum for navigation tasks",
            trainer__curriculum="/env/mettagrid/curriculum/navigation/learning_progress",
            run="curriculum_lp"
        )
        .add_variant(
            name="prioritized_regressed",
            description="Prioritized regressed curriculum for navigation tasks",
            trainer__curriculum="/env/mettagrid/curriculum/nav_memory_sequence",
            run="curriculum_pr"
        )
        .set_runs_per_variant(5)
        .set_base_config(
            defaults=[
                "/common",
                "/agent/fast",
                "/trainer/trainer",
                "/sim/all"
            ],
            total_timesteps=1_000_000_000,
            trainer__num_workers=4,
            trainer__simulation__evaluate_interval=100
        )
        .set_wandb_config(
            entity="metta-research"
        )
        .build()
    )

# Run with: python tools/run_ab_test.py curriculum_comparison.py
```

## Configuration Options

### Experiment Configuration

- `name`: Unique experiment name
- `description`: Human-readable description
- `date`: Experiment date (auto-generated if not provided)
- `runs_per_variant`: Number of runs per variant (default: 5)
- `wandb_project`: WandB project name (auto-generated if not provided)
- `wandb_entity`: WandB entity/username

### Variant Configuration

- `name`: Unique variant name
- `description`: Human-readable description
- `overrides`: Dictionary of configuration overrides
- `tags`: List of tags for WandB runs

### Runner Configuration

- `output_dir`: Directory for experiment results
- `parallel_runs`: Whether to run variants in parallel
- `max_parallel_runs`: Maximum number of parallel runs
- `retry_failed_runs`: Whether to retry failed runs
- `max_retries`: Maximum number of retry attempts

## WandB Integration

Each experiment creates a single WandB project with the naming convention:
`ab_test_{experiment_name}_{date}`

All runs within an experiment are grouped together, with tags indicating the variant. This allows for easy comparison and analysis in the WandB interface.

## Output Structure

```
ab_test_results/
└── experiment_name/
    ├── experiment_config.yaml    # Experiment configuration
    ├── summary.txt              # Experiment summary
    ├── variant_a_run_1/         # Individual run directories
    │   ├── config.yaml
    │   └── ...
    ├── variant_a_run_2/
    └── variant_b_run_1/
```

## CLI Options

```bash
python tools/run_ab_test.py experiment.py [OPTIONS]

Options:
  --dry-run              Show configuration without running
  --output-dir DIR       Output directory (default: ab_test_results)
  --parallel             Run variants in parallel
  --max-parallel N       Maximum parallel runs (default: 4)
  --no-retry             Don't retry failed runs
  --verbose, -v          Enable verbose logging
```

## Advanced Usage

### Custom Experiment Classes

For more complex experiments, you can create custom experiment classes:

```python
from metta.ab_test.config import ABExperiment, ABVariant

class MyCustomExperiment(ABExperiment):
    def __init__(self):
        super().__init__(
            name="custom_experiment",
            description="My custom experiment"
        )

        # Add variants
        self.add_variant(ABVariant(
            name="variant_a",
            description="First variant",
            overrides={"trainer__learning_rate": 0.001}
        ))

        # Set configuration
        self.runs_per_variant = 10
        self.base_config = {
            "defaults": ["/common", "/agent/fast"],
            "total_timesteps": 1_000_000_000
        }
```

### Programmatic Execution

You can also run experiments programmatically:

```python
from metta.ab_test.runner import run_ab_test

experiment = create_my_experiment()
results = run_ab_test(
    experiment,
    output_dir="custom_results",
    parallel_runs=True,
    max_parallel_runs=8
)

# Access results
for variant_name, runs in results.items():
    successful_runs = [r for r in runs if r["success"]]
    print(f"{variant_name}: {len(successful_runs)}/{len(runs)} successful")
```

## Best Practices

1. **Clear Naming**: Use descriptive names for experiments and variants
2. **Single Axis**: Keep experiments focused on a single variable
3. **Adequate Sample Size**: Use at least 5 runs per variant for statistical significance
4. **Documentation**: Include clear descriptions and tags
5. **Base Configuration**: Use the builder pattern to set common configuration
6. **WandB Organization**: Leverage WandB groups and tags for easy analysis

## Integration with Existing Infrastructure

The framework integrates seamlessly with existing Metta infrastructure:

- Uses existing `tools/train.py` for training runs
- Leverages existing configuration system (Hydra/OmegaConf)
- Integrates with existing WandB setup
- Compatible with existing curriculum and agent configurations
