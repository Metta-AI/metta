# Axiom Framework Cookbook 🧪

Welcome to the Axiom Framework Cookbook! This collection of recipes demonstrates practical, real-world usage patterns for the Axiom experiment orchestration framework in reinforcement learning research.

## What is Axiom?

Axiom is a powerful experiment framework that provides:
- **Spec-driven experiments** with perfect reproducibility
- **Safe debugging** through explicit exposure control (`expose=True`)
- **Pipeline composition** for complex training workflows
- **Deep override capability** for surgical debugging
- **Comprehensive tracking** via manifests and run handles

## Cookbook Organization

Each recipe is a standalone Python file with extensive comments, demonstrating a specific use case:

### 🔍 Debugging & Testing
- [`01_debugging_training_divergence.py`](01_debugging_training_divergence.py) - Isolate issues in training implementations
- [`04_progressive_debugging.py`](04_progressive_debugging.py) - Systematically test each component
- [`02_ab_testing_algorithms.py`](02_ab_testing_algorithms.py) - Compare different algorithm implementations

### 📊 Experiment Management
- [`03_reproducible_experiments.py`](03_reproducible_experiments.py) - Save and reproduce exact experiments
- [`09_experiment_resumption.py`](09_experiment_resumption.py) - Smart checkpoint resumption strategies
- [`07_monitoring_with_hooks.py`](07_monitoring_with_hooks.py) - Real-time monitoring and intervention

### 🚀 Advanced Workflows
- [`05_hyperparameter_search.py`](05_hyperparameter_search.py) - Safe hyperparameter exploration
- [`06_multi_stage_pipelines.py`](06_multi_stage_pipelines.py) - Complex multi-stage training
- [`08_composite_experiments.py`](08_composite_experiments.py) - Compose multiple experiment phases
- [`10_curriculum_learning.py`](10_curriculum_learning.py) - Progressive curriculum experiments

## Quick Start

### Basic Experiment Structure

```python
from metta.sweep.axiom.train_and_eval import (
    TrainAndEvalSpec,
    TrainAndEvalExperiment,
    create_quick_experiment
)

# 1. Create a spec (configuration)
spec = create_quick_experiment()

# 2. Create experiment from spec
experiment = TrainAndEvalExperiment(spec)

# 3. Prepare (set seeds, initialize)
experiment.prepare()

# 4. Run the experiment
result = experiment.run()

# 5. Analyze results
manifest = result.manifest()
print(f"Final reward: {manifest['eval_results']['arena/basic']['mean_reward']}")
```

### The Power of Overrides

```python
# Get the pipeline
pipeline = experiment._create_pipeline({})

# Override a specific component (must be exposed!)
pipeline.override("train.compute_advantage", custom_advantage_function)

# Run with the override
result = experiment.run("with_custom_advantage")
```

## Key Concepts

### 1. Specs Define Experiments
Every experiment starts with a spec - a Pydantic model that defines all configuration:
```python
spec = TrainAndEvalSpec(
    name="my_experiment",
    trainer_config=trainer_config,
    eval_configs=eval_configs,
    controls=AxiomControls(seed=42)
)
```

### 2. Exposure Control for Safety
Components must be explicitly exposed to be overrideable:
```python
pipeline.stage("internal", func)  # Cannot be overridden
pipeline.stage("algorithm", func, expose=True)  # Can be overridden
```

### 3. Deep Nested Overrides
Override components at any depth through dot-notation paths:
```python
pipeline.override("train.optimizer.adam_step", custom_adam)
```

### 4. Comprehensive Tracking
Every run produces a manifest with complete information:
```python
manifest = result.manifest()
# Contains: config, results, timestamps, seeds, etc.
```

## Best Practices

1. **Always use specs** - Don't configure experiments imperatively
2. **Expose sparingly** - Only expose what needs to be debuggable
3. **Tag your runs** - Use descriptive tags for different experiment variants
4. **Save important specs** - Use `save_experiment_spec()` for reproducibility
5. **Use factory functions** - Create reusable spec factories for common configs
6. **Test with quick experiments** - Use `create_quick_experiment()` for rapid iteration
7. **Mock expensive operations** - Use overrides to skip expensive steps during debugging

## Common Patterns

### Pattern 1: Debug by Substitution
Replace components with known-good implementations to isolate issues:
```python
pipeline.override("train.compute_loss", reference_implementation)
```

### Pattern 2: Progressive Complexity
Start simple and add complexity through overrides:
```python
# Start with simple rewards
pipeline.override("compute_reward", lambda x: 1.0)
# Later, add shaped rewards
pipeline.override("compute_reward", shaped_reward_function)
```

### Pattern 3: Conditional Execution
Skip expensive steps when debugging:
```python
if debug_mode:
    pipeline.override("train", lambda x: load_checkpoint())
```

## Requirements

The cookbook assumes you have:
- Metta repository set up with `uv` package manager
- Basic familiarity with PyTorch and RL concepts
- Understanding of Pydantic models

## Running the Examples

Each recipe can be run standalone:
```bash
uv run python metta/sweep/axiom/cookbook/01_debugging_training_divergence.py
```

Many recipes include mock implementations for demonstration. To run with real training:
1. Remove the `@mock` decorators
2. Ensure you have GPU access (or set `device="cpu"`)
3. Adjust timesteps and batch sizes for your hardware

## Contributing

When adding new recipes:
1. Follow the numbering scheme (XX_description.py)
2. Include extensive comments explaining each step
3. Provide both mock and real examples where applicable
4. Test that the code actually runs
5. Add common pitfalls and debugging tips

## Support

For issues or questions:
- Check the individual recipe files for detailed explanations
- Review the test files in `tests/sweep/test_axiom_*.py`
- Consult the main Axiom documentation

Happy experimenting! 🚀