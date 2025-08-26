# AxiomExperiment Design: Spec-Driven Experimental Control

## Overview

The AxiomExperiment system provides **spec-driven, pipeline-agnostic experimental control** for ML/RL experiments. It ensures reproducibility through complete configuration capture, deterministic execution, and auditable manifests.

## Core Components

### 1. ExperimentSpec
The complete configuration for an experiment. This is the **single source of truth** that contains everything needed to reproduce an experiment:

```python
spec = TrainingExperimentSpec(
    name="ppo_baseline",
    pipeline_config={...},      # Config passed to Tool
    exposed_joins=["trainer.optimizer"],  # Variation points
    join_configs={...},          # Configs for sub-pipelines
    controls=AxiomControls(...), # Seeds, determinism
)
```

### 2. AxiomExperiment
The harness that:
- Consumes an ExperimentSpec
- Builds pipelines from Tool configurations
- Manages join points for customization
- Enforces experimental controls
- Generates manifests for every run

### 3. Pipeline Integration
Works with existing Tool-based pipelines:

```python
# Tool (e.g., TrainJobPipeline) receives config from spec
tool = TrainJobPipeline(**spec.pipeline_config)

# Tool produces a Pipeline
pipeline = tool.get_pipeline()

# AxiomExperiment manages the pipeline execution
exp = AxiomExperimentV2(spec)
result = exp.run()
```

## Key Design Principles

### 1. Spec Contains Everything
The ExperimentSpec must contain **all** information needed to reproduce an experiment:
- Pipeline configuration (what Tool, what settings)
- Control variables (seeds, determinism flags)
- Join configurations (how to build sub-pipelines)
- Metadata (name, description, tags)

### 2. Tools Build Pipelines from Configs
Tools like `TrainJobPipeline` are **configured** by the spec, then produce pipelines:
```python
TrainConfig → TrainTool → Pipeline → Execution
```

### 3. Hierarchical Join Points
Experiments can expose join points at multiple levels:
- `trainer` - Replace entire training pipeline
- `trainer.optimizer` - Replace just optimizer
- `trainer.advantage` - Replace advantage computation
- `trainer.rollout` - Replace rollout collection

### 4. Variation Through Diffs
Rather than explicitly marking what can vary, we rely on **manifest diffs** to show what changed between experiments. This works better with config inheritance and composition.

## Usage Patterns

### Basic Training Experiment
```python
# Define spec with complete configuration
spec = TrainingExperimentSpec(
    name="baseline",
    pipeline_config={
        "trainer": {"total_timesteps": 100000},
        "optimizer": {"learning_rate": 3e-4},
    },
    exposed_joins=["trainer.optimizer"],
)

# Create and run experiment
exp = AxiomExperiment(spec)
exp.prepare()  # Freeze seeds, capture environment
baseline = exp.run("baseline")
```

### A/B Comparison
```python
# Run with different optimizer
adam_result = exp.run(
    "adam_variant",
    override_joins={"trainer.optimizer": "adam"}
)

sgd_result = exp.run(
    "sgd_variant", 
    override_joins={"trainer.optimizer": "sgd"}
)

# Compare manifests
print(exp.diff(adam_result, sgd_result))
```

### Single-Factor Enforcement
```python
spec = ComparisonExperimentSpec(
    name="strict_ablation",
    controls=AxiomControls(single_factor_enforce=True),
    # ...
)

# This ensures only one thing changes between runs
exp.run_comparison(
    baseline_joins={"trainer.optimizer": "adam"},
    variants={"sgd": {"trainer.optimizer": "sgd"}}
)  # OK - only optimizer changes

exp.run_comparison(
    baseline_joins={"trainer.optimizer": "adam"},
    variants={
        "complex": {
            "trainer.optimizer": "sgd",
            "trainer.advantage": "vtrace",  # ERROR: Two changes!
        }
    }
)
```

## Implementation Bridge

### From Spec to Pipeline

1. **Spec defines configuration**
   ```python
   spec.pipeline_config = {"trainer": {...}, "optimizer": {...}}
   ```

2. **Factory creates Tool from config**
   ```python
   tool = TrainJobPipeline(**config)
   ```

3. **Tool produces Pipeline**
   ```python
   pipeline = tool.get_pipeline()
   ```

4. **AxiomExperiment manages execution**
   ```python
   exp = AxiomExperiment(spec)
   result = exp.run()
   ```

### Custom Pipeline Factories

You can provide custom factories to create pipelines from specs:

```python
def my_pipeline_factory(config: dict) -> Pipeline:
    # Create your tool
    tool = MyCustomTool(**config)
    # Get pipeline
    return tool.get_pipeline()

exp = AxiomExperiment(spec, pipeline_factory=my_pipeline_factory)
```

## Manifest Structure

Every run produces a manifest with:

```json
{
  "experiment": "name",
  "spec": {...},           // Complete ExperimentSpec
  "controls": {            // Frozen control variables
    "seed": 42,
    "enforce_determinism": true
  },
  "fingerprints": {        // Input hashes
    "dataset": "hash123",
    "env": "hash456"
  },
  "environment": {         // Runtime environment
    "platform": "darwin",
    "torch": "2.0.1",
    "cuda": false
  },
  "code": {               // Code version
    "git_commit": "abc123",
    "has_uncommitted_changes": false
  },
  "joins": {              // Join point usage
    "exposed": ["trainer.optimizer"],
    "provided": ["trainer.optimizer"],
    "implementations": {"trainer.optimizer": "adam"}
  }
}
```

## Benefits

1. **Reproducibility**: Spec + manifest = complete experiment record
2. **Flexibility**: Override joins without changing specs
3. **Auditability**: Every decision is tracked in manifests
4. **Composability**: Build complex experiments from simple specs
5. **Type Safety**: Pydantic validation catches errors early

## Summary

The AxiomExperiment system provides a **spec-driven wrapper** around pipeline-based experiments. It doesn't replace Tools or Pipelines - it **orchestrates** them with strong experimental controls.

The key insight: **ExperimentSpec is configuration**, Tools are factories, Pipelines are execution. AxiomExperiment ties them together with reproducibility guarantees.