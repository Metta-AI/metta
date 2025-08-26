# Axiom: Hyperlightweight Pipeline DSL

Axiom is a minimal DSL for composing pipelines with built-in support for A/B testing, ablation studies, and experiment tracking.

## Core Concepts

### 1. Pipeline Composition
```python
pipeline = (
    Pipeline()
    .stage("process", process_func)    # Deterministic computation
    .io("fetch", fetch_func)           # External I/O operation
    .stage("analyze", analyze_func)    # More computation
)
result = pipeline.run(Ctx())
```

### 2. A/B Testing with Expose/Provide
```python
# Base pipeline with variation points
base = (
    Pipeline()
    .stage("load", load_data)
    .expose_join("optimizer")     # Variation point!
    .stage("train", train_model)
)

# Variant A: Adam optimizer
variant_a = base.provide_join("optimizer", 
    Pipeline().stage("adam", setup_adam))

# Variant B: SGD optimizer  
variant_b = base.provide_join("optimizer",
    Pipeline().stage("sgd", setup_sgd))
```

### 3. The Canonical Pattern

Following the `sequential_sweep` pattern, all pipelines should:

1. **Pipeline Builder**: Accepts factories (not configs)
2. **Tool Wrapper**: Thin orchestration layer
3. **Factory Functions**: Create configured instances

```python
class MyPipelineBuilder:
    def __init__(self, tool_factory: Callable):
        self.tool_factory = tool_factory  # Factory, not config!
    
    def build_pipeline(self) -> Pipeline:
        return (
            Pipeline()
            .io("external", self._external_op)
            .stage("compute", self._compute)
        )

class MyTool(Tool):
    def invoke(self, args, overrides):
        builder = MyPipelineBuilder(factory)
        pipeline = builder.build_pipeline()
        return pipeline.run(Ctx())
```

## Examples

See the `examples/` directory:
- `axiom_basic_pipeline.py` - Simplest pipeline
- `axiom_ab_test.py` - A/B testing with expose/provide
- `axiom_manifest_diff.py` - Manifest comparison for ablations

## Key Files

- `core.py` - Pipeline DSL implementation
- `training_guards.py` - Decorators for cross-cutting concerns
- `manifest.py` - Experiment tracking and diffing
- `sequential_sweep.py` - The canonical pattern example
- `train_and_eval.py` - Train+eval pipeline example

## Design Principles

1. **Minimal**: No heavy frameworks, just pipelines
2. **Explicit**: Clear I/O vs computation distinction
3. **Composable**: Pipelines can be combined and extended
4. **Trackable**: Automatic manifest generation
5. **Testable**: Easy to test individual stages

## Guards for Cross-Cutting Concerns

```python
@wandb_context(master_only=True)
def _train(self, state):
    # Automatically wrapped in WandB context
    pass

@platform_specific("Darwin")
def _adjust_for_mac(self, state):
    # Only runs on macOS
    pass
```

## The Power of Expose/Provide

The expose/provide pattern enables:
- **A/B Testing**: Compare different implementations
- **Ablation Studies**: Remove components to measure impact
- **Feature Flags**: Toggle features on/off
- **Progressive Rollout**: Gradually enable new code paths

This is the KEY to safe experimentation in production ML systems.