# tAXIOM (tiny AXIOM)

```
        _      _____  __  __  _____   ____   __  __ 
       | |_   |  _  | \ \/ / |_   _| / __ \ |  \/  |
       | __|  | |_| |  \  /    | |  | |  | || |\/| |
       | |_   |  _  |  /  \    | |  | |__| || |  | |
        \__|  |_| |_| /_/\_\  |___|  \____/ |_|  |_|
                                                     
```

> **A minimalist DSL for orchestrating RL experiments with explicit data flow and typed contracts**
# Autonomous eXperimentation & Iterative Optimization Manager
## Philosophy

tAXIOM models every experiment as a **composable, idempotent chain of atomic stages** separated by **typed, programmable membranes** (`through`). These membranes enforce explicit contracts—making data flow safe and self-documenting—while exposing hooks for observability, validation, caching, retries, and timing without polluting business logic.

Stages are agnostic to input provenance and combine via clear control-flow primitives (`through`, `do_while`, `do_until`) so simple cases stay simple while complex curricula, sweeps, and meta-learning arise naturally from composition. 

The result is a system that favors:
- **Explicitness over magic** — Every data transformation is visible
- **Reproducibility over ad-hocism** — Experiments can reliably rerun
- **Clean separation** — Experiment flow, optimization, and scheduling are independent

So experiments can reliably rerun, safely evolve, and ultimately optimize themselves.

## Core Concepts

### Pipelines & Stages
```python
from metta.sweep.axiom import Pipeline

pipeline = (
    Pipeline()
    .stage("load", load_data)
    .stage("process", process_data)
    .stage("analyze", analyze_results)
)

result = pipeline.run()
```

### Type Contracts
```python
pipeline = (
    Pipeline()
    .stage("generate", generate_params)
    .through(ParamConfig)  # Enforce output type
    
    .stage("optimize", run_optimization)
    .through(OptResult, input_type=ParamConfig)  # Input and output types
    
    .stage("report", create_report)
    .through(Report, input_type=OptResult)
)
```

### Context Management
```python
from metta.sweep.axiom import Ctx, context_aware

@context_aware
def stateful_stage(ctx: Ctx) -> dict:
    # Access previous outputs
    prev_result = ctx.get_stage_output("previous_stage")
    
    # Store metadata
    ctx.metadata["key"] = "value"
    
    # Access metadata
    value = ctx.metadata.get("key")
    
    return {"result": value}
```

## Quick Start

### Hello World
```python
from metta.sweep.axiom import Pipeline

# Simple function pipeline
def greet(name: str) -> str:
    return f"Hello, {name}!"

def exclaim(text: str) -> str:
    return f"{text} Welcome to tAXIOM!"

pipeline = (
    Pipeline()
    .stage("greet", lambda: greet("World"))
    .through(str)  # Enforce string output
    .stage("exclaim", exclaim)
    .through(str)
)

result = pipeline.run()
# Output: "Hello, World! Welcome to tAXIOM!"
```

### Complex Example: Hyperparameter Optimization with Phase Scheduling

tAXIOM excels at complex experiment orchestration. Here's a real-world example of multi-phase optimization with parallel seed execution:

```python
from metta.sweep.axiom import Pipeline, Ctx, context_aware
from pydantic import BaseModel
from typing import List, Dict

# Domain models with type safety
class Phase(BaseModel):
    name: str
    exploration_rate: float
    
class Trial(BaseModel):
    params: Dict[str, float]
    score: float
    phase: str

# Phase-aware optimization
@context_aware
def schedule_phase(ctx: Ctx) -> Phase:
    """Adaptive phase scheduling based on progress."""
    trial_num = ctx.metadata.get("trial_num", 0)
    
    if trial_num < 30:
        return Phase(name="explore", exploration_rate=0.8)
    elif trial_num < 60:
        return Phase(name="balance", exploration_rate=0.4)
    else:
        return Phase(name="exploit", exploration_rate=0.1)

def optimize_with_phase(phase: Phase) -> Trial:
    """Run optimization with phase-specific strategy."""
    # Your optimization logic here
    params = sample_params(exploration=phase.exploration_rate)
    score = evaluate(params)
    return Trial(params=params, score=score, phase=phase.name)

# Build pipeline with explicit data flow
optimization_pipeline = (
    Pipeline()
    .stage("schedule", schedule_phase)
    .through(Phase)  # Phase configuration flows out
    
    .stage("optimize", optimize_with_phase)
    .through(Trial)  # Trial results flow out
    
    .stage("update_best", update_best_solution)
    .through(bool)  # Success flag
)

# Run with parallel seeds for statistical robustness
from multiprocessing import Pool

def run_seed(seed: int) -> List[Trial]:
    ctx = Ctx()
    ctx.metadata["seed"] = seed
    
    trials = []
    for i in range(100):
        ctx.metadata["trial_num"] = i
        trial = optimization_pipeline.run(ctx)
        trials.append(trial)
    
    return trials

# Parallel execution across seeds
with Pool() as pool:
    all_results = pool.map(run_seed, range(10))

# Aggregate and analyze
aggregate_pipeline = (
    Pipeline()
    .stage("combine", lambda: combine_results(all_results))
    .through(ExperimentSummary)
    
    .stage("analyze", statistical_analysis)
    .through(Analysis)
    
    .stage("report", generate_report)
    .through(Report)
)

final_report = aggregate_pipeline.run()
```

## Features

### 🔄 **Control Flow**
- Sequential composition with `.stage()`
- Conditional execution with `do_while` and `do_until`
- Parallel execution with `.parallel()`
- Manual loops for complex patterns

### 📝 **Type Safety**
- Pydantic models for structured data
- Runtime type validation at stage boundaries
- Self-documenting data flow

### 🔍 **Observability**
- Hooks for logging, metrics, and monitoring
- Clean separation from business logic
- Non-invasive instrumentation

### 🎯 **Context Management**
- Stateful execution with `Ctx`
- Metadata persistence across stages
- Access to previous stage outputs

### ⚡ **Performance**
- Lazy evaluation
- Parallel execution support
- Minimal overhead

## Use Cases

tAXIOM is ideal for:
- **Hyperparameter optimization** — Multi-phase search with adaptive strategies
- **Curriculum learning** — Progressive task complexity with clean stage transitions
- **Meta-learning** — Composable learning-to-learn pipelines
- **Experiment tracking** — Reproducible workflows with explicit data contracts
- **A/B testing** — Parallel experiment branches with statistical aggregation
- **Model evaluation** — Multi-metric assessment pipelines

## Installation

```bash
# As part of metta
pip install -e .
```

## Examples

See the `examples/` directory for complete examples:
- `hartmann_example.py` — Basic optimization pipeline
- `hartmann_protein.py` — Integration with PROTEIN optimizer
- `hartmann_protein_multiseed.py` — Parallel multi-seed optimization

## Design Principles

1. **Explicit > Implicit**: All data flow is visible and typed
2. **Composition > Inheritance**: Build complex from simple
3. **Separation of Concerns**: Business logic ≠ infrastructure
4. **Fail Fast**: Type violations caught at stage boundaries
5. **Reproducibility First**: Experiments as deterministic graphs

## API Reference

### Core Classes

- `Pipeline`: Main pipeline builder with method chaining
- `Stage`: Atomic computation unit with typed contracts  
- `Ctx`: Context object for state management
- `@context_aware`: Decorator for context-receiving functions

### Pipeline Methods

- `.stage(name, func)`: Add a computation stage
- `.through(output_type, input_type=None)`: Define type contracts
- `.T(type)`: Shorthand for `through(type)`
- `.run(ctx=None)`: Execute the pipeline

### Context Methods

- `ctx.get_stage_output(name)`: Get output from a named stage
- `ctx.metadata`: Dict for storing cross-stage state
- `ctx.artifacts`: Dict for storing large objects

## Contributing

tAXIOM follows a minimalist philosophy. Contributions should:
- Maintain the explicit > implicit principle
- Add power through composition, not features
- Include type annotations and contracts
- Provide clear examples

## License

Part of the Metta project. See main repository for license details.

---

*tAXIOM: Where experiments flow through typed membranes.*
