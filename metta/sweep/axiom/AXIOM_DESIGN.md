# AXIOM_DESIGN.md

## tAXIOM Design Philosophy

### Core Philosophy

tAXIOM (tiny AXIOM) is a minimalist DSL for orchestrating complex experiments through **composable, idempotent chains of atomic operations** separated by **typed, programmable membranes**. It emerged from the need to make reinforcement learning experiments reproducible, debuggable, and self-documenting while maintaining the flexibility required for cutting-edge research.

The framework is built on three fundamental principles:

1. **Explicitness over Magic** — Every data transformation, side effect, and state change is visible in the pipeline definition
2. **Predictability through Purity** — Core computations are deterministic and side-effect free
3. **Composition through Isolation** — Complex workflows emerge from simple, isolated components

### The Membrane Model

At the heart of tAXIOM is the concept of **membranes** — boundaries between pipeline stages where data flows through typed contracts. These membranes serve as:

- **Type gates** that validate data structure
- **Observation points** for monitoring and logging
- **Control points** for error handling and recovery
- **Documentation** of data flow expectations

```python
Pipeline()
  .stage("compute", pure_function)
  .through(OutputType)  # <-- Membrane: validates, documents, observes
  .stage("next", another_function)
```

## Core Concepts

### The Five Component Types

tAXIOM distinguishes between five fundamental component types, each with a specific purpose and attachment point:

1. **Stages** - Pure computation (deterministic given inputs)
   - Attached via: `.stage(name, func)`
   - Example: `train_model`, `calculate_metrics`

2. **I/O Operations** - External data transfer
   - Attached via: `.io(name, func)`
   - Example: `load_data`, `save_results`, `fetch_from_api`

3. **Hooks** - Observational side effects
   - Attached via: `.through(type, hooks=[...])`
   - Example: logging, metrics, notifications
   - Property: Removable without affecting computation

4. **Guards** - Execution control policies
   - Attached via: Decorators on functions
   - Example: `@master_only`, `@gpu_required`, `@timeout(60)`
   - Property: Modify when/where/if a function executes

5. **Types** - Data contracts
   - Attached via: `.through(TypeClass)`
   - Example: `dict`, `MyDataClass`, `pd.DataFrame`
   - Property: Documentation and optional validation

### Clean Separation Principle

```python
# Guards: Decorators on the function (execution control)
@master_only
@gpu_required
def train_model(data):
    return model.train(data)

# Pipeline: Composition and data flow
pipeline = (
    Pipeline()
    .io("load", load_data)              # I/O operation
    .stage("train", train_model)        # Stage with guards pre-applied
    .through(ModelOutput, hooks=[       # Membrane: types + hooks
        log_metrics,
        save_checkpoint
    ])
)
```

This separation ensures:
- Guards modify the function before pipeline entry
- Hooks observe data flow between stages
- Types document and validate at boundaries
- Stages remain pure and testable
- I/O is clearly marked and isolated

### Pipeline

The `Pipeline` is the primary composition mechanism — a lazy-evaluated chain of operations that describes a computation graph.

**What it is:**
- A declarative description of data flow
- A composable unit that can be nested within other pipelines
- A type-safe container ensuring data contracts

**What it is NOT:**
- An executor (execution happens on `.run()`)
- A scheduler (no built-in parallelism management)
- A state container (state lives in Context)

```python
pipeline = Pipeline()  # Declaration
result = pipeline.run(ctx)  # Execution
```

### Stage

A `Stage` is an atomic unit of **predictable computation**. Given the same input, a stage will always produce the same output.

**What it is:**
- A pure function wrapper
- A deterministic transformation
- The core computational unit
- May read immutable data from context

**What it is NOT:**
- A place for side effects
- A network or filesystem accessor
- A context mutator (can read, cannot write)

```python
# ✅ Good Stage
.stage("process", lambda data: transform(data))
.stage("prepare", lambda sugg, ctx: merge_configs(ctx.base_config, sugg))  # Reading immutable config is OK

# ❌ Bad Stage  
.stage("fetch", lambda: requests.get(url))  # Should be I/O
.stage("mutate", lambda ctx: ctx.update(value=5))  # Context mutation should be I/O
```

### I/O Operation

An `I/O` operation represents an **unpredictable interaction** with the world outside the pipeline's computational model.

**What it is:**
- External system interactions (network, filesystem, database)
- Context mutations (state changes)
- Non-deterministic operations

**What it is NOT:**
- Pure computation
- Removable logging
- Internal implementation details

```python
# ✅ Good I/O
.io("load_data", lambda: fetch_from_database())
.io("update_budget", lambda ctx, cost: ctx.update(budget=ctx.budget - cost))

# ❌ Bad I/O
.io("calculate", lambda x: x * 2)  # Should be stage
```

### Hook

A `Hook` is a **removable observer** that performs side effects without affecting the pipeline's data flow.

**What it is:**
- Logging and monitoring
- Notifications and alerts
- Debugging aids
- Optional checkpointing

**What it is NOT:**
- Essential for pipeline execution
- Data transformer
- State mutator

```python
# ✅ Good Hook
.hook(lambda data: logger.info(f"Processing: {data}"))
.hook(lambda data: send_slack_notification(data))

# ❌ Bad Hook
.hook(lambda data: save_critical_state(data))  # Should be I/O if essential
```

### Context

The `Context` (Ctx) is the **state container** that flows through the pipeline, providing both immutable data and mutable flow control.

**What it is:**
- Immutable data carrier for stages
- Mutable state for flow control
- Resource container for services

**What it is NOT:**
- A global variable store
- A communication channel between stages
- A cache for intermediate results

```python
@dataclass
class Context:
    # Immutable data (accessed by stages)
    config: TrainConfig
    eval_data: EvalData
    
    # Mutable flow state (managed by flow operations)
    trial_id: int
    budget: float
    
    # Resources (used by I/O operations)
    logger: LoggerService
    storage: StorageService
```

### Control Flow Operations

Control flow operations (`do_while`, `do_until`, `repeat_n`) are **pipeline composers** that manage iteration and conditional execution.

**What they are:**
- Iteration managers
- Context update owners
- Pipeline composers

**What they are NOT:**
- Regular stages
- Side effect performers
- Data transformers

```python
# Control flow owns context updates
.do_while(
    lambda ctx: ctx.score < threshold,
    inner_pipeline  # This pipeline is state-idempotent
)  # do_while manages iteration count, score tracking, etc.
```

## Decision Flowchart

When deciding whether to use a Stage, I/O Operation, or Hook:

```
┌─────────────────────────────────────────────────────────────────┐
│                      DECISION FLOWCHART                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐                                          │
│  │ Operation to Add │                                          │
│  └────────┬─────────┘                                          │
│           │                                                     │
│           ▼                                                     │
│  ┌────────────────────────────┐                               │
│  │ Can this be removed        │                               │
│  │ without breaking pipeline? │────Yes───► HOOK               │
│  └────────┬───────────────────┘           • Logging          │
│           │                                • Notifications    │
│           No                               • Debug output     │
│           │                                                   │
│           ▼                                                   │
│  ┌────────────────────────────┐                              │
│  │ Is output predictable      │                              │
│  │ given same input?          │────No────► I/O OPERATION     │
│  └────────┬───────────────────┘           • API calls        │
│           │                                • File reads       │
│           Yes                              • Context updates  │
│           │                                • datetime.now()   │
│           ▼                                                   │
│  ┌────────────────────────────┐                              │
│  │ Does it access external    │                              │
│  │ systems?                   │────No────► STAGE             │
│  └────────┬───────────────────┘           • Computations     │
│           │                                • Transformations  │
│           Yes                              • Pure functions   │
│           │                                                   │
│           ▼                                                   │
│  ┌────────────────────────────┐                              │
│  │ Are external calls just    │                              │
│  │ implementation details?    │────Yes───► STAGE             │
│  └────────┬───────────────────┘           • train_model()    │
│           │                                  (logs internally)│
│           No                                                  │
│           │                                                   │
│           └────────────────────────────────► I/O OPERATION   │
│                                              • Data fetching  │
│                                              • State loading  │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

## Design Principles

### Functional Core, Imperative Shell

tAXIOM follows the "Functional Core, Imperative Shell" pattern:

- **Functional Core**: Stages perform pure computation on data
- **Imperative Shell**: I/O operations handle effects at boundaries

```python
Pipeline()
  # Imperative Shell: Load data
  .io("load", fetch_observations)
  
  # Functional Core: Pure computation
  .stage("suggest", compute_suggestion)
  .stage("configure", prepare_config)
  .stage("train", train_model)
  .stage("evaluate", compute_metrics)
  
  # Imperative Shell: Save results
  .io("save", store_results)
```

### State Idempotency

Composed pipelines must be **state idempotent** — running an inner pipeline should not affect the outer context:

```python
# Inner pipeline doesn't mutate outer context
outer_pipeline = Pipeline()
  .do_while(
      condition,
      Pipeline()  # This inner pipeline
        .stage("work", work_fn)  # doesn't mutate
        .stage("check", check_fn)  # outer state
  )
# Only do_while manages state updates
```

### Type Safety Through Membranes

Type contracts are enforced at stage boundaries:

```python
Pipeline()
  .stage("generate", generate_params)
  .through(ParamConfig)  # Validates output type
  
  .stage("optimize", run_optimization)  
  .through(OptResult, input_type=ParamConfig)  # Input and output types
```

### Explicit Effects

All effects must be visible in the pipeline definition:

```python
# ✅ Explicit
.io("save_checkpoint", lambda m: write_to_disk(m))

# ❌ Hidden
.stage("train", lambda c: train_and_save(c))  # Save is hidden!
```

## Common Patterns

### Resource Management

```python
# Resources are passed via context to I/O operations
ctx = Context(
    resources={
        "logger": WandbLogger(),
        "storage": S3Storage(),
        "optimizer": ProteinOptimizer(config)
    }
)

Pipeline()
  .io("suggest", lambda ctx: ctx.resources.optimizer.suggest())
  .stage("train", train_model)
  .io("log", lambda ctx, r: ctx.resources.logger.log(r))
```

### Budget-Aware Execution

```python
Pipeline()
  .stage("estimate_cost", estimate_training_cost)
  .io("check_budget", lambda cost, ctx: cost <= ctx.budget)
  .when(
      lambda can_afford: can_afford,
      Pipeline()
        .stage("train", train_model)
        .io("deduct_budget", lambda r, ctx: ctx.update(
            budget=ctx.budget - r.actual_cost
        ))
  )
```

### Multi-Phase Optimization

```python
Pipeline()
  .do_while(
      lambda ctx: not converged(ctx),
      Pipeline()
        .stage("select_phase", lambda ctx: choose_phase(ctx.trial_count))
        .stage("suggest", lambda phase: suggest_with_strategy(phase))
        .stage("evaluate", evaluate_suggestion)
  )
```

## Anti-Patterns

### ❌ Hidden State Mutation

```python
# Bad: State mutation hidden in stage
.stage("process", lambda data: process_and_update_global(data))

# Good: Explicit I/O for state changes
.stage("process", process_data)
.io("update_state", lambda r, ctx: ctx.update(state=r))
```

### ❌ Impure Stages

```python
# Bad: Non-deterministic stage
.stage("generate", lambda: random.random())

# Good: Seed in context for determinism
.stage("generate", lambda ctx: random.Random(ctx.seed).random())
```

### ❌ Essential Operations as Hooks

```python
# Bad: Critical operation as removable hook
.hook(lambda r: save_training_result(r))

# Good: Essential operations as I/O
.io("save_result", lambda r: save_training_result(r))
```

### ❌ Object Passing in Stages

```python
# Bad: Passing mutable objects
.stage("modify", lambda optimizer: optimizer.step())

# Good: Pass configs, return new state
.stage("step", lambda config: compute_new_params(config))
```

## Configuration vs Objects

### Pass Configurations to Stages

Stages should receive **typed configurations** (data structures) rather than objects:

- Ensures explicit dependencies
- Makes breaking changes visible
- Enables serialization and replay
- Simplifies testing

```python
# ✅ Good: Config
.stage("train", lambda cfg: train_model(cfg: TrainConfig))

# ❌ Bad: Object
.stage("train", lambda trainer: trainer.train())
```

### Use Objects for Services

I/O operations and hooks can use **service objects** for abstraction:

- Hides implementation details
- Allows runtime substitution
- Encapsulates stateful operations

```python
# ✅ Good: Service object in I/O
.io("log", lambda ctx, data: ctx.logger.write(data))

# Logger can be WandbLogger, FileLogger, etc.
```

## Summary

tAXIOM provides a disciplined approach to experiment orchestration through:

1. **Clear separation** between computation (stages), effects (I/O), and observation (hooks)
2. **Explicit data flow** with typed contracts at boundaries
3. **Predictable execution** through pure functions and immutable data
4. **Flexible composition** via nested pipelines and control flow
5. **State management** through context with clear update semantics

The framework doesn't aim to hide complexity but rather to **make it manageable** through explicit boundaries and predictable behavior. By following these design principles, experiments become reproducible, debuggable, and ultimately, trustworthy.