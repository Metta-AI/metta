# tAXIOM (tiny AXIOM)

```
        _      _____  __  __  _____   ____   __  __ 
       | |_   |  _  | \ \/ / |_   _| / __ \ |  \/  |
       | __|  | |_| |  \  /    | |  | |  | || |\/| |
       | |_   |  _  |  /  \    | |  | |__| || |  | |
        \__|  |_| |_| /_/\_\  |___|  \____/ |_|  |_|
                                                     
```

> **A minimalist DSL for orchestrating RL experiments with explicit data flow**

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                           🚀 NEXT WEEK'S ROADMAP                              ┃
┃                                                                               ┃
┃  Production-ready release requires these five features:                       ┃
┃                                                                               ┃
┃  1. CONTROL FLOW PRIMITIVES                                                  ┃
┃     • do_while / do_until loops for iterative optimization                   ┃
┃     • Conditional execution (if_then, switch)                                ┃
┃     • Parallel stage execution for independent operations                    ┃
┃                                                                               ┃
┃  2. ERROR HANDLING                                                           ┃
┃     • Graceful failure recovery with retry logic                             ┃
┃     • Error boundaries to isolate failures                                   ┃
┃     • Fallback stages for critical operations                                ┃
┃                                                                               ┃
┃  3. TYPE VALIDATION                                                          ┃
┃     • Static validation at pipeline construction time                        ┃
┃     • Optional runtime validation for debugging                              ┃
┃     • MyPy plugin for compile-time type checking                             ┃
┃                                                                               ┃
┃  4. REUSABLE HOOK LIBRARY                                                    ┃
┃     • Timing hooks (measure stage duration)                                  ┃
┃     • Logging hooks (structured logging with levels)                         ┃
┃     • Metrics hooks (wandb, tensorboard, mlflow)                             ┃
┃     • Checkpoint hooks (save/restore pipeline state)                         ┃
┃     • Notification hooks (slack, email on completion/failure)                ┃
┃                                                                               ┃
┃  5. REUSABLE GUARD LIBRARY                                                   ┃
┃     • @master_only (distributed execution control)                           ┃
┃     • @gpu_required / @cpu_only (hardware constraints)                       ┃
┃     • @retry_on_failure(n) (automatic retry logic)                           ┃
┃     • @timeout(seconds) (execution time limits)                              ┃
┃     • @requires_auth (permission checks)                                     ┃
┃     • @cache_result (memoization control)                                    ┃
┃     • @rate_limit(n) (API throttling)                                        ┃
┃                                                                               ┃
┃  With these features, tAXIOM becomes a production-ready, lightweight         ┃
┃  Autonomous eXperimentation & Iterative Optimization Manager.                ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

## Current Status: MVP Complete ✅

The tAXIOM MVP is now complete and working in production. It successfully orchestrates hyperparameter sweeps with clear separation between stages (deterministic computation), I/O (external operations), and hooks (observability).

## Philosophy

tAXIOM models experiments as **sequential chains of stages** separated by **typed membranes**. These membranes serve as attachment points for hooks (observability) and type documentation, while keeping business logic clean.

### Pragmatism Over Dogma

While tAXIOM was developed with a modular, functional philosophy in mind, it is in no way dogmatic. It is perfectly feasible — and often encouraged — to have stages that are 500+ lines of code, have side effects, and occasionally violate the tenets of modular design. 

The goal of tAXIOM is, first and foremost, to help write experiments that are:
- **Self-documenting** through explicit data flow
- **Testable** when you need that confidence
- **Debuggable** with clear observation points
- **Lucid** in their structure and intent

...while simultaneously promoting flexibility and fast iteration. Use the structure when it helps, ignore it when it doesn't. A 500-line stage that works is better than 50 perfectly factored stages that don't.

We developed tAXIOM with the intent to make self-optimizing pipelines ubiquitous and easy to use, to enable automated A/B testing, and to clearly separate the **flow** of an experiment from its logic (and along the way, its housekeeping). In those domains, this tiny DSL hits a sweet spot in balancing clarity and flexibility. It is also, by all accounts, completely overkill in many other experimental situations. If you're running a simple script once to get a result, you probably don't need tAXIOM — and that's perfectly fine.

### Core Components

1. **Stages**: Pure, deterministic computation (given config/seed)
2. **I/O**: External operations (data loading, API calls, saving)
3. **Hooks**: Removable observations attached at membranes (logging, metrics)
4. **Guards**: Execution control policies as decorators (@master_only, @gpu_required)
5. **Types**: Data contracts declared at membranes

### Design Principles

- **Explicit > Magic**: All data flow and configuration access is visible
- **Separation of Concerns**: Guards modify functions, hooks observe data flow
- **Composability**: Complex pipelines built from simple, reusable components
- **Idempotency**: Pipelines are stateless, context carries state

## Core API

### Standard Notation

```python
from metta.sweep.axiom import Pipeline
from metta.sweep.axiom.guards import master_only, gpu_required, timeout

# Guards are decorators on functions
@master_only
@gpu_required
@timeout(60)
def train_model(data):
    """This stage only runs on master, needs GPU, has 60s timeout."""
    return model.train(data)

# Pipeline composition
pipeline = (
    Pipeline()
    .io("load", load_data)              # I/O operation
    .stage("process", process_data)      # Deterministic computation
    .through(dict, hooks=[log_hook])     # Membrane with hooks and types
    .stage("train", train_model)          # Guards already applied via decorators
    .stage("evaluate", evaluate_model)    # Evaluation computation
    .io("save", save_results)            # I/O operation
)

result = pipeline.run(ctx)
```

### Clean Separation of Concerns

- **Guards** = Decorators on functions (execution control)
- **Hooks** = Attached at membranes (observation)
- **Types** = Declared at membranes (contracts)

Guards modify the function before it enters the pipeline, while hooks observe data flowing between stages. This keeps the pipeline API focused on data flow and composition, while execution policies stay with the functions themselves.

### Whimsical AXIOM Notation (Experimental)

Using letters from "AXIOM" for ultra-compact pipelines:
- `ax()` - Add stage (compute**ax**ion)
- `io()` - Input/output (unchanged)
- `om()` - Attach h**o**ok **m**embrane
- `xi()` - E**xi**t with type contract (membrane)

```python
# Current sweep pipeline in AXIOM notation
pipeline = (
    Pipeline()
    .io("init", exp.initialize_services)
    .io("load", exp.load_previous_observations)
    .ax("suggest", exp.suggest_hyperparameters)
    .om(lambda r, ctx: logger.info(f"Trial {ctx.metadata['trial_index']}: suggestion"))
    .io("get_run", exp.get_run_name)
    .ax("train", exp.train_model)
    .xi(dict)  # Type membrane
    .ax("evaluate", exp.evaluate_model)
    .io("fetch", exp.fetch_metrics)
    .ax("calculate", exp.calculate_metrics)
    .om(lambda r, ctx: logger.info(f"Score: {r['score']:.4f}"))
    .ax("update", exp.update_optimizer)
    .io("record", exp.record_to_wandb)
)
```

This notation is more compact but perhaps less immediately readable. The standard notation remains the recommended approach for clarity.

## Working Example: Hyperparameter Sweep

```python
from metta.sweep.axiom import Pipeline, Ctx
from metta.sweep.axiom_sweep import get_protein_sweep

# Get configured pipeline from factory
sweep_pipeline = get_protein_sweep(
    sweep_name="my_sweep",
    protein_config=protein_config,
    train_tool_factory=lambda name: TrainTool(name),
    wandb_cfg=wandb_config,
    num_trials=10,
    evaluation_simulations=[navigation_sim]
)

# Orchestrate execution
for trial_idx in range(num_trials):
    ctx = Ctx()
    ctx.metadata["trial_index"] = trial_idx
    result = sweep_pipeline.run(ctx)
    print(f"Trial {trial_idx}: score={result['score']:.4f}")
```

## Configuration Access Patterns

### Method-Based (Preferred)
```python
class SweepExperiment:
    def __init__(self, cfg):
        self.cfg = cfg
    
    def process(self, data):
        return data * self.cfg.multiplier

exp = SweepExperiment(config)
pipeline.stage("process", exp.process)
```

### Lambda with Closure
```python
cfg = config  # Close over config
pipeline.stage("transform", lambda x: x * cfg.multiplier)
```

### Partial Application
```python
from functools import partial
pipeline.stage("process", partial(process_fn, multiplier=cfg.multiplier))
```

## Stage vs I/O Classification

### Stages (Deterministic Computation)
- Training models
- Evaluation
- Parameter generation
- Metric calculation
- Data transformation

### I/O Operations (External Data Transfer)
- File reading/writing
- Network API calls
- Database queries
- Service communication
- External ID generation

The distinction is about **what** the operation does, not **how long** it takes.

## Design Documents

- [AXIOM_DESIGN.md](AXIOM_DESIGN.md) - Design philosophy and decisions
- [SWEEP_DESIGN.md](SWEEP_DESIGN.md) - Integration with sweep system
- [CONFIG_ACCESS.md](CONFIG_ACCESS.md) - Configuration access patterns
- [IO_VS_STAGE.md](IO_VS_STAGE.md) - Classification guide

## Testing

```bash
# Run all tAXIOM tests
uv run python -m pytest tests/sweep/test_axiom_core.py tests/sweep/test_axiom_types.py -v

# Run a quick sweep test
uv run experiments/sweep_arena.py sweep_hpo_quick --no-launch sweep_name=axiom_test num_trials=2
```

## Contributing

tAXIOM follows a minimalist philosophy. Contributions should:
- Maintain the explicit > implicit principle
- Keep the stage/I/O/hook separation clear
- Include tests for new functionality
- Update documentation to match implementation

---

*tAXIOM: Sequential. Explicit. Simple.*