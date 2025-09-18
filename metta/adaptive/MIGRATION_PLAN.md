# Migration Plan: Remove AdaptiveTool, Create Specialized Tools

## Overview

Migrate from generic AdaptiveTool with enum-based scheduler registration to specialized tools (TrainAndEvalTool, SweepTool) and direct AdaptiveController access.

## Motivation

The current AdaptiveTool creates unnecessary boilerplate by trying to be generic over fundamentally different experiment types. Each new scheduler requires:
1. Creating scheduler class
2. Creating scheduler config class
3. Adding enum entry to SchedulerType
4. Adding case to AdaptiveTool._create_scheduler()
5. Creating recipe function
6. Wiring everything through AdaptiveTool

This migration eliminates steps 3-6 entirely.

## Phase 1: Remove AdaptiveTool Infrastructure

### Files to delete:
- `metta/tools/adaptive.py` - The entire AdaptiveTool and its enums

### Code to remove:
- `SchedulerType` enum - No longer needed
- `DispatcherType` enum - Move to dispatcher module if needed
- All the manual scheduler wiring in `_create_scheduler()`

## Phase 2: Create Specialized Tools Structure

### New directory structure:
```
metta/tools/experiments/
├── __init__.py
├── train_and_eval.py    # TrainAndEvalTool
├── sweep.py              # SweepTool
└── validation.py         # ValidationTool (future)
``````

### TrainAndEvalTool desig
```python
class TrainAndEvalTool(Tool):
    # Direct, specific fields - no generic configs
    recipe_module: str
    train_entrypoint: str
    eval_entrypoint: str
    max_trials: int
    gpus: int

    def invoke(self, args):
        # Create scheduler directly - no factory needed
        scheduler = TrainAndEvalScheduler(
            TrainAndEvalConfig(...)
        )

        # Create controller directly
        controller = AdaptiveController(
            scheduler=scheduler,
            dispatcher=LocalDispatcher() if self.local else SkypilotDispatcher(),
            store=WandbStore(),
            config=AdaptiveConfig(),
        )
        controller.run()
```

### SweepTool design:
```python
class SweepTool(Tool):
    search_space: dict[str, list[Any]]
    max_trials: int
    strategy: Literal["random", "bayesian", "grid"]

    def invoke(self, args):
        # Create appropriate optimizer based on strategy
        if self.strategy == "bayesian":
            optimizer = ProteinOptimizer(...)
            scheduler = BatchedSyncedOptimizingScheduler(config, optimizer)
        elif self.strategy == "random":
            scheduler = RandomSearchScheduler(...)

        controller = AdaptiveController(scheduler=scheduler, ...)
        controller.run()
```

## Phase 3: Refactor AdaptiveController for Direct Use

### Make AdaptiveController more user-friendly:

1. **Remove hook callbacks from constructor** - They break serializability anyway

   Instead, pass hooks to the `run()` method:
   ```python
   # Constructor becomes simpler - no hooks
   controller = AdaptiveController(
       scheduler=scheduler,
       dispatcher=dispatcher,
       store=store,
       experiment_id="my_exp",
       config=AdaptiveConfig(),
   )

   # Hooks are passed at runtime to run()
   controller.run(
       on_eval_completed=my_eval_hook,
       on_job_dispatch=my_dispatch_hook,
   )
   ```

   This keeps the controller serializable while still supporting hooks for advanced use cases.

2. **Add builder pattern or factory methods:**
```python
# Option A: Builder pattern
controller = (AdaptiveController.builder()
    .with_scheduler(MyScheduler())
    .with_dispatcher(SkypilotDispatcher())
    .with_store(WandbStore())
    .build())

# Option B: Factory methods
controller = AdaptiveController.create_skypilot(
    scheduler=MyScheduler(),
    experiment_id="my_exp"
)

# Option C: Just better defaults
controller = AdaptiveController(
    scheduler=MyScheduler(),  # Required
    experiment_id="my_exp",   # Required
    # Everything else has sensible defaults
)
```

3. **Export controller directly from adaptive module:**
```python
# metta/adaptive/__init__.py
from .adaptive_controller import AdaptiveController
__all__ = ["AdaptiveController", ...]  # Make it public API
```

## Phase 4: Update Experiment Recipes

### Old pattern (remove):
```python
# experiments/recipes/adaptive/train_and_eval.py
def train_and_eval(...) -> AdaptiveTool:
    return AdaptiveTool(
        scheduler_type=SchedulerType.TRAIN_AND_EVAL,
        scheduler_config=config.model_dump(),
        ...
    )
```

### New pattern (two options):

**Option A - Return specialized tool:**
```python
# experiments/recipes/adaptive/train_and_eval.py
def train_and_eval(...) -> TrainAndEvalTool:
    return TrainAndEvalTool(
        recipe_module=recipe_module,
        max_trials=max_trials,
        ...
    )
```

**Option B - Run controller directly:**
```python
# experiments/adaptive/train_and_eval.py
def train_and_eval(...):
    controller = AdaptiveController(
        scheduler=TrainAndEvalScheduler(...),
        ...
    )
    controller.run()
```

## Phase 5: Clean Up Schedulers

### Simplify scheduler initialization:
- Remove state parameter if unused
- Make configs optional with good defaults
- Consider making schedulers dataclasses for cleaner init

```python
@dataclass
class TrainAndEvalScheduler:
    max_trials: int = 10
    recipe_module: str = "experiments.recipes.arena"
    train_entrypoint: str = "train"
    eval_entrypoint: str = "evaluate"

    def schedule(self, runs, available_slots):
        # Implementation
        pass
```

## Phase 6: Documentation Updates

### Update paths:
- `metta/adaptive/README.md` - Update all examples
- `docs/adaptive/README.md` - Update recipes documentation

### Key changes to document:
1. No more AdaptiveTool
2. Two ways to run experiments:
   - Use specialized tools (simple)
   - Use controller directly (flexible)
3. No registration needed for new schedulers
4. Simpler imports

## Migration Benefits

1. **Less code:** Remove ~200 lines of boilerplate
2. **Clearer intent:** `TrainAndEvalTool` vs generic `AdaptiveTool`
3. **Better discovery:** Specialized tools are self-documenting
4. **Easier testing:** Test schedulers and controller directly
5. **Progressive disclosure:** Simple API for simple cases, full control when needed

## Breaking Changes

- All existing adaptive experiment recipes need updating
- AdaptiveTool no longer exists
- SchedulerType enum removed
- No backwards compatibility

## Example: Custom Experiment After Migration

### Before (with AdaptiveTool):
```python
# 1. Create scheduler class
class MyScheduler:
    def __init__(self, config): ...

# 2. Create config class
class MySchedulerConfig(Config):
    param1: str
    param2: int

# 3. Add to enum
class SchedulerType(StrEnum):
    MY_SCHEDULER = "my_scheduler"  # NEW

# 4. Add to factory
def _create_scheduler(self, ...):
    if scheduler_type == SchedulerType.MY_SCHEDULER:  # NEW
        config = MySchedulerConfig.model_validate(...)
        return MyScheduler(config)

# 5. Create recipe
def my_experiment(...) -> AdaptiveTool:
    return AdaptiveTool(
        scheduler_type=SchedulerType.MY_SCHEDULER,
        scheduler_config=MySchedulerConfig(...).model_dump(),
        ...
    )

# 6. Run
uv run ./tools/run.py experiments.recipes.adaptive.my_experiment
```

### After (direct controller):
```python
# 1. Create scheduler class
class MyScheduler:
    def __init__(self, param1, param2): ...

# 2. Use it directly
from metta.adaptive import AdaptiveController

scheduler = MyScheduler(
    param1=value,
    param2=value2,
)

controller = AdaptiveController(
    experiment_id="my_experiment",
    scheduler=scheduler,
)

controller.run()
```

That's it. No enum registration, no factory methods, no tool wrapper.

## Implementation Order

1. **Create new specialized tools** (can coexist with AdaptiveTool initially)
2. **Update AdaptiveController** for public use (better defaults, exports)
3. **Create example migrations** (one recipe using each pattern)
4. **Migrate all existing recipes** to new patterns
5. **Delete AdaptiveTool** and associated boilerplate
6. **Update documentation** with new examples

## Timeline Estimate

- Phase 1-2: 2 hours (create new structure)
- Phase 3: 1 hour (refactor controller)
- Phase 4: 2 hours (migrate recipes)
- Phase 5: 1 hour (clean up schedulers)
- Phase 6: 1 hour (documentation)

Total: ~7 hours for complete migration
