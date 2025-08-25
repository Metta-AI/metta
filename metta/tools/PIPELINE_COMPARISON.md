# Pipeline vs Monolithic Design: A Practical Comparison

This document compares two implementations of the same training tool to illustrate the trade-offs between monolithic and pipeline-based designs.

## The Challenge

We wanted to make our training implementation swappable - allowing users to provide their own training logic while reusing our setup and teardown code. This is a common need: different training algorithms, distributed strategies, or testing with mock implementations.

## Monolithic Approach (train.py)

The original implementation embeds training deep within the execution flow:

```python
def invoke(self, args: dict[str, str], overrides: list[str]) -> int | None:
    # ... 50+ lines of setup ...
    
    if torch_dist_cfg.is_master:
        with WandbContext(self.wandb, self) as wandb_run:
            handle_train(self, torch_dist_cfg, wandb_run, logger)
    else:
        handle_train(self, torch_dist_cfg, None, logger)
    
    # ... cleanup ...
```

### Making Training Swappable in the Monolith

To make training pluggable here, we'd add an optional parameter:

```python
def invoke(self, args, overrides, train_func=None):
    # ... setup code ...
    
    if torch_dist_cfg.is_master:
        with WandbContext(self.wandb, self) as wandb_run:
            if train_func:
                train_func(
                    run=self.run,
                    run_dir=run_dir,
                    system_cfg=self.system,
                    # ... 7 more parameters ...
                )
            else:
                handle_train(self, torch_dist_cfg, wandb_run, logger)
```

This is actually not that complex - maybe 20 lines of changes. The main friction points are:
- You need to know exactly which parameters to pass
- The WandB context wrapping needs to be preserved
- The if/else for distributed training adds some duplication
- Testing requires setting up all those parameters

It's definitely doable, just a bit less obvious than the pipeline approach.

## Pipeline Approach (train_pipeline.py)

The pipeline version naturally separates stages:

```python
def get_pipeline(self) -> Pipeline:
    return (
        Pipeline()
        .stage("initialize", self._initialize)
        .io("setup_logging", self._setup_logging)
        .io("setup_distributed", self._setup_distributed)
        .through(TrainingState, hooks=[lambda s, c: record_heartbeat()])
        .stage("configure", self._configure_training)
        .io("create_policy_store", self._create_policy_store)
        .io("save_config", self._save_configuration)
        .stage("platform_adjustments", self._apply_platform_adjustments)
        .io("train", self._execute_training, timeout=3600 * 24)
        .io("cleanup", self._cleanup_distributed)
    )
```

### Making Training Swappable in the Pipeline

To make training pluggable, we just change one line:

```python
.require_join("train")  # Training is now a join point
```

Users can then provide their own training:
```python
pipeline.provide_join("train", custom_training_pipeline)
```

The pipeline handles:
- Passing the correct state to training
- Managing the execution order
- Applying guards (like `@wandb_context`) consistently
- Type checking via `TrainingState`

## The Real Difference

Both approaches can be made flexible. The monolithic version would need maybe 20 lines of changes, while the pipeline needs just 1. But the deeper difference is about design intent:

**Monolithic approach:**
- The flexibility is added when needed
- You need to understand the whole flow to make changes
- Parameters are passed explicitly
- Works well when variations are rare

**Pipeline approach:**
- Flexibility is built into the structure
- Each stage is isolated and clear
- State flows through typed objects
- Works well when variations are expected

The pipeline's advantage isn't that the monolithic version *can't* do it - it's that the pipeline makes it obvious and safe. You can see the join points, understand the data flow, and make changes with confidence.

## The Cost

The pipeline version is about 26% longer (265 vs 210 lines). This overhead comes from:
- Method signatures with type hints
- Pipeline construction DSL
- Explicit state passing

## When Each Approach Wins

**Monolithic wins when:**
- The code rarely changes
- The flow is always the same
- You value brevity over flexibility
- The team is small and context is shared

**Pipeline wins when:**
- Different users need different behaviors
- Testing individual components matters
- The flow might vary (different environments, configs)
- Clear data dependencies help onboarding

## Conclusion

Neither approach is universally better. The monolithic version is perfectly fine code - it's clear, works well, and is more concise. Making training swappable in the monolithic version is totally doable - maybe 20 lines of careful changes.

The pipeline version trades some verbosity for making these changes more obvious and safer. It's the difference between "you can do it if you're careful" and "it's hard to do it wrong."

In our case, we decided the pipeline version was worth keeping because:
1. The extension points are visible in the pipeline structure
2. The overhead was reasonable (26% more lines)
3. Changes feel safer with explicit stage boundaries
4. The typed `TrainingState` makes data dependencies clear

But honestly? If you're not planning to swap implementations or test individual stages, the monolithic version is probably the better choice. It's more concise and still perfectly maintainable.

Choose the abstraction that fits your needs, not the one that seems most clever.