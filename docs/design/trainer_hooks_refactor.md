# Trainer Hook Architecture Design Document

## Executive Summary

This document proposes a hook-based architecture to extract non-training concerns from the Trainer class, achieving true separation of concerns while maintaining simplicity. The design consolidates related functionality into 3 coarse-grained hooks that respect natural system boundaries.

## Current State Analysis

### Tightly Coupled Operations

After analyzing the current trainer implementation, several operations are naturally coupled:

1. **Stats + Wandb + Monitoring**
   - Stats collection feeds directly into wandb logging
   - Memory/system monitoring data goes through stats processing
   - Gradient stats are computed and logged together
   - These three are always used together when wandb is enabled

2. **Checkpointing + Wandb Artifacts**
   - Saving checkpoints often triggers wandb artifact upload
   - Checkpoint metadata includes wandb URIs
   - These operations share state (latest_wandb_uri, latest_saved_epoch)

3. **Evaluation + Stats Recording**
   - Evaluations write to stats database
   - Evaluation scores feed back into training metrics
   - Remote evaluation uses wandb artifacts from checkpointing

4. **Progress Logging + Heartbeats**
   - Both are simple periodic operations
   - Heartbeats are just a single function call
   - Progress logging reads from timer state

## Proposed Hook Architecture

### Core Design Principles

1. **Coarse-grained hooks** - Combine related concerns to reduce complexity
2. **State sharing** - Allow hooks to share state when operations are coupled
3. **Optional dependencies** - Hooks can be present but inactive (e.g., wandb hook when wandb disabled)
4. **Event-driven** - Trainer emits events, hooks respond

### Recommended Hooks (Only 3!)

#### 1. **MetricsHook** - All observability concerns
Combines: Stats tracking, Wandb logging, Monitoring, Progress logging, Heartbeats

```python
class MetricsHook(TrainerHook):
    """Handles all metrics collection and reporting."""
    def __init__(self, 
                 wandb_run: WandbRun | None,
                 stats_client: StatsClient | None,
                 timer: Stopwatch,
                 verbose: bool = True):
        self.wandb_run = wandb_run
        self.stats_client = stats_client
        self.timer = timer
        self.verbose = verbose
        # Initialize monitoring if master rank
        
    def on_rollout_end(self, rollout_stats, trainer_state):
        # Accumulate rollout stats
        
    def on_epoch_end(self, losses_stats, experience, policy, optimizer, trainer_state):
        # Process all stats
        # Log to wandb if enabled
        # Update stats database if client exists
        # Log progress to console if verbose
        # Record heartbeat
```

**Rationale**: These systems are tightly integrated - stats feed wandb, monitoring feeds stats, progress needs timing data. Separating them would require complex inter-hook communication.

#### 2. **CheckpointHook** - Model persistence
Combines: Checkpoint saving/loading, Wandb artifact management

```python
class CheckpointHook(TrainerHook):
    """Handles all checkpoint operations."""
    def __init__(self,
                 checkpoint_manager: CheckpointManager,
                 checkpoint_interval: int,
                 wandb_checkpoint_interval: int):
        self.checkpoint_manager = checkpoint_manager
        self.checkpoint_interval = checkpoint_interval
        self.wandb_checkpoint_interval = wandb_checkpoint_interval
        self.latest_wandb_uri = None
        
    def on_training_start(self, trainer_state) -> PolicyAgent | None:
        # Load existing checkpoint if resuming
        # Return loaded policy or None
        
    def on_epoch_end(self, policy, optimizer, trainer_state, eval_scores, wandb_run):
        # Save checkpoints at intervals
        # Upload to wandb if configured
        # Track latest URIs for evaluation
```

**Rationale**: Checkpointing and wandb artifacts are naturally coupled - saves trigger uploads, URIs are tracked together.

#### 3. **EvaluationHook** - Policy evaluation
Handles: Local/remote evaluation, Replay generation

```python
class EvaluationHook(TrainerHook):
    """Handles policy evaluation."""
    def __init__(self,
                 eval_config: EvaluationConfig | None,
                 curriculum: Curriculum):
        self.eval_config = eval_config
        self.curriculum = curriculum
        self.eval_scores = EvalRewardSummary()
        
    def on_epoch_end(self, checkpoint_hook, trainer_state, stats_client, wandb_run):
        # Run evaluations at configured intervals
        # Can get latest checkpoint URI from checkpoint_hook
        # Updates self.eval_scores for metrics hook to use
        
    def get_eval_scores(self) -> EvalRewardSummary:
        return self.eval_scores
```

**Rationale**: Evaluation is a distinct concern but needs checkpoint URIs and feeds metrics back.

## Implementation Phases

### Phase 1: Hook Infrastructure (Low Risk)
**Goal**: Create the foundational hook system without changing existing trainer

#### Tasks:
1. Create `metta/rl/hooks/base.py` with `TrainerHook` base class
2. Define lifecycle methods in base class
3. Create `metta/rl/hooks/__init__.py` to expose hooks
4. Add hook registration mechanism to `TrainerState`
5. Create unit tests for hook infrastructure

#### Files to create:
- `metta/rl/hooks/base.py`
- `metta/rl/hooks/__init__.py`
- `tests/rl/hooks/test_base.py`

#### Success criteria:
- [ ] Hook base class exists with all lifecycle methods
- [ ] Hooks can be registered and called in sequence
- [ ] Unit tests pass for hook infrastructure

### Phase 2: CheckpointHook Implementation (Medium Risk)
**Goal**: Extract checkpointing logic as our first hook

#### Tasks:
1. Create `metta/rl/hooks/checkpoint.py` with `CheckpointHook` class
2. Move checkpoint loading logic from trainer to hook
3. Move checkpoint saving logic from trainer to hook
4. Move wandb artifact upload logic to hook
5. Add hook to trainer in `train()` function
6. Test checkpoint functionality still works

#### Files to modify:
- Create: `metta/rl/hooks/checkpoint.py`
- Modify: `metta/rl/trainer.py` (remove checkpoint code)
- Modify: `metta/rl/trainer.py` `train()` function (add hook)
- Create: `tests/rl/hooks/test_checkpoint.py`

#### Success criteria:
- [ ] Checkpointing works through hook
- [ ] Wandb artifacts still upload correctly
- [ ] Can resume from checkpoints
- [ ] Tests pass for checkpoint operations

### Phase 3: MetricsHook Implementation (High Complexity)
**Goal**: Extract all metrics, stats, and monitoring logic

#### Tasks:
1. Create `metta/rl/hooks/metrics.py` with `MetricsHook` class
2. Move `StatsTracker` initialization to hook
3. Move `setup_monitoring` and `cleanup_monitoring` to hook
4. Move `process_stats` call to hook
5. Move `log_training_progress` to hook
6. Move gradient stats computation to hook
7. Move heartbeat recording to hook
8. Test all metrics still work

#### Files to modify:
- Create: `metta/rl/hooks/metrics.py`
- Modify: `metta/rl/trainer.py` (remove stats/monitoring code)
- Modify: `metta/rl/stats.py` (may need refactoring)
- Create: `tests/rl/hooks/test_metrics.py`

#### Success criteria:
- [ ] Wandb metrics still log correctly
- [ ] Stats database updates work
- [ ] Progress logging displays
- [ ] Monitoring data collected
- [ ] Heartbeats recorded

### Phase 4: EvaluationHook Implementation (Medium Risk)
**Goal**: Extract evaluation logic into its own hook

#### Tasks:
1. Create `metta/rl/hooks/evaluation.py` with `EvaluationHook` class
2. Move evaluation logic from trainer to hook
3. Handle interaction with CheckpointHook for URIs
4. Move replay upload logic to hook
5. Test evaluations still run

#### Files to modify:
- Create: `metta/rl/hooks/evaluation.py`
- Modify: `metta/rl/trainer.py` (remove evaluation code)
- Create: `tests/rl/hooks/test_evaluation.py`

#### Success criteria:
- [ ] Local evaluations work
- [ ] Remote evaluations work
- [ ] Evaluation scores feed into metrics
- [ ] Replay uploads work

### Phase 5: Clean Trainer Creation (High Risk)
**Goal**: Create the new clean trainer class

#### Tasks:
1. Create `metta/rl/clean_trainer.py` with new `Trainer` class
2. Implement pure training loop (rollout + train)
3. Add hook calling at appropriate points
4. Move loss instantiation to trainer
5. Remove all non-training imports

#### Files to modify:
- Create: `metta/rl/clean_trainer.py`
- Create: `tests/rl/test_clean_trainer.py`

#### Success criteria:
- [ ] Trainer only imports training modules
- [ ] Training loop works with hooks
- [ ] No direct wandb/stats/checkpoint code in trainer
- [ ] Tests pass for core training

### Phase 6: Integration and Migration (High Risk)
**Goal**: Wire everything together and migrate to new system

#### Tasks:
1. Update `train()` function to use new Trainer and hooks
2. Create hooks based on configuration
3. Ensure backward compatibility
4. Add feature flag to toggle old vs new trainer
5. Update integration tests
6. Performance testing

#### Files to modify:
- Modify: `metta/rl/trainer.py` (train function)
- Update: All integration tests
- Update: Example configurations

#### Success criteria:
- [ ] Can toggle between old and new trainer
- [ ] All existing functionality preserved
- [ ] Performance comparable or better
- [ ] Integration tests pass

### Phase 7: Cleanup and Deprecation (Low Risk)
**Goal**: Remove old trainer code once new system is stable

#### Tasks:
1. Remove old trainer implementation
2. Remove feature flag
3. Clean up unused imports
4. Update documentation
5. Update all examples

#### Files to modify:
- Delete old code from `metta/rl/trainer.py`
- Update all documentation
- Update example scripts

#### Success criteria:
- [ ] Old trainer code removed
- [ ] Documentation updated
- [ ] All examples use new system

## Risk Mitigation

### Incremental Rollout
- Each phase can be tested independently
- Feature flag allows rollback if issues found
- Old trainer remains until new one proven stable

### Testing Strategy
- Unit tests for each hook
- Integration tests with mock hooks
- Performance benchmarks to ensure no regression
- Gradual rollout to catch issues early

### Backward Compatibility
- Configuration remains the same
- External API unchanged
- Internal changes transparent to users

## Benefits of This Design

1. **Simplicity** - Only 3 hooks to understand and maintain
2. **Natural boundaries** - Hooks align with how systems actually interact
3. **Testability** - Can mock/stub entire subsystems easily
4. **Flexibility** - Hooks can be enabled/disabled/swapped at runtime
5. **Maintains coupling where appropriate** - Doesn't artificially separate tightly coupled systems

## Example Usage

```python
def train(run_dir, config, ...):
    # Initialize core training components
    curriculum = Curriculum(config.curriculum)
    policy = MettaAgent(env, system_cfg, agent_cfg)
    trainer = Trainer(config, curriculum, policy)
    
    # Add hooks based on configuration
    if config.checkpoint.checkpoint_interval > 0:
        checkpoint_hook = CheckpointHook(
            checkpoint_manager, 
            config.checkpoint.checkpoint_interval,
            config.checkpoint.wandb_checkpoint_interval
        )
        trainer.add_hook(checkpoint_hook)
    
    if wandb_run or stats_client or config.verbose:
        metrics_hook = MetricsHook(
            wandb_run, 
            stats_client,
            timer,
            config.verbose
        )
        trainer.add_hook(metrics_hook)
    
    if config.evaluation:
        eval_hook = EvaluationHook(config.evaluation, curriculum)
        trainer.add_hook(eval_hook)
    
    # Run training with clean separation
    trainer.train()
```

## Open Questions

1. **Hook communication**: Should hooks communicate through:
   - Shared trainer_state dict? ✓ (Recommended - simple and flexible)
   - Direct hook references?
   - Event bus pattern?

2. **Error handling**: How should hooks handle failures?
   - Silent failure with logging?
   - Propagate to trainer? ✓ (Recommended - trainer decides criticality)
   - Configurable behavior?

3. **Distributed training**: How do hooks behave across ranks?
   - Only run on master? ✓ (Recommended for most hooks)
   - Run on all with rank awareness?
   - Hook-specific behavior? ✓ (CheckpointHook needs coordination)

## Timeline Estimate

- Phase 1 (Infrastructure): 2-3 hours
- Phase 2 (CheckpointHook): 3-4 hours
- Phase 3 (MetricsHook): 4-6 hours
- Phase 4 (EvaluationHook): 3-4 hours
- Phase 5 (Clean Trainer): 4-5 hours
- Phase 6 (Integration): 4-6 hours
- Phase 7 (Cleanup): 2-3 hours

**Total estimate**: 22-31 hours of development time

## Conclusion

This design achieves the goal of extracting non-training concerns from the Trainer while maintaining simplicity. By using coarse-grained hooks that respect natural system boundaries, we avoid unnecessary complexity while gaining modularity and testability. The phased implementation approach ensures we can deliver value incrementally while maintaining system stability.