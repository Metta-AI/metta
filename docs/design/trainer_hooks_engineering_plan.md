# Trainer Hooks Engineering Plan

**Status:** Approved by QA Team (8.6/10)  
**Date:** 2025-09-03  
**Implementation Ready:** Yes ✅

## Executive Summary

This engineering plan describes a hook-based architecture to extract non-training concerns from the Trainer class. The design aligns with existing composable losses patterns, extends the current callback system, and maintains consistency with `BaseLoss` patterns while achieving clean separation of concerns.

**QA Assessment:** The revised design successfully addresses all audit findings and demonstrates deep understanding of the existing codebase. Ready for implementation with minor clarifications addressed in this document.

## Key Design Changes from Original Plan

1. **Align with existing callbacks** - Use `on_new_training_run`, `on_rollout_start`, `on_train_phase_end`, `on_mb_end` 
2. **Add missing callbacks minimally** - Only add `on_epoch_end` and `on_train_start` (renamed from on_rollout_end for clarity)
3. **Follow TrainerState pattern** - All data passes through `TrainerState`, not explicit parameters
4. **Extend rather than replace** - Build hooks alongside losses, not as replacement

## Architecture Overview

### Hook Base Class (Aligned with BaseLoss)

```python
class TrainerHook:
    """Base class for trainer hooks, following BaseLoss patterns."""
    
    def __init__(self, trainer_cfg, device: torch.device):
        self.trainer_cfg = trainer_cfg
        self.device = device
    
    # Existing callbacks from BaseLoss
    def on_new_training_run(self, trainer_state: TrainerState) -> None:
        """Called at the start of training."""
        pass
    
    def on_rollout_start(self, trainer_state: TrainerState) -> None:
        """Called before rollout phase begins."""
        pass
    
    def on_mb_end(self, trainer_state: TrainerState) -> None:
        """Called after each minibatch."""
        pass
    
    def on_train_phase_end(self, trainer_state: TrainerState) -> None:
        """Called after training phase completes."""
        pass
    
    # New callbacks we need to add
    def on_rollout_end(self, trainer_state: TrainerState) -> None:
        """Called after rollout phase completes."""
        pass
    
    def on_epoch_end(self, trainer_state: TrainerState) -> None:
        """Called at the end of each epoch."""
        pass
    
    def on_training_end(self, trainer_state: TrainerState) -> None:
        """Called when training completes."""
        pass
```

### Extended TrainerState

```python
from metta.rl.stats import StatsTracker
from metta.eval.eval_request_config import EvalRewardSummary
from metta.rl.experience import Experience
from metta.agent.metta_agent import PolicyAgent
from metta.common.profiling.stopwatch import Stopwatch

@dataclass(slots=True)
class TrainerState:
    """Extended to carry hook-related data."""
    # Existing fields
    agent_step: int = 0
    epoch: int = 0
    update_epoch: int = 0
    mb_idx: int = 0
    optimizer: torch.optim.Optimizer | None = None
    training_env_id: slice | None = None
    stop_rollout: bool = False
    stop_update_epoch: bool = False
    
    # New fields for hooks to share data (with proper types)
    rollout_stats: dict[str, list[float]] | None = None
    loss_stats: dict[str, float] | None = None
    eval_scores: EvalRewardSummary | None = None
    experience: Experience | None = None
    policy: PolicyAgent | None = None
    latest_checkpoint_uri: str | None = None
    latest_wandb_uri: str | None = None
    stats_tracker: StatsTracker | None = None
    timer: Stopwatch | None = None
```

## Pre-Implementation Checklist

- [ ] Confirm type hints for TrainerState extensions
- [ ] Document hook execution order dependencies
- [ ] Define error handling strategy (log vs fail-fast)
- [ ] Confirm distributed training behavior
- [ ] Set up performance benchmarking baseline

## Implementation Phases

### Pre-Phase: Add Missing Callbacks to Trainer (2-3 hours)
**Goal**: Add the two missing callbacks we need to the trainer flow

#### Tasks:
1. Add `on_rollout_end` callback after rollout phase (line ~383)
2. Add `on_epoch_end` callback after epoch increment (line ~444)  
3. Extend `TrainerState` to carry hook data
4. Test callbacks are called correctly

#### Files to modify:
- `metta/rl/trainer.py` - Add callback invocations
- `metta/rl/trainer_state.py` - Add new fields
- `tests/rl/test_trainer.py` - Test new callbacks

#### Implementation:
```python
# In trainer.py, after line 383 (accumulate_rollout_stats)
trainer_state.rollout_stats = stats_tracker.rollout_stats
for hook in hooks:
    hook.on_rollout_end(trainer_state)

# After line 444 (epoch increment)
trainer_state.loss_stats = losses_stats
trainer_state.experience = experience
trainer_state.policy = policy
for hook in hooks:
    hook.on_epoch_end(trainer_state)
```

### Phase 1: Hook Infrastructure (2-3 hours)
**Goal**: Create hook base class and management system

#### Tasks:
1. Create `metta/rl/hooks/base.py` with `TrainerHook` base class
2. Create hook registry mechanism in trainer
3. Ensure hooks and losses can coexist
4. Create unit tests

#### Files to create:
- `metta/rl/hooks/base.py` - Base hook class
- `metta/rl/hooks/__init__.py` - Hook exports
- `tests/rl/hooks/test_base.py` - Base tests

#### Success criteria:
- [ ] Hooks can be registered alongside losses
- [ ] Callbacks are invoked in correct order
- [ ] Unit tests pass

### Phase 2: CheckpointHook (3-4 hours)
**Goal**: Extract checkpointing logic using existing patterns

#### Tasks:
1. Create `CheckpointHook` class extending `TrainerHook`
2. Move checkpoint loading to `on_new_training_run`
3. Move checkpoint saving to `on_epoch_end`
4. Store URIs in `TrainerState` for other hooks
5. Test checkpoint functionality

#### Implementation approach:
```python
class CheckpointHook(TrainerHook):
    def __init__(self, trainer_cfg, device, checkpoint_manager, wandb_run):
        super().__init__(trainer_cfg, device)
        self.checkpoint_manager = checkpoint_manager
        self.wandb_run = wandb_run
        self.checkpoint_interval = trainer_cfg.checkpoint.checkpoint_interval
        self.wandb_interval = trainer_cfg.checkpoint.wandb_checkpoint_interval
    
    def on_new_training_run(self, trainer_state: TrainerState) -> None:
        # Load checkpoint if resuming
        loaded_state = self.checkpoint_manager.load_trainer_state()
        if loaded_state:
            trainer_state.agent_step = loaded_state["agent_step"]
            trainer_state.epoch = loaded_state["epoch"]
            # Load policy handled separately in trainer setup
    
    def on_epoch_end(self, trainer_state: TrainerState) -> None:
        if should_run(trainer_state.epoch, self.checkpoint_interval):
            # Save checkpoint using data from trainer_state
            policy = trainer_state.policy
            optimizer = trainer_state.optimizer
            eval_scores = trainer_state.eval_scores
            
            wandb_uri = self.checkpoint_manager.save_agent(
                policy, trainer_state.epoch, metadata, 
                wandb_run=self.wandb_run if should_upload else None
            )
            trainer_state.latest_checkpoint_uri = ...
            trainer_state.latest_wandb_uri = wandb_uri
```

### Phase 3: MetricsHook (4-5 hours)
**Goal**: Consolidate all metrics, monitoring, and reporting

#### Tasks:
1. Create `MetricsHook` combining stats, wandb, monitoring, progress
2. Move stats processing to `on_epoch_end`
3. Move monitoring setup to `on_new_training_run`
4. Move progress logging to `on_epoch_end`
5. Use `TrainerState` to access needed data

#### Implementation approach:
```python
class MetricsHook(TrainerHook):
    def __init__(self, trainer_cfg, device, wandb_run, stats_client, timer):
        super().__init__(trainer_cfg, device)
        self.wandb_run = wandb_run
        self.stats_client = stats_client
        self.timer = timer
        self.memory_monitor = None
        self.system_monitor = None
    
    def on_new_training_run(self, trainer_state: TrainerState) -> None:
        if self.wandb_run:
            setup_wandb_metrics(self.wandb_run)
            log_model_parameters(trainer_state.policy, self.wandb_run)
        
        # Setup monitoring
        self.memory_monitor, self.system_monitor = setup_monitoring(
            policy=trainer_state.policy,
            experience=trainer_state.experience,
            timer=self.timer
        )
    
    def on_rollout_end(self, trainer_state: TrainerState) -> None:
        # Process rollout stats from trainer_state
        record_heartbeat()
    
    def on_epoch_end(self, trainer_state: TrainerState) -> None:
        # All data comes from trainer_state
        if self.wandb_run:
            process_stats(
                stats=trainer_state.rollout_stats,
                losses_stats=trainer_state.loss_stats,
                evals=trainer_state.eval_scores,
                # ... etc from trainer_state
            )
        
        # Log progress
        log_training_progress(
            epoch=trainer_state.epoch,
            agent_step=trainer_state.agent_step,
            # ... etc
        )
```

### Phase 4: EvaluationHook (3-4 hours)  
**Goal**: Extract evaluation logic

#### Tasks:
1. Create `EvaluationHook` for policy evaluation
2. Move evaluation logic to `on_epoch_end`
3. Access checkpoint URIs from `TrainerState`
4. Store scores back in `TrainerState`

#### Implementation approach:
```python
class EvaluationHook(TrainerHook):
    def __init__(self, trainer_cfg, device, eval_config, curriculum):
        super().__init__(trainer_cfg, device)
        self.eval_config = eval_config
        self.curriculum = curriculum
        
    def on_epoch_end(self, trainer_state: TrainerState) -> None:
        if not should_run(trainer_state.epoch, self.eval_config.evaluate_interval):
            return
            
        # Get checkpoint URI from trainer_state
        policy_uri = (trainer_state.latest_wandb_uri or 
                     trainer_state.latest_checkpoint_uri)
        
        if policy_uri:
            results = evaluate_policy(
                checkpoint_uri=policy_uri,
                simulations=self._get_simulations(),
                device=self.device,
                # ...
            )
            # Store results back for MetricsHook to use
            trainer_state.eval_scores = results.scores
```

### Phase 5: Integrate Hooks into Trainer (3-4 hours)
**Goal**: Wire hooks into existing trainer without breaking it

#### Tasks:
1. Add hook list to trainer
2. Call hooks at appropriate points
3. Populate `TrainerState` with needed data
4. Ensure backward compatibility
5. Test with all hooks enabled

#### Modified trainer structure:
```python
def train(...):
    # Create hooks based on configuration
    # NOTE: Order matters! EvaluationHook before MetricsHook
    hooks = []
    
    if checkpoint_manager:
        hooks.append(CheckpointHook(trainer_cfg, device, checkpoint_manager, wandb_run))
    
    if trainer_cfg.evaluation:
        hooks.append(EvaluationHook(trainer_cfg, device, trainer_cfg.evaluation, curriculum))
        
    if wandb_run or stats_client:
        hooks.append(MetricsHook(trainer_cfg, device, wandb_run, stats_client, timer))
    
    # Main training loop
    trainer_state = TrainerState(
        agent_step=agent_step,
        epoch=0,
        policy=policy,
        experience=experience,
        optimizer=optimizer,
        timer=timer,
        # ... etc
    )
    
    # Call hooks alongside losses
    for hook in hooks:
        hook.on_new_training_run(trainer_state)
    
    while agent_step < trainer_cfg.total_timesteps:
        # Rollout phase
        for hook in hooks:
            hook.on_rollout_start(trainer_state)
        
        # ... existing rollout logic with losses ...
        
        trainer_state.rollout_stats = stats_tracker.rollout_stats
        for hook in hooks:
            hook.on_rollout_end(trainer_state)
        
        # Training phase
        # ... existing training logic with losses ...
        
        # Epoch end
        trainer_state.loss_stats = losses_stats
        for hook in hooks:
            hook.on_epoch_end(trainer_state)
```

### Phase 6: Create Clean Trainer Class (4-5 hours)
**Goal**: New trainer class with only training logic

#### Tasks:
1. Create `metta/rl/clean_trainer.py`
2. Copy core training loop without external concerns
3. Support both losses and hooks
4. Remove all non-training imports
5. Test training works

#### Clean trainer structure:
```python
class Trainer:
    """Pure training engine with hook support."""
    
    def __init__(self, config: TrainerConfig, curriculum: Curriculum, policy: PolicyAgent):
        self.config = config
        self.curriculum = curriculum
        self.policy = policy
        self.hooks: list[TrainerHook] = []
        self.losses = config.losses.init_losses(...)  # Still support losses
        
    def add_hook(self, hook: TrainerHook) -> None:
        self.hooks.append(hook)
    
    def train(self) -> None:
        trainer_state = TrainerState(...)
        
        # Notify hooks of training start
        for hook in self.hooks:
            hook.on_new_training_run(trainer_state)
        
        while trainer_state.agent_step < self.config.total_timesteps:
            # Pure rollout logic
            self._rollout(trainer_state)
            
            # Pure training logic  
            self._train(trainer_state)
            
            # Update epoch
            trainer_state.epoch += 1
            
            # Notify hooks
            for hook in self.hooks:
                hook.on_epoch_end(trainer_state)
```

### Phase 7: Migration Strategy (2-3 hours)
**Goal**: Safely migrate to new system

#### Tasks:
1. Add feature flag to choose trainer implementation
2. Run both trainers in parallel for validation
3. Gradual rollout with monitoring
4. Update documentation
5. Deprecate old code once stable

## Key Design Decisions

### Hook Execution Order
- **Decision**: Hooks execute in registration order
- **Important**: EvaluationHook must be registered before MetricsHook
- **Rationale**: EvaluationHook sets `eval_scores` that MetricsHook consumes
- **Implementation**: Document order dependencies clearly

### Error Handling Strategy
- **Decision**: Hooks use try-catch with logging, training continues
- **Rationale**: Non-critical hooks shouldn't crash training
- **Implementation**:
  ```python
  for hook in hooks:
      try:
          hook.on_epoch_end(trainer_state)
      except Exception as e:
          logger.error(f"Hook {hook.__class__.__name__} failed: {e}", exc_info=True)
          if hook.critical:  # Optional: allow critical hooks
              raise
  ```

### Distributed Training
- **Decision**: Most hooks run only on rank 0
- **Exception**: CheckpointHook coordinates across ranks for loading
- **Implementation**:
  ```python
  if torch_dist_cfg.is_master or hook.run_on_all_ranks:
      hook.on_epoch_end(trainer_state)
  ```

## Key Design Decisions (Original)

### 1. Data Flow Through TrainerState
- **Decision**: All data passes through `TrainerState`, not explicit parameters
- **Rationale**: Maintains consistency with existing `BaseLoss` pattern
- **Benefits**: Extensible without breaking changes, clear state ownership

### 2. Hooks Complement Losses
- **Decision**: Hooks and losses coexist, not replace each other
- **Rationale**: Losses handle training logic, hooks handle external concerns
- **Benefits**: Clean separation, backward compatibility

### 3. Minimal New Callbacks
- **Decision**: Only add `on_rollout_end` and `on_epoch_end`
- **Rationale**: Minimize changes to existing system
- **Benefits**: Lower risk, easier integration

### 4. Coarse-Grained Hooks
- **Decision**: Three hooks (Checkpoint, Metrics, Evaluation)
- **Rationale**: Natural boundaries based on data coupling
- **Benefits**: Simpler than many fine-grained hooks

## Risk Mitigation

### Incremental Approach
1. Add callbacks first (Pre-Phase) - Low risk
2. Build hooks alongside existing code - No breaking changes
3. Test extensively before switching - Validation period
4. Feature flag for rollback - Safety net

### Testing Strategy
- Unit tests for each hook
- Integration tests with mock trainer
- Parallel testing (old vs new)
- Performance benchmarks

## Timeline

- Pre-Phase: 2-3 hours (add callbacks)
- Phase 1: 2-3 hours (infrastructure)
- Phase 2: 3-4 hours (CheckpointHook)
- Phase 3: 4-5 hours (MetricsHook)
- Phase 4: 3-4 hours (EvaluationHook)
- Phase 5: 3-4 hours (integration)
- Phase 6: 4-5 hours (clean trainer)
- Phase 7: 2-3 hours (migration)

**Total: 23-31 hours**

## Success Metrics

1. **Functionality preserved** - All features still work
2. **Performance maintained** - No regression in training speed
3. **Code cleanliness** - Trainer imports only training modules
4. **Testability improved** - Hooks can be mocked/stubbed
5. **Maintainability enhanced** - Clear separation of concerns

## Conclusion

This revised plan aligns with existing patterns while achieving the goal of extracting non-training concerns. By following `BaseLoss` patterns and using `TrainerState` for data flow, we maintain consistency and reduce implementation risk. The incremental approach allows validation at each step while maintaining backward compatibility.