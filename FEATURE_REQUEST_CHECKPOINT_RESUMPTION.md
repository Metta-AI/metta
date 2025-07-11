# Feature Request: Checkpoint Resumption for Sweep Jobs

**Type:** Enhancement
**Priority:** Medium
**Component:** Sweep System / Job Management
**Status:** Proposed

## Problem Statement

Currently, the sweep system does not check for unfinished training runs when jobs restart. Every sweep rollout creates a fresh run ID and starts training from scratch, even if a previous run was interrupted and has valid checkpoints available.

### Current Behavior
1. `sweep_rollout.sh` always calls `sweep_init.py` to generate a new run ID
2. `sweep_init.py` creates a unique run ID (e.g., `simple_sweep.r.12345`)
3. Training starts fresh in a new directory
4. Any interrupted runs with valid checkpoints are abandoned

### Impact
- **Resource Waste**: Interrupted runs lose all progress, wasting computational resources
- **Longer Sweep Times**: Sweeps take longer to complete due to repeated work
- **Cost Inefficiency**: Cloud compute costs increase from redundant training
- **User Frustration**: Manual intervention required to resume interrupted runs

## Proposed Solution

Implement **smart checkpoint resumption** that safely detects and resumes incomplete training runs while maintaining distributed training safety.

### Core Features

#### 1. **Incomplete Run Detection**
- Add `.incomplete` marker files to track active training runs
- Check for existing incomplete runs before generating new run IDs
- Validate checkpoint integrity before resuming

#### 2. **Safe Resumption Logic**
- Leverage existing `load_checkpoint()` infrastructure for distributed safety
- Maintain WandB run continuity using existing `resume=True` functionality
- Ensure all distributed ranks synchronize properly during resumption

#### 3. **Fallback to Fresh Start**
- If checkpoint is corrupted or invalid, automatically start fresh run
- Preserve existing behavior as fallback for maximum reliability

## Technical Implementation

### 1. **Modify `devops/sweep_rollout.sh`**

Add resumption check before sweep initialization:

```bash
# Check for incomplete runs before generating new run
INCOMPLETE_RUN=$(find "$DATA_DIR/sweep/$sweep_run/runs" -name "*.incomplete" -type f 2>/dev/null | head -1)

if [ -n "$INCOMPLETE_RUN" ]; then
  run_id=$(basename "$INCOMPLETE_RUN" .incomplete)
  echo "[RESUME] Found incomplete run: $run_id"

  # Verify checkpoint validity
  if [ -f "$DATA_DIR/sweep/$sweep_run/runs/$run_id/checkpoints/trainer_checkpoint.json" ]; then
    echo "[RESUME] Valid checkpoint found, resuming training..."
    # Skip sweep_init.py - use existing run configuration
  else
    echo "[RESUME] Invalid checkpoint, starting fresh..."
    rm -f "$INCOMPLETE_RUN"
    INCOMPLETE_RUN=""
  fi
fi

# Only initialize new run if no valid incomplete run found
if [ -z "$INCOMPLETE_RUN" ]; then
  # ... existing sweep_init.py logic ...
fi
```

### 2. **Add Run State Tracking in `devops/train.sh`**

Track training lifecycle with markers:

```bash
# Mark run as incomplete at start
echo "Training started at $(date)" > "$run_dir/$run_id.incomplete"

# Remove marker on successful completion
trap 'rm -f "$run_dir/$run_id.incomplete"' EXIT
```

### 3. **Leverage Existing Infrastructure**

**Checkpoint Loading**: Already handles distributed resumption safely via `metta/api.py:load_checkpoint()`

**WandB Resumption**: Already supports run resumption via `wandb.init(resume=True)`

**Distributed Coordination**: Existing barrier synchronization ensures consistent state

### 4. **Configuration Preservation**

Store run configuration for resumption:

```bash
# In sweep_rollout.sh - preserve config for resumption
if [ -n "$INCOMPLETE_RUN" ]; then
  # Load existing configuration instead of generating new
  cp "$DATA_DIR/sweep/$sweep_run/runs/$run_id/train_config_overrides.yaml" "$DIST_CFG_PATH"
fi
```

## Benefits

### Resource Efficiency
- **Computational Savings**: Resume from last checkpoint instead of starting over
- **Time Savings**: Reduce sweep completion time significantly
- **Cost Reduction**: Lower cloud compute costs from eliminated redundant work

### Reliability
- **Fault Tolerance**: Automatic recovery from job interruptions
- **Distributed Safety**: Leverages existing checkpoint coordination
- **Backward Compatible**: Existing sweeps continue working unchanged

### User Experience
- **Transparent Operation**: Resumption happens automatically
- **Progress Preservation**: No loss of training progress
- **Reduced Manual Intervention**: No need to manually resume interrupted runs

## Implementation Strategy

### Phase 1: Core Resumption Logic
1. Add incomplete run detection to `sweep_rollout.sh`
2. Implement run state tracking in `train.sh`
3. Add configuration preservation logic

### Phase 2: Enhanced Validation
1. Add checkpoint integrity validation
2. Implement robust fallback mechanisms
3. Add logging and monitoring for resumption events

### Phase 3: Optimization
1. Add resumption metrics to WandB
2. Optimize checkpoint validation performance
3. Add user controls for resumption behavior

## Testing Strategy

### Unit Tests
- Test incomplete run detection logic
- Validate checkpoint integrity checking
- Test configuration preservation

### Integration Tests
- Test end-to-end sweep resumption
- Verify distributed training resumption
- Test WandB run continuity

### Failure Mode Tests
- Test behavior with corrupted checkpoints
- Verify fallback to fresh start
- Test resumption with different hardware configurations

## Risk Assessment

### Low Risk
- **Minimal Code Changes**: Leverages existing infrastructure
- **Backward Compatible**: Existing behavior preserved as fallback
- **Well-Tested Components**: Uses proven checkpoint and WandB resumption

### Mitigation Strategies
- **Checkpoint Validation**: Verify integrity before resumption
- **Graceful Fallback**: Default to fresh start on any resumption failure
- **Comprehensive Testing**: Test all failure modes thoroughly

## Success Metrics

### Performance Metrics
- **Resumption Success Rate**: % of interrupted runs successfully resumed
- **Time Savings**: Reduction in sweep completion time
- **Resource Utilization**: Decrease in wasted computational resources

### Reliability Metrics
- **Failure Rate**: Frequency of resumption failures
- **Fallback Rate**: How often system falls back to fresh start
- **User Satisfaction**: Feedback on automatic resumption functionality

## Alternative Approaches Considered

### Manual Resumption
- **Pros**: User control, simple implementation
- **Cons**: Requires manual intervention, error-prone

### Database-Based State Tracking
- **Pros**: More robust state management
- **Cons**: Adds complexity, requires infrastructure changes

### Checkpoint-Based Detection Only
- **Pros**: Simpler implementation
- **Cons**: Less reliable, no run lifecycle tracking

## Related Work

- **Existing Checkpoint System**: `metta/api.py:load_checkpoint()`
- **WandB Resumption**: `wandb.init(resume=True)`
- **Distributed Coordination**: Existing barrier synchronization
- **Sweep Infrastructure**: Current sweep rollout system

## Files to Modify

1. **`devops/sweep_rollout.sh`** - Add resumption detection logic
2. **`devops/train.sh`** - Add run state tracking
3. **`tools/sweep_init.py`** - Add configuration preservation (optional)
4. **`tests/sweep/test_resumption.py`** - Add comprehensive tests

## Acceptance Criteria

- [ ] Interrupted sweep runs automatically resume from last checkpoint
- [ ] Resumption works correctly in distributed training environments
- [ ] WandB run continuity is maintained during resumption
- [ ] System gracefully falls back to fresh start on resumption failure
- [ ] Existing sweep behavior is preserved for new runs
- [ ] Comprehensive test coverage for all resumption scenarios
- [ ] Performance metrics show measurable resource savings

---

**Requested by**: User
**Date**: 2024-12-28
**Estimated Effort**: 2-3 developer days
**Dependencies**: None (leverages existing infrastructure)
