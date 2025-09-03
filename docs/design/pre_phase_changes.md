# Pre-Phase Implementation Changes

## Analysis of Existing Callbacks

After reviewing the callback timing and existing alternatives, we confirmed:

### No Existing Alternative for `on_rollout_end`
- **Checked for**: Any callback between rollout and training phases
- **Found**: Nothing - there's a gap in the existing callback system
- **Existing callbacks**: 
  - `on_rollout_start` (before rollout)
  - `on_train_phase_end` (after training)
  - Nothing in between where we need it

### Kept `on_epoch_end` Separate
- **Not replaced by**: `on_train_phase_end`
- **Rationale**: Different timing and purpose:
  - `on_train_phase_end`: Inside training timer, before epoch increment
  - `on_epoch_end`: Outside training timer, after epoch increment, with loss stats available

## Final Callback Additions

Only 2 new callbacks were added to minimize footprint:

1. **`on_rollout_end`** - Called between rollout and training phases
2. **`on_epoch_end`** - Called after epoch completes and increments

Both follow the existing `BaseLoss` pattern with `TrainerState` as the only parameter.