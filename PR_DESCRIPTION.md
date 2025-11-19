# Refactor run_evaluation.py to use ParallelRollout with current_steps support

## Summary

Refactors `run_evaluation.py` to use the reusable `ParallelRollout` class for episode-level parallelization, replacing the manual `ThreadPoolExecutor` implementation. Enhances `ParallelRollout` to track and return `current_steps` per episode, enabling this integration.

## Motivation

Previously, `run_evaluation.py` implemented its own episode-level parallelization using `ThreadPoolExecutor` directly. Meanwhile, `ParallelRollout` was created as a reusable parallelization utility but lacked support for tracking episode step counts, which `run_evaluation.py` requires for generating evaluation results.

This PR consolidates parallelization logic by:
1. Enhancing `ParallelRollout` to capture and return `current_steps` per episode
2. Refactoring `run_evaluation.py` to use `ParallelRollout` instead of manual parallelization

## Technical Changes

### 1. Enhanced `ParallelRollout` Implementation

**File:** `packages/mettagrid/python/src/mettagrid/simulator/multi_episode/rollout.py`

- **Added `current_steps` field to `MultiEpisodeRolloutResult`**: New optional field `current_steps: list[int]` with default empty list for backward compatibility
- **Modified `_run_single_episode_rollout()`**: Now returns `current_step` as the 6th tuple element (extracted from `rollout._sim.current_step`)
- **Updated `multi_episode_rollout()`**: Captures `current_steps` for each episode and includes it in the result
- **Updated `ParallelRollout.__call__()`**: Captures `current_steps` in both parallel (ThreadPoolExecutor) and serial execution paths

**Backward Compatibility**: The `current_steps` field has a default value, so existing code that doesn't use it continues to work without modification.

### 2. Integration into `run_evaluation.py`

**File:** `packages/cogames/scripts/run_evaluation.py`

- **Replaced manual episode loop**: The `_run_case()` function now uses `ParallelRollout` instead of manually looping over episodes with `Rollout`
- **Simplified parallelization logic**: Removed direct `ThreadPoolExecutor` usage in favor of `ParallelRollout`'s built-in parallelization
- **Result conversion**: Converts `MultiEpisodeRolloutResult` back to `List[EvalResult]`, using `rollout_result.current_steps` instead of accessing `rollout._sim.current_step` directly

**Key Changes in `_run_case()`**:
- Before: Manual loop creating individual `Rollout` instances, accessing `rollout._sim.current_step` directly
- After: Single `ParallelRollout` call with `episodes=runs_per_case`, extracting step counts from `rollout_result.current_steps`

## Benefits

1. **Code Reuse**: Eliminates duplicate parallelization logic by using the shared `ParallelRollout` utility
2. **Consistency**: Both `cogames eval` and `run_evaluation.py` now use the same parallelization mechanism
3. **Maintainability**: Future improvements to `ParallelRollout` automatically benefit all consumers
4. **Performance**: Maintains the same parallelization performance while reducing code complexity

## Testing Considerations

- Existing tests for `MultiEpisodeRolloutResult` continue to pass (backward compatible default)
- `run_evaluation.py` behavior should be functionally equivalent to before
- Parallel execution paths in `ParallelRollout` now properly track step counts

## Files Changed

- `packages/mettagrid/python/src/mettagrid/simulator/multi_episode/rollout.py`
- `packages/cogames/scripts/run_evaluation.py`

## Related Work

This work was motivated by PR #8303, which introduced parallelization to `run_evaluation.py`. The question "can we make ParallelRollout so we can use this other places?" led to this refactoring to make the parallelization logic reusable.

