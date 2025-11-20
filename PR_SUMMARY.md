# PR Summary: Agent-Level Parallelization with Wrapper Architecture

## Overview

This PR refactors agent-level parallelization for `cogames eval` to use a wrapper-based architecture, making parallelization transparent to the core rollout code and enabling new parallelization strategies.

## Key Changes

### Architecture Refactoring

**Before**: Parallelization logic was embedded in `Rollout.step()`, making it tightly coupled and hard to extend.

**After**: Parallelization is handled by `PerAgentSubprocessWrapper` that wraps `AgentPolicy` instances transparently.

### New Components

1. **`PerAgentSubprocessWrapper`** (`packages/mettagrid/python/src/mettagrid/policy/subprocess_wrapper.py`)
   - Wraps `AgentPolicy` instances to execute policy steps in subprocesses
   - Handles observation/action serialization transparently
   - Falls back to sequential execution if policies aren't pickleable
   - Shared process pool across all agents

2. **`WrappedMultiAgentPolicy`** (in `evaluate.py`)
   - Intercepts `agent_policy()` calls to wrap policies
   - Created when `--parallel-policy` flag is set
   - Manages wrapper lifecycle

### Modified Components

1. **`Rollout.step()`** - Simplified
   - Removed all parallelization logic
   - Now just calls `policy.step()` normally
   - Wrapper handles parallelization transparently

2. **`evaluate()`** - Enhanced
   - Creates shared `ProcessPoolExecutor` when `--parallel-policy` is set
   - Wraps policies after loading
   - Explicit cleanup in `finally` block

3. **`multi_episode_rollout()`** - Added episode-level parallelization
   - New `--jobs` parameter for parallel episodes
   - Uses `ThreadPoolExecutor` for episode parallelization
   - Maintains deterministic ordering

### CLI Changes

- **Renamed**: `--parallel-agents` → `--parallel-policy`
- **Added**: `--jobs` / `-j` parameter for episode-level parallelization
  - `--jobs 0` (default): Sequential episodes
  - `--jobs N`: Run N episodes in parallel

## Usage Examples

```bash
# Agent-level parallelization
cogames eval -m machina_1 -p random --cogs 4 --parallel-policy

# Episode-level parallelization
cogames eval -m machina_1 -p random --episodes 10 --jobs 4

# Combined (three-level parallelization)
cogames eval -m machina_1 -p random --cogs 4 --episodes 10 --jobs 4 --parallel-policy
```

## Architecture Benefits

1. **Transparency**: `Rollout` doesn't know about parallelization
2. **Composability**: Can combine mission/episode/agent parallelization
3. **Extensibility**: Easy to add new wrapper types (thread, GPU, batch)
4. **Separation of Concerns**: Parallelization logic isolated in wrappers
5. **Future-Proof**: New strategies don't require core code changes

## New Opportunities Opened

The wrapper architecture enables several new parallelization opportunities:

1. **Adaptive Wrapper Strategy**: Different wrapper types per policy (thread/process/GPU)
2. **Batch-Aware Wrapper**: Intercept and batch calls before reaching policy
3. **GPU-Aware Wrapper**: Handle GPU batching entirely in wrapper layer
4. **Hybrid Parallelization**: Mix different wrapper types in same evaluation
5. **Dynamic Strategy Selection**: Choose strategy at runtime based on policy/system characteristics

See `packages/cogames/parallel.md` for detailed documentation.

## Files Changed

- **New**: `packages/mettagrid/python/src/mettagrid/policy/subprocess_wrapper.py`
- **Modified**: `packages/mettagrid/python/src/mettagrid/simulator/rollout.py` (simplified)
- **Modified**: `packages/mettagrid/python/src/mettagrid/simulator/multi_episode/rollout.py` (added episode parallelization)
- **Modified**: `packages/cogames/src/cogames/evaluate.py` (wrapping + cleanup)
- **Modified**: `packages/cogames/src/cogames/main.py` (CLI flags)
- **Updated**: `packages/cogames/parallel.md` (documentation)

## Bug Fixes

### Stateful Policy State Preservation

**Issue**: The subprocess wrapper was pickling the policy once in `__init__` and reusing the same bytes for every step submission. This caused stateful policies (RNNs, exploration decay, cooldown counters) to lose their state between steps, producing incorrect behavior compared to sequential execution.

**Fix**: 
- Worker function now re-pickles the policy after each step to capture state changes
- Returns updated policy state from subprocess
- Wrapper syncs state back to wrapped policy by copying attributes
- Falls back to sequential execution if state can't be pickled

This ensures stateful policies maintain correct state across steps when using `--parallel-policy`.

## Testing

- ✅ Basic evaluation works
- ✅ Episode-level parallelization works (`--jobs`)
- ✅ Agent-level parallelization works (`--parallel-policy`)
- ✅ Stateful policy state is preserved across steps
- ✅ All linting passes
- ✅ Backward compatible (sequential by default)

## Breaking Changes

- `--parallel-agents` flag renamed to `--parallel-policy`

## Next Steps

Future PRs can implement:
1. Mission-level parallelization (Opportunity 1 in parallel.md)
2. Adaptive wrapper strategies (thread/GPU/batch wrappers)
3. Three-level parallelization optimization

