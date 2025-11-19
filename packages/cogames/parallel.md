# Parallelization Opportunities for `cogames eval`

This document outlines current parallelization implementation and future optimization opportunities for the `cogames eval` codepath.

## Current Implementation

### Agent-Level Parallelization (Implemented)

**Status**: ✅ Implemented
**Location**: `packages/mettagrid/python/src/mettagrid/policy/subprocess_wrapper.py`

Agent-level parallelization runs each agent's policy computation in parallel subprocesses within each rollout step. This is enabled via the `--parallel-policy` flag.

**How it works**:
- `PerAgentSubprocessWrapper` wraps individual `AgentPolicy` instances transparently
- Wrapping happens in `evaluate()` after policy loading
- Uses shared `ProcessPoolExecutor` to run `policy.step(obs)` calls in parallel
- Each agent's observation → action computation happens in a separate subprocess
- All agents synchronize before calling `sim.step()`
- Falls back to sequential execution if policies are not pickleable
- Process pool is cleaned up explicitly in `evaluate()`

**Architecture Benefits**:
- **Transparent to Rollout**: `Rollout.step()` doesn't know about parallelization
- **Flexible**: Can use different wrapper strategies per policy type
- **Composable**: Can combine with episode-level parallelization (`--jobs`)

**Limitations**:
- Policies must be pickleable (may not work with all policy types)
- High IPC overhead for fast policies (serialization cost may exceed computation time)
- Best suited for CPU-intensive policies with many agents (8+)

**Usage**:
```bash
cogames eval -m machina_1 -p random --cogs 4 --parallel-policy
```

### Episode-Level Parallelization (Implemented)

**Status**: ✅ Implemented
**Location**: `packages/mettagrid/python/src/mettagrid/simulator/multi_episode/rollout.py`

Episode-level parallelization runs multiple episodes concurrently within each mission. This is enabled via the `--jobs` parameter.

**How it works**:
- Uses `ThreadPoolExecutor` to run episodes in parallel
- Each episode gets its own seed: `seed + episode_idx`
- Results are collected and ordered deterministically

**Usage**:
```bash
cogames eval -m machina_1 -p random --episodes 10 --jobs 4
```

## Future Optimization Opportunities

### Opportunity 1: Mission-Level Parallelization

**Scope**: Parallelize across different missions
**Location**: `packages/cogames/src/cogames/evaluate.py`
**Priority**: High
**Complexity**: Low

**Benefit**: Excellent speedup when evaluating many missions (5+)

**Implementation**:
- Modify `evaluate()` to use `ThreadPoolExecutor` for mission-level parallelization
- Each mission loads its own policy instances (already independent)
- Each mission can have its own process pool for `--parallel-policy`
- Aggregate results after all missions complete
- Update progress reporting to handle concurrent missions

**When to implement**: When evaluating 5+ missions simultaneously

**Estimated speedup**: Near-linear with number of CPU cores (e.g., 4x for 4 cores, 8x for 8 cores)

**Note**: This works seamlessly with the wrapper architecture - each mission's wrapped policies are independent.

---

### Opportunity 2: Three-Level Parallelization

**Scope**: Combine mission, episode, and agent-level parallelization
**Priority**: Medium
**Complexity**: Medium

**Benefit**: Maximum speedup potential

**Implementation**:
- Nested parallelization: missions in outer pool, episodes in middle pool, agents in inner pool
- Control total parallelism via `--jobs` parameter to avoid oversubscription
- Requires careful resource management to avoid oversubscription
- Each level can be independently enabled/disabled

**When to implement**: After Opportunity 1 (Mission-Level) is implemented

**Estimated speedup**: Multiplicative (e.g., 4 missions × 4 episodes × 4 agents = 64x potential speedup, limited by CPU cores)

---

### Opportunity 3: Adaptive Wrapper Strategy

**Scope**: Choose wrapper type based on policy characteristics
**Priority**: Medium
**Complexity**: Medium

**Benefit**: Optimal parallelization strategy per policy type

**Implementation**:
- Create `PerAgentThreadWrapper` for GIL-released policies (NumPy, I/O-bound)
- Create `PerAgentProcessWrapper` for CPU-bound policies (current implementation)
- Auto-detect policy characteristics or allow manual selection
- Wrapper selection happens transparently in `evaluate()`

**When to implement**: When profiling shows different policies benefit from different strategies

**Estimated speedup**: 10-30% improvement by matching strategy to policy type

**New Opportunity Opened by Wrapper Architecture**: ✅ The wrapper pattern makes this easy - just swap wrapper classes!

---

### Opportunity 4: Batch-Aware Wrapper

**Scope**: Batch multiple agent observations for vectorized policy computation
**Location**: `packages/mettagrid/python/src/mettagrid/policy/subprocess_wrapper.py`
**Priority**: Medium
**Complexity**: Medium

**Benefit**: Leverage existing batch processing in policies (e.g., `step_batch()`)

**Implementation**:
- Create `PerAgentBatchWrapper` that collects observations across agents
- When all agents in a step have called `step()`, batch them together
- Call underlying policy's batch method if available
- Falls back to individual `step()` calls if not supported
- Works transparently with `Rollout.step()`

**When to implement**: When policies support efficient batch processing

**Estimated speedup**: 2-3x for policies with batch support

**New Opportunity Opened by Wrapper Architecture**: ✅ The wrapper can intercept and batch calls before they reach the policy!

---

### Opportunity 5: GPU-Aware Wrapper

**Scope**: Coordinate agent parallelism with GPU utilization
**Priority**: Medium
**Complexity**: Medium-High

**Benefit**: Better GPU utilization for neural network policies

**Implementation**:
- Create `PerAgentGPUWrapper` that batches observations for GPU inference
- Collect observations from multiple agents/steps before GPU call
- Coordinate CPU parallelism with GPU batch processing
- Manages GPU context and memory efficiently
- Works transparently with `Rollout.step()`

**When to implement**: When GPU is bottleneck for policy computation

**Estimated speedup**: 2-4x for GPU-bound policies

**New Opportunity Opened by Wrapper Architecture**: ✅ GPU batching can be handled entirely in the wrapper layer!

---

## New Opportunities Opened by Wrapper Architecture

The wrapper-based architecture opens several new parallelization opportunities:

1. **Per-Policy-Type Wrappers**: Different wrapper strategies for different policy types
   - Thread wrapper for I/O-bound policies
   - Process wrapper for CPU-bound policies
   - GPU wrapper for neural network policies
   - Batch wrapper for policies with batch support

2. **Dynamic Strategy Selection**: Wrapper can choose strategy at runtime based on:
   - Policy type detection
   - Current system load
   - Policy state (stateful vs stateless)

3. **Hybrid Parallelization**: Mix different wrapper types in the same evaluation
   - Some agents use threads, others use processes
   - All transparent to `Rollout`

4. **Resource-Aware Wrappers**: Wrappers can manage their own resources
   - GPU context per wrapper
   - Process pool per wrapper
   - Memory management

## Implementation Priority

1. **Mission-Level Parallelization** (Opportunity 1) - Highest impact, low complexity
2. **Three-Level Parallelization** (Opportunity 2) - High impact, medium complexity
3. **Adaptive Wrapper Strategy** (Opportunity 3) - Medium impact, medium complexity
4. **Batch-Aware Wrapper** (Opportunity 4) - Medium impact, medium complexity
5. **GPU-Aware Wrapper** (Opportunity 5) - Medium impact, medium-high complexity

## Notes

- Current agent-level parallelization is experimental and may not provide speedup for all use cases
- Consider policy characteristics (CPU-bound vs I/O-bound, pickleability) when choosing parallelization strategy
- Monitor overhead vs speedup trade-offs - parallelization isn't always beneficial
- Test with various configurations (single/multiple missions, different agent counts, different policy types)

## Architecture Benefits

The wrapper-based architecture provides several key advantages:

1. **Separation of Concerns**: Parallelization logic is isolated in wrappers, not scattered in `Rollout`
2. **Composability**: Can combine multiple parallelization strategies (mission + episode + agent)
3. **Flexibility**: Easy to add new wrapper types for different strategies
4. **Transparency**: `Rollout` doesn't need to know about parallelization
5. **Testability**: Wrappers can be tested independently
6. **Future-Proof**: New parallelization strategies can be added without changing core code

