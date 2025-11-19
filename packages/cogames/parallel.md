# Parallelization Opportunities for `cogames eval`

This document outlines current parallelization implementation and future optimization opportunities for the `cogames eval` codepath.

## Current Implementation

### Agent-Level Parallelization (Implemented)

**Status**: ✅ Implemented
**Location**: `packages/mettagrid/python/src/mettagrid/simulator/rollout.py`

Agent-level parallelization runs each agent's policy computation in parallel subprocesses within each rollout step. This is enabled via the `--parallel-agents` flag.

**How it works**:
- Uses `ProcessPoolExecutor` to run `policy.step(obs)` calls in parallel
- Each agent's observation → action computation happens in a separate subprocess
- All agents synchronize before calling `sim.step()`
- Falls back to sequential execution if policies are not pickleable

**Limitations**:
- Policies must be pickleable (may not work with all policy types)
- High IPC overhead for fast policies (serialization cost may exceed computation time)
- Best suited for CPU-intensive policies with many agents (8+)

**Usage**:
```bash
cogames eval -m machina_1 -p random --cogs 4 --parallel-agents
```

## Future Optimization Opportunities

### Opportunity 1: Episode-Level Parallelization

**Scope**: Parallelize episodes within each mission
**Location**: `packages/mettagrid/python/src/mettagrid/simulator/multi_episode/rollout.py`
**Priority**: High
**Complexity**: Low

**Benefit**: Good speedup for single-mission evaluations with many episodes (10+)

**Implementation**:
- Modify `multi_episode_rollout()` to use `ThreadPoolExecutor` or `ProcessPoolExecutor`
- Run episodes concurrently, each with its own seed: `seed + episode_idx`
- Maintain episode order in results for deterministic output
- Aggregate results after all episodes complete

**When to implement**: When evaluating single mission with 10+ episodes

**Estimated speedup**: 2-4x for 4-8 CPU cores with 10+ episodes

---

### Opportunity 2: Mission-Level Parallelization

**Scope**: Parallelize across different missions
**Location**: `packages/cogames/src/cogames/evaluate.py`
**Priority**: High
**Complexity**: Low

**Benefit**: Excellent speedup when evaluating many missions (5+)

**Implementation**:
- Modify `evaluate()` to use `ThreadPoolExecutor` for mission-level parallelization
- Each mission loads its own policy instances (already independent)
- Aggregate results after all missions complete
- Update progress reporting to handle concurrent missions

**When to implement**: When evaluating 5+ missions simultaneously

**Estimated speedup**: Near-linear with number of CPU cores (e.g., 4x for 4 cores, 8x for 8 cores)

---

### Opportunity 3: Two-Level Parallelization

**Scope**: Combine episode and mission-level parallelization
**Priority**: Medium
**Complexity**: Medium

**Benefit**: Maximum speedup potential

**Implementation**:
- Nested parallelization: missions in outer pool, episodes in inner pool
- Control total parallelism via `--jobs` parameter to avoid oversubscription
- Requires careful resource management

**When to implement**: After both Opportunity 1 and 2 are implemented

**Estimated speedup**: Multiplicative (e.g., 4 missions × 4 episodes = 16x potential speedup)

---

### Opportunity 4: Thread-Based Agent Parallelization

**Scope**: Use `ThreadPoolExecutor` instead of `ProcessPoolExecutor` for agent-level
**Priority**: Low
**Complexity**: Low

**Benefit**: Lower overhead for I/O-bound or GIL-released policies

**Implementation**:
- Swap executor type based on policy characteristics
- Use threads for policies that release GIL (e.g., NumPy operations)
- Use processes for CPU-bound policies (current implementation)

**When to implement**: If profiling shows IPC overhead dominates for certain policy types

**Estimated speedup**: 10-20% reduction in overhead for suitable policies

---

### Opportunity 5: Batch Policy Execution

**Scope**: Batch multiple agent observations for vectorized policy computation
**Location**: `packages/mettagrid/python/src/mettagrid/simulator/rollout.py`
**Priority**: Medium
**Complexity**: Medium

**Benefit**: Leverage existing batch processing in policies (e.g., `step_batch()`)

**Implementation**:
- Collect all agent observations in a batch
- Call `policy.step_batch()` if available (some policies support this)
- Fall back to individual `step()` calls if not supported
- Requires policy API support for batch processing

**When to implement**: When policies support efficient batch processing

**Estimated speedup**: 2-3x for policies with batch support

---

### Opportunity 6: GPU-Aware Parallelization

**Scope**: Coordinate agent parallelism with GPU utilization
**Priority**: Low
**Complexity**: High

**Benefit**: Better GPU utilization for neural network policies

**Implementation**:
- Batch agent observations for GPU inference
- Coordinate CPU parallelism with GPU batch processing
- Requires understanding GPU scheduling and memory management

**When to implement**: When GPU is bottleneck for policy computation

**Estimated speedup**: 2-4x for GPU-bound policies

---

## Implementation Priority

1. **Mission-Level Parallelization** (Opportunity 2) - Highest impact, low complexity
2. **Episode-Level Parallelization** (Opportunity 1) - High impact, low complexity
3. **Two-Level Parallelization** (Opportunity 3) - High impact, medium complexity
4. **Batch Policy Execution** (Opportunity 5) - Medium impact, medium complexity
5. **Thread-Based Agent Parallelization** (Opportunity 4) - Low impact, low complexity
6. **GPU-Aware Parallelization** (Opportunity 6) - Low impact, high complexity

## Notes

- Current agent-level parallelization is experimental and may not provide speedup for all use cases
- Consider policy characteristics (CPU-bound vs I/O-bound, pickleability) when choosing parallelization strategy
- Monitor overhead vs speedup trade-offs - parallelization isn't always beneficial
- Test with various configurations (single/multiple missions, different agent counts, different policy types)

