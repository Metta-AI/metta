# Evaluation Architecture: Per-Agent Subprocess Model

## Overview

The `cogames eval` command supports parallelizing policy evaluation through a per-agent subprocess architecture. This document describes the systems design and demonstrates how the architecture maintains evaluation protocol integrity.

## Table of Contents

1. [Systems Design](#systems-design)
2. [Cybernetics: Protocol Integrity](#cybernetics-protocol-integrity)
3. [Architecture Details](#architecture-details)
4. [Correctness Guarantees](#correctness-guarantees)
5. [Performance Characteristics](#performance-characteristics)

---

## Systems Design

### Architecture Overview

The evaluation system supports two levels of parallelization:

1. **Episode-level parallelization** (`--jobs`): Multiple episodes run concurrently using thread pools
2. **Agent-level parallelization** (`--parallel-policy`): Each agent's policy runs in its own persistent subprocess

```
┌─────────────────────────────────────────────────────────────┐
│                    Evaluation Orchestrator                   │
│  (evaluate() in cogames/evaluate.py)                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
   ┌────▼────┐                  ┌─────▼─────┐
   │ Episode │                  │  Episode  │
   │ Worker  │                  │  Worker   │
   │ (Thread)│                  │ (Thread)  │
   └────┬────┘                  └─────┬─────┘
        │                             │
        │  For each agent:            │
        │  ┌──────────────────────┐   │
        │  │ AgentPolicy          │   │
        │  │ (wrapped in          │   │
        │  │  subprocess)         │   │
        │  └──────┬───────────────┘   │
        │         │                    │
        │  ┌──────▼───────────────┐   │
        │  │ Subprocess           │   │
        │  │ - Policy instance    │   │
        │  │ - Persistent state   │   │
        │  │ - Queue comms        │   │
        │  └──────────────────────┘   │
        │                             │
```

### Component Hierarchy

```
MultiAgentPolicy (main process)
    ↓ agent_policy(agent_id)
AgentPolicy (main process)
    ↓ wrap_agent_policy_in_subprocess()
PerAgentSubprocessPolicy (main process)
    ├─ Queue (request/response)
    └─ Process (subprocess)
        └─ AgentPolicy (subprocess)
            └─ Policy state (persistent)
```

### Communication Pattern

The subprocess model uses a request-response pattern over `multiprocessing.Queue`:

1. **Request**: Main process sends `(step_id, pickled_observation)` to subprocess
2. **Computation**: Subprocess unpickles observation, calls `agent_policy.step(obs)`, pickles action
3. **Response**: Subprocess sends `(step_id, pickled_action)` back to main process
4. **Ordering**: Step IDs ensure responses match requests even if processing order varies

### Lifecycle Management

**Initialization**:
- Each `AgentPolicy` instance gets wrapped in `PerAgentSubprocessPolicy`
- Wrapper spawns a dedicated `Process` for that agent
- Subprocess initializes its own `MultiAgentPolicy` and `AgentPolicy` instances
- Subprocess calls `agent_policy.reset()` to initialize state

**Execution**:
- Each `step()` call serializes observation, sends to subprocess, waits for action
- State persists within the subprocess across steps
- Reset operations are forwarded to subprocess

**Cleanup**:
- After episode completion, main process calls `shutdown()` on each wrapped policy
- Shutdown sends termination signal to subprocess
- Subprocess joins with timeout, then terminates if still alive
- `__del__` methods provide fallback cleanup

---

## Cybernetics: Protocol Integrity

### Evaluation Protocol Requirements

An evaluation protocol must maintain:

1. **Determinism**: Given the same seed and configuration, results are reproducible
2. **State Consistency**: Agent state evolves correctly across steps within an episode
3. **Isolation**: Agent policies do not interfere with each other
4. **Correctness**: Actions are computed from the correct observations at the correct time
5. **Completeness**: All episodes complete and all metrics are collected

### Integrity Guarantees

#### 1. Determinism Preservation

**Guarantee**: The subprocess model preserves determinism.

**Mechanism**:
- Episode seeds are generated deterministically from the base seed using a separate RNG
- Each episode uses its own seed, ensuring independent but reproducible randomness
- Subprocess initialization is deterministic (same policy class, same data path, same env config)
- Observation serialization/deserialization is deterministic (same observation → same bytes)

**Verification**:
```python
# Same seed → same results
result1 = evaluate(..., seed=42, parallel_policy=True)
result2 = evaluate(..., seed=42, parallel_policy=True)
assert result1 == result2  # Deterministic
```

#### 2. State Consistency

**Guarantee**: Agent state evolves correctly within each episode.

**Mechanism**:
- Each agent's policy runs in a **persistent subprocess** that maintains state across steps
- State is never serialized/deserialized between steps (only observations and actions cross process boundaries)
- `reset()` operations are forwarded to subprocess, ensuring clean state between episodes
- Stateful policies (LSTMs, scripted agents with memory) maintain their internal state correctly

**Verification**:
```python
# Stateful policy maintains state across steps
policy = LSTMPolicy(...)
wrapped = wrap_agent_policy_in_subprocess(policy, ...)

obs1 = get_observation()
action1 = wrapped.step(obs1)  # Uses initial state

obs2 = get_observation()
action2 = wrapped.step(obs2)  # Uses updated state from step 1

# State evolution is correct because it's maintained in-process
```

#### 3. Isolation

**Guarantee**: Agent policies are isolated from each other.

**Mechanism**:
- Each agent runs in its own separate process with its own memory space
- No shared state between agents (except through the environment, which is intentional)
- Process boundaries prevent accidental state leakage
- Queue-based communication ensures no direct memory sharing

**Verification**:
```python
# Agents don't interfere with each other
agent0_policy = wrap_agent_policy_in_subprocess(policy0, ...)
agent1_policy = wrap_agent_policy_in_subprocess(policy1, ...)

# Each maintains independent state
action0 = agent0_policy.step(obs0)
action1 = agent1_policy.step(obs1)
# No cross-contamination possible
```

#### 4. Correctness

**Guarantee**: Actions are computed from the correct observations at the correct time.

**Mechanism**:
- Step IDs ensure request-response matching (even if processing order varies)
- Observation serialization preserves all information needed for action computation
- Action deserialization preserves the action object structure
- Synchronous communication (blocking queue operations) ensures step ordering

**Verification**:
```python
# Request-response matching
step_id = 0
request_queue.put((step_id, pickle.dumps(obs)))
response = response_queue.get()  # Blocks until response
response_step_id, action_data = response
assert response_step_id == step_id  # Correct matching
```

#### 5. Completeness

**Guarantee**: All episodes complete and all metrics are collected.

**Mechanism**:
- Cleanup is guaranteed via `try...finally` blocks and explicit `shutdown()` calls
- Subprocess termination is handled with timeouts and force-termination fallbacks
- Episode results are collected regardless of whether subprocesses are used
- Error handling ensures partial results are still returned

**Verification**:
```python
# Cleanup always happens
try:
    results = run_episodes(...)
finally:
    for wrapped_policy in wrapped_policies:
        wrapped_policy.shutdown()  # Always called
```

---

## Architecture Details

### Wrapping at AgentPolicy Level

The architecture wraps at the `AgentPolicy` level rather than the `MultiAgentPolicy` level:

**Benefits**:
- **Modularity**: Each agent's policy is self-contained
- **Explicitness**: Wrapping happens where policies are actually used
- **Flexibility**: Can wrap individual agents selectively
- **Lifecycle**: Each agent manages its own subprocess lifecycle

**Implementation**:
```python
# In _run_single_episode()
for agent_id in range(num_agents):
    base_agent_policy = policies[policy_idx].agent_policy(agent_id)

    if parallel_policy:
        wrapped_policy = wrap_agent_policy_in_subprocess(
            base_agent_policy,
            policies[policy_idx],  # For extracting class/data path
            agent_id,
        )
        agent_policies.append(wrapped_policy)
    else:
        agent_policies.append(base_agent_policy)
```

### Subprocess Worker Function

Each subprocess runs a worker function that:
1. Initializes the policy from class path and data path
2. Gets the specific `AgentPolicy` for the agent ID
3. Resets the policy to initialize state
4. Enters a loop:
   - Receives observation requests
   - Computes actions
   - Sends action responses
   - Handles reset and shutdown signals

```python
def _agent_worker_process(
    agent_id: int,
    policy_class_path: str,
    policy_data_path: Optional[str],
    env_info_dict: dict,
    request_queue: Queue,
    response_queue: Queue,
) -> None:
    # Initialize policy in subprocess
    env_info = PolicyEnvInterface(**env_info_dict)
    policy_spec = PolicySpec(class_path=policy_class_path, data_path=policy_data_path)
    multi_agent_policy = initialize_or_load_policy(env_info, policy_spec)
    agent_policy = multi_agent_policy.agent_policy(agent_id)
    agent_policy.reset()

    # Main loop
    while True:
        request = request_queue.get()
        if request is None:  # Shutdown
            break
        if request == "reset":
            agent_policy.reset()
            response_queue.put("reset_ok")
            continue

        step_id, obs_data = request
        obs = pickle.loads(obs_data)
        action = agent_policy.step(obs)
        action_data = pickle.dumps(action)
        response_queue.put((step_id, action_data))
```

### Serialization Requirements

**Observations** (`AgentObservation`):
- Must be pickle-serializable
- Contains all information needed for action computation
- Preserves agent-specific context

**Actions** (`Action`):
- Must be pickle-serializable
- Preserves action type and parameters
- Compatible with environment's action space

**Policy Configuration**:
- Policy class path (string) - used to recreate policy in subprocess
- Policy data path (optional string) - used to load weights
- Environment info (dict) - used to recreate `PolicyEnvInterface`

---

## Correctness Guarantees

### Invariants

1. **State Invariant**: Agent state in subprocess matches what it would be in main process
   - State is never serialized between steps
   - State evolution follows the same logic as in-process execution

2. **Ordering Invariant**: Actions are computed in the correct order
   - Step IDs ensure request-response matching
   - Synchronous communication preserves step ordering

3. **Isolation Invariant**: Agents cannot observe or modify each other's state
   - Process boundaries enforce memory isolation
   - No shared mutable state between agents

4. **Determinism Invariant**: Same inputs produce same outputs
   - Deterministic seed generation
   - Deterministic serialization
   - Deterministic subprocess initialization

### Failure Modes and Mitigations

**Subprocess Crash**:
- Detection: Exception raised when deserializing response
- Mitigation: Error propagated to main process, episode marked as failed
- Recovery: Subprocess is terminated, new one could be spawned (not currently implemented)

**Deadlock**:
- Detection: Timeout on queue operations (not currently implemented)
- Mitigation: Process termination with timeout
- Prevention: Synchronous request-response pattern prevents deadlock

**State Corruption**:
- Detection: Not directly detectable
- Mitigation: Process isolation prevents cross-agent corruption
- Prevention: State never crosses process boundaries

**Resource Exhaustion**:
- Detection: Process creation failures
- Mitigation: Graceful degradation (fall back to in-process execution)
- Prevention: Limit number of concurrent subprocesses (not currently implemented)

---

## Performance Characteristics

### Overhead

**Per-Step Overhead**:
- Serialization: ~0.1-1ms per observation/action (depends on observation size)
- Queue operations: ~0.01-0.1ms per send/receive
- Process context switching: ~0.01-0.1ms per step
- **Total**: ~0.1-1.2ms per step per agent

**Per-Episode Overhead**:
- Subprocess creation: ~10-100ms per agent
- Policy initialization: ~10-1000ms per agent (depends on policy size)
- Cleanup: ~10-100ms per agent
- **Total**: ~30-1200ms per episode per agent

### Scalability

**Memory**:
- Each subprocess has its own memory space
- Policy weights are duplicated in each subprocess
- Memory overhead: ~(policy_size × num_agents) bytes

**CPU**:
- Parallelization allows true parallelism (not just concurrency)
- CPU-bound policies benefit from multi-core systems
- I/O-bound policies may see less benefit

**Tradeoffs**:
- **Pros**: True parallelism, state isolation, supports stateful policies
- **Cons**: Higher memory overhead, serialization overhead, process management complexity

### When to Use

**Use `--parallel-policy` when**:
- Policies are CPU-bound (neural networks, complex scripted agents)
- Stateful policies need state persistence (LSTMs, scripted agents with memory)
- Multiple CPU cores are available
- Memory overhead is acceptable

**Don't use `--parallel-policy` when**:
- Policies are very fast (simple random policies)
- Memory is constrained
- Single-core systems
- Stateless policies with minimal computation

---

## Testing and Validation

### Unit Tests

Tests should verify:
- Determinism: Same seed → same results
- State consistency: Stateful policies maintain state correctly
- Isolation: Agents don't interfere with each other
- Correctness: Actions match in-process execution
- Cleanup: Subprocesses are properly terminated

### Integration Tests

Tests should verify:
- End-to-end evaluation produces correct results
- Parallel and serial execution produce equivalent results
- Error handling works correctly
- Resource cleanup happens properly

### Performance Tests

Tests should measure:
- Overhead per step
- Overhead per episode
- Memory usage
- Scalability with number of agents

---

## Future Improvements

1. **Process Pooling**: Reuse subprocesses across episodes to reduce creation overhead
2. **Async Communication**: Use async queues for non-blocking communication
3. **Timeout Handling**: Add timeouts to queue operations to detect deadlocks
4. **Resource Limits**: Limit number of concurrent subprocesses
5. **Monitoring**: Add metrics for subprocess health and performance
6. **Error Recovery**: Automatically restart crashed subprocesses

---

## References

- Implementation: `packages/cogames/src/cogames/policy/per_agent_subprocess_wrapper.py`
- Evaluation: `packages/cogames/src/cogames/evaluate.py`
- CLI: `packages/cogames/src/cogames/main.py`

