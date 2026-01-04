# CLAUDE.md - Debugging Guide for CoGsGuard Policy

Guide for AI assistants debugging this scripted agent policy.

## Quick Start

```bash
# Run with limited steps and log output (no GUI)
./tools/run.py recipes.experiment.cogsguard.play policy_uri=metta://policy/cogsguard render=log max_steps=100

# Filter output for specific agent
./tools/run.py ... 2>&1 | grep -E "^\[A0\]"

# Filter for specific events
./tools/run.py ... 2>&1 | grep -E "HAS_GEAR|DISCOVERED|MINER:"
```

## Enable Debug Mode

Set `DEBUG = True` in `policy.py`:

```python
DEBUG = True  # Line ~23
```

This enables detailed logging for:

- Agent step summaries: `[A0] Step 1: miner | Phase=get_gear | ...`
- Discovery events: `[A0] DISCOVERED miner_station at (105, 100)`
- Phase transitions and decisions

## Debugging Workflow

### 1. Identify the Problem

Run with logging and observe which agents/roles are misbehaving:

```bash
./tools/run.py ... render=log max_steps=100 2>&1 | grep -E "^\[A[0-9]\] Step 50:"
```

Check role distribution:

- A0, A4, A8: Miner
- A1, A5, A9: Scout
- A2, A6: Aligner
- A3, A7: Scrambler

### 2. Trace a Single Agent

Focus on one problematic agent:

```bash
# All output for agent A2
./tools/run.py ... 2>&1 | grep -E "^\[A2\]" | head -50

# Just step summaries
./tools/run.py ... 2>&1 | grep -E "^\[A2\].*Step [0-9]+:" | head -30
```

### 3. Check Phase Transitions

Verify agents progress through phases:

```bash
# Find when agent gets gear
./tools/run.py ... 2>&1 | grep -E "^\[A2\].*(HAS_GEAR|execute_role)"

# Check if stuck in get_gear
./tools/run.py ... 2>&1 | grep -E "^\[A2\].*get_gear" | tail -10
```

### 4. Debug Discovery

Check if agent finds required structures:

```bash
# What does agent discover?
./tools/run.py ... 2>&1 | grep -E "^\[A2\].*DISCOVERED"

# Specifically gear stations
./tools/run.py ... 2>&1 | grep -E "DISCOVERED.*station"
```

### 5. Debug Pathfinding

Check if agent navigates correctly:

```bash
# Track position changes
./tools/run.py ... 2>&1 | grep -E "^\[A2\].*Pos=" | head -20

# Check if stuck (same position repeated)
./tools/run.py ... 2>&1 | grep -E "^\[A2\].*Step" | tail -20
```

## Common Issues

### Agent Stuck in GET_GEAR Phase

**Symptom**: Agent never reaches `execute_role` phase **Check**:

```bash
grep -E "^\[A2\].*DISCOVERED.*aligner_station"
```

**Causes**:

1. Exploration not reaching station location
2. Station not being discovered (wrong object name pattern)

**Fix**: Adjust exploration in `_explore()` or discovery in `_update_occupancy_and_discover()`

### Agent Adjacent but Not Getting Gear

**Symptom**: "Adjacent to X_station, bumping it!" repeated but still NO_GEAR **Check**:

```bash
grep -E "^\[A2\].*GET_GEAR.*adjacent"
```

**Causes**:

1. Commons out of resources (gear_costs not met)
2. Agent not aligned with station's commons

**Debug**: Check commons inventory in game state output

### Position Not Updating

**Symptom**: Same `Pos=(x,y)` for many steps **Causes**:

1. Movement blocked by obstacle/wall
2. `using_object_this_step` flag blocking position update
3. Pathfinding returning blocked path

**Debug**: Add print in `_use_object_at()` to trace

### Role-Specific Debugging

#### Miner Issues

```bash
# Check extractor discovery
grep "DISCOVERED.*extractor"

# Check cargo levels
grep "MINER: cargo="
```

#### Aligner Issues

```bash
# Check influence/heart levels
grep "ALIGNER_EXEC: influence=.*, heart="

# Need both influence >= 1 AND heart >= 1 to align
```

#### Scrambler Issues

```bash
# Check heart levels (needs >= 1)
grep "scrambler.*heart="

# Check if finding enemy depots
grep "DISCOVERED.*charger"
```

## Adding Debug Output

### Temporary Debug Print

```python
# In any method, add:
if s.step_count <= 50:  # Limit output
    print(f"[A{s.agent_id}] DEBUG: var={value}")
```

### Using DEBUG Flag

```python
if DEBUG:
    print(f"[A{s.agent_id}] MESSAGE")

# Or with step limit:
if DEBUG and s.step_count <= 100:
    print(f"[A{s.agent_id}] MESSAGE")
```

## Key Files to Modify

| Issue              | File           | Method                             |
| ------------------ | -------------- | ---------------------------------- |
| Phase logic        | `policy.py`    | `_update_phase()`                  |
| Gear acquisition   | `policy.py`    | `_do_get_gear()`                   |
| Object discovery   | `policy.py`    | `_update_occupancy_and_discover()` |
| Exploration        | `policy.py`    | `_explore()`                       |
| Pathfinding        | `policy.py`    | `_move_towards()`                  |
| Miner behavior     | `miner.py`     | `execute_role()`                   |
| Aligner behavior   | `aligner.py`   | `execute_role()`                   |
| Scrambler behavior | `scrambler.py` | `execute_role()`                   |
| Scout behavior     | `scout.py`     | `execute_role()`                   |

## Game State Output

At episode end, the log shows game state:

```
{'cogs': {'aligned.assembler': 1.0, ...},
 'clips': {'aligned.charger': 42.0}}
```

This helps verify:

- Did cogs align any depots?
- Are miners depositing resources?
- Is the scrambler working (charger alignment decreasing)?

## Useful Grep Patterns

```bash
# All discoveries
grep "DISCOVERED"

# All role executions
grep "execute_role"

# Energy tracking
grep "Energy="

# Specific action types
grep "Action=move_"
grep "Action=noop"

# Phase changes
grep "Phase="
```
