# CLAUDE.md - Debugging Guide for CoGsGuard Policy

Guide for AI assistants debugging this scripted agent policy.

## Python Debug Harness (Recommended)

The `debug_agent.py` harness provides programmatic access to simulation state for diagnosing issues.

### Quick Start

```python
from cogames.policy.scripted_agent.cogsguard.debug_agent import DebugHarness

# Create harness from recipe
harness = DebugHarness.from_recipe(num_agents=10, seed=42)

# Step through simulation
harness.step(50)

# Print agent summary
harness.print_agent_summary()

# Find stuck agents and diagnose
stuck = harness.diagnose_stuck_agents(threshold=10)

# Diagnose coordinate issues (common bug)
harness.diagnose_coordinate_system()
```

### Key Methods

| Method                                  | Description                                |
| --------------------------------------- | ------------------------------------------ |
| `step(n)`                               | Run n simulation steps                     |
| `print_agent_summary(ids)`              | Print state summary for agents             |
| `print_simulation_info()`               | Print object types and assembler locations |
| `diagnose_stuck_agents(threshold)`      | Find and diagnose stuck agents             |
| `diagnose_coordinate_system()`          | Check for coordinate mismatches            |
| `run_until_stuck(threshold, max_steps)` | Run until agent stuck                      |
| `get_agent_state(id)`                   | Get internal policy state                  |
| `get_grid_objects()`                    | Get all simulation objects                 |
| `find_assemblers()`                     | Get actual assembler positions             |

### Common Diagnostic Patterns

```python
# Check if internal positions match simulation
harness.diagnose_coordinate_system()

# Inspect specific agent state
state = harness.get_agent_state(0)
print(f"Role: {state.role}, Phase: {state.phase}")
print(f"Position: ({state.row}, {state.col})")
print(f"Known assembler: {state.stations.get('assembler')}")

# Check actual vs believed assembler positions
actual = harness.find_assemblers()
for i in range(harness.num_agents):
    info = harness.agent_info[i]
    print(f"Agent {i}: believes assembler at {info.assembler_pos}, actual: {actual}")

# Run until issues appear
harness.run_until_stuck(threshold=10, max_steps=200)
```

### Run from Command Line

```bash
cd /Users/daveey/code/metta
uv run python -c "
from cogames.policy.scripted_agent.cogsguard.debug_agent import DebugHarness
h = DebugHarness.from_recipe(num_agents=10)
h.step(100)
h.diagnose_stuck_agents()
h.diagnose_coordinate_system()
"
```

## Coordinate System

**IMPORTANT**: Agents track positions **relative to their starting point**, not absolute map coordinates.

- Agents start at internal position `(0, 0)` (their origin)
- All positions are stored relative to this origin
- The actual map size is irrelevant - only relative offsets matter
- If you see position mismatches, the coordinate tracking is broken

### Diagnosing Coordinate Issues

The most common bug is agents storing wrong object positions. Use the debug harness:

```python
harness = DebugHarness.from_recipe()
harness.step(50)
harness.diagnose_coordinate_system()

# Output will show:
# - Actual assembler positions in simulation
# - What each agent believes the assembler position is
# - Whether there's a mismatch
```

## Vibe-Based Role System

Agents use **vibes** to determine their behavior:

| Vibe        | Behavior                                                                     |
| ----------- | ---------------------------------------------------------------------------- |
| `default`   | Do nothing (noop)                                                            |
| `gear`      | Pick a random role (scout/miner/aligner/scrambler), change vibe to that role |
| `miner`     | Get miner gear if needed, then mine resources                                |
| `scout`     | Get scout gear if needed, then explore                                       |
| `aligner`   | Get aligner gear if needed, then align chargers to cogs                      |
| `scrambler` | Get scrambler gear if needed, then scramble enemy chargers                   |
| `heart`     | Do nothing (noop)                                                            |

## Quick Start

```bash
# Run with limited steps and log output (no GUI)
./tools/run.py recipes.experiment.cogsguard.play policy_uri=metta://policy/cogsguard render=log max_steps=100

# Custom vibe distribution: 4 miners, 2 scramblers, 1 gear
./tools/run.py recipes.experiment.cogsguard.play \
    policy_uri="metta://policy/cogsguard?miner=4&scrambler=2&gear=1" render=log max_steps=100

# Filter output for specific agent
./tools/run.py ... 2>&1 | grep -E "^\[A0\]"

# Filter for specific events
./tools/run.py ... 2>&1 | grep -E "HAS_GEAR|DISCOVERED|MINER:|GEAR_VIBE"

# Filter for initial vibe assignments
./tools/run.py ... 2>&1 | grep -E "INITIAL_VIBE"
```

### Initial Vibe URI Parameters

Control agent role distribution via query parameters:

```
metta://policy/cogsguard?miner=4&scrambler=2&gear=1&aligner=2&scout=1
```

- `miner`, `scout`, `aligner`, `scrambler`: Count for each role
- `gear`: Agents that pick a random role
- Assignment order: `scrambler → aligner → miner → scout → gear`
- Default: `scrambler=1, miner=4`

## Enable Debug Mode

Set `DEBUG = True` in `policy.py`:

```python
DEBUG = True  # Line ~48
```

This enables detailed logging for:

- Agent step summaries: `[A0] Step 1: vibe=gear role=miner | Phase=get_gear | ...`
- Vibe transitions: `[A0] GEAR_VIBE: Picking random role vibe: scout`
- Discovery events: `[A0] DISCOVERED miner_station at (105, 100)`
- Phase transitions and decisions

## Debugging Workflow

### 1. Identify the Problem

Run with logging and observe which agents/vibes are misbehaving:

```bash
./tools/run.py ... render=log max_steps=100 2>&1 | grep -E "^\[A[0-9]\] Step 50:"
```

Note: Roles are now dynamic based on vibes. Check what vibe each agent has:

```bash
./tools/run.py ... 2>&1 | grep -E "^\[A[0-9]\].*vibe="
```

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

### Initial Vibe Not Applied

**Symptom**: Agent doesn't switch to expected initial vibe from URI params

**Check**:

```bash
# Look for initial vibe assignment messages
grep -E "INITIAL_VIBE|CogsguardPolicy.*Initial vibe"

# Check what vibe agent actually has
grep -E "^\[A0\].*vibe="
```

**Causes**:

1. URI params not parsed correctly (check quoting in shell)
2. Agent ID exceeds configured count (only agents 0..N-1 get assigned vibes)

**Debug**: The policy logs initial vibe assignment at startup:

```
[CogsguardPolicy] Initial vibe assignment: ['scrambler', 'miner', 'miner', 'miner', 'miner']
```

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
3. Low energy causing action delay (not failure)

**Debug**: Check `last_action_executed` vs `last_action` (intended) for mismatches.

### Position Tracking Drift

**Symptom**: Agent's internal position diverges from actual simulation position over time.

**Key insight**: Position only updates when intended action matches executed action:

- Agent intends `move_east`, simulator executes `move_east` → position updates
- Agent intends `move_east`, simulator executes `noop` (failed) → position stays
- Human moves cog with `move_north`, agent intended `noop` → position stays frozen

**Human takeover behavior**: When a human takes control and moves a cog, the agent's internal position does NOT update.
This is intentional - it keeps the agent's internal map consistent. When control returns to the agent, it continues from
where it "thinks" it is.

**1-step lag**: Internal position is always 1 step behind simulation because:

- At step N, we read `last_action` = step N-1's executed action
- We update internal position based on step N-1
- Then sim.step() executes step N's action

This lag is expected and correct. To verify position tracking:

```python
# int_delta[N] should equal act_delta[N-1]
harness.track_position_drift(num_steps=50)
```

### Coordinate Mismatch (Agent Thinks Assembler is Elsewhere)

**Symptom**: Agent tries to deposit but cargo never decreases. Debug harness shows:

```
Agent X: believes assembler at (100, 102), actual: [(29, 29)]
```

**Note**: Internal coords are RELATIVE (centered at 100,100), simulation coords are ABSOLUTE. These will NOT match
directly - compare DELTAS instead.

**Causes**:

1. Position drift from second-guessing simulator's `last_action`
2. Not updating position when `last_action` is a successful move
3. Timing mismatch checking occupancy from wrong timestep

**Diagnose with harness**:

```python
harness = DebugHarness.from_recipe()
harness.step(100)
harness.verify_position_tracking()  # Check for drift
harness.diagnose_coordinate_system()
```

**Fix**: In `_update_agent_position`, trust `last_action_executed` from observation. If it says `move_X`, update
position unconditionally (simulator already validated the move).

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

| Issue               | File           | Method/Class                              |
| ------------------- | -------------- | ----------------------------------------- |
| Initial vibe counts | URI params     | `?scrambler=1&miner=4` in policy URI      |
| Initial vibe switch | `policy.py`    | `CogsguardMultiRoleImpl._execute_phase()` |
| Phase logic         | `policy.py`    | `_update_phase()`                         |
| Gear acquisition    | `policy.py`    | `_do_get_gear()`                          |
| Object discovery    | `policy.py`    | `_update_occupancy_and_discover()`        |
| Exploration         | `policy.py`    | `_explore()`                              |
| Pathfinding         | `policy.py`    | `_move_towards()`                         |
| Miner behavior      | `miner.py`     | `execute_role()`                          |
| Aligner behavior    | `aligner.py`   | `execute_role()`                          |
| Scrambler behavior  | `scrambler.py` | `execute_role()`                          |
| Scout behavior      | `scout.py`     | `execute_role()`                          |

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

## Action Failure Detection and Retry System

Moves require energy. If an agent doesn't have enough energy, the move fails silently (simulator executes `noop`
instead).

**Key insight**: Agents auto-regenerate energy every step, and regenerate full energy when near aligned buildings (like
the nexus AOE). So we don't need complex "go recharge" logic - just detect failures and retry.

### How It Works

1. **Track intended vs executed actions**: Compare `last_action` (what we wanted) vs `last_action_executed` (from
   observation - what simulator did)
2. **Detect action success**: By comparing before/after state (hearts, cargo)
3. **Retry failed actions**: Up to MAX_RETRIES (default 3) times - agent will have auto-regened energy

### Action Success Detection

- **Scramble/Align**: Success detected when heart count decreases (consumes 1 heart)
- **Mine**: Success detected when cargo increases
- **Move**: Success detected when `last_action_executed` matches intended action

### Position Tracking

**CRITICAL**: Position is only updated when the agent's intended move matches the executed move:

```python
# Only update if WE intended this move (not human control)
if intended_action == executed_action and executed_action.startswith("move_"):
    # Update position
```

This design ensures:

1. Position doesn't update when moves fail (executed=noop vs intended=move_X)
2. Position stays frozen when a human takes over and moves the cog around
3. When control returns to the agent, its internal map remains consistent

### Debugging Action Failures

```bash
# See move failures (action mismatch)
grep "ACTION_MISMATCH"

# Check for retry messages
grep -E "retrying|failed after.*retries"

# Track action success
grep -E "succeeded|failed"
```

## Useful Grep Patterns

```bash
# Initial vibe assignment (policy startup)
grep "Initial vibe assignment"

# Initial vibe switches (per agent)
grep "INITIAL_VIBE"

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

# Vibe transitions
grep "GEAR_VIBE\|change_vibe"

# Action retry tracking
grep -E "SCRAMBLING|ALIGNING|MINING"
grep -E "retry|failed|succeeded"
```
