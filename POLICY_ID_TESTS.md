# Policy Identification Feature Tests

This directory contains comprehensive tests for the policy identification feature that adds `policy_id` tracking to replays and stats.

## Quick Start

Run all tests:
```bash
./run_policy_tests.sh
```

Run individual tests:
```bash
# Test 1: Basic single policy
uv run python test_policy_basic.py

# Test 2: Multiple policies
uv run python test_policy_multi.py

# Test 3: Training environment
uv run python test_policy_training.py

# Test 4: Replay structure validation
uv run python test_policy_replay_structure.py
```

## Test Descriptions

### Test 1: Basic Single Policy (`test_policy_basic.py`)
**What it tests:**
- All agents controlled by a single policy
- Replay contains `policies` array with one entry
- All agents have `policy_id=0`
- Policy structure includes `name`, `uri`, and `is_scripted` fields

**Expected output:**
```
✅ PASS: Replay contains 'policies' array
✅ PASS: Exactly 1 unique policy found
✅ PASS: Policy structure correct
✅ PASS: All 4 agents have correct policy_id=0
```

### Test 2: Multi-Policy (`test_policy_multi.py`)
**What it tests:**
- Different policies controlling different agents
- Agents 0-1 use "random" policy
- Agents 2-3 use "noop" policy
- Replay contains 2 unique policies
- Each agent has correct policy_id

**Expected output:**
```
✅ PASS: Found 2 unique policies
✅ PASS: Both 'random' and 'noop' policies present
✅ PASS: All agents assigned to correct policies
```

**Agent assignments:**
```
Agent 0: policy_id=0 (random)
Agent 1: policy_id=0 (random)
Agent 2: policy_id=1 (noop)
Agent 3: policy_id=1 (noop)
```

### Test 3: Training Environment (`test_policy_training.py`)
**What it tests:**
- Training environments use "training" policy descriptor
- All agents get the same "training" policy
- Training policy has `is_scripted=False`
- Only 1 unique policy across all agents

**Expected output:**
```
✅ PASS: Environment has 4 policy descriptors (one per agent)
✅ PASS: All agents use 'training' policy descriptor
✅ PASS: Only 1 unique 'training' policy
✅ PASS: All agents have policy_id=0
```

### Test 4: Replay Structure (`test_policy_replay_structure.py`)
**What it tests:**
- Replay format matches specification
- All required top-level keys present
- Policies array has correct structure
- Agent objects include `policy_id` field
- Policy IDs are valid indices into policies array

**Expected output:**
```
✅ PASS: All required top-level keys present
✅ PASS: Policies array structure valid
✅ PASS: All agent objects have valid policy_id
```

## What Gets Tested

### Replay Format
```json
{
  "version": 3,
  "policies": [
    {
      "name": "random",
      "uri": "",
      "is_scripted": true
    }
  ],
  "objects": [
    {
      "agent_id": 0,
      "policy_id": 0,
      ...
    }
  ],
  ...
}
```

### Key Validations
- ✅ `policies` array exists at replay root
- ✅ Each policy has `name`, `uri`, `is_scripted`
- ✅ Each agent object has `policy_id`
- ✅ Policy IDs are valid indices
- ✅ Multi-policy scenarios work correctly
- ✅ Training environments have proper descriptors

## Interpreting Results

### Success
When all tests pass, you'll see:
```
╔════════════════════════════════════════════════════════════╗
║            🎉 ALL TESTS PASSED! 🎉                        ║
╚════════════════════════════════════════════════════════════╝
```

Exit code: `0`

### Failure
If any test fails, you'll see:
```
╔════════════════════════════════════════════════════════════╗
║            ⚠️  SOME TESTS FAILED  ⚠️                      ║
╚════════════════════════════════════════════════════════════╝

Failed tests:
  - Test Name
```

Exit code: `1`

## Integration with CI/CD

These tests can be integrated into your CI pipeline:

```bash
# In your CI script
cd /path/to/metta
./run_policy_tests.sh
if [ $? -ne 0 ]; then
    echo "Policy ID tests failed"
    exit 1
fi
```

## Manual Replay Inspection

To manually inspect a generated replay:

```python
import json
import zlib

# Read replay
with open('path/to/replay.json.z', 'rb') as f:
    replay_data = json.loads(zlib.decompress(f.read()))

# Check policies
print("Policies:", replay_data['policies'])

# Check agents
for obj in replay_data['objects']:
    if 'agent_id' in obj:
        print(f"Agent {obj['agent_id']}: policy_id={obj.get('policy_id')}")
```

## Troubleshooting

### Tests fail to import mettagrid
Make sure you're in the correct directory and using `uv run`:
```bash
cd /Users/prashantcraju/.cursor/worktrees/metta/22wMz
uv run python test_policy_basic.py
```

### "policies array is empty"
This means policy descriptors weren't set on the simulation. Check that:
1. Rollout is setting descriptors after creating simulation
2. ReplayLogWriter builds policies array in `get_replay_data()`

### "Agent missing policy_id"
This means `format_grid_object` isn't receiving agent_policy_ids. Check:
1. ReplayLogWriter's `log_step()` builds the mapping
2. `format_grid_object()` accepts and uses `agent_policy_ids` parameter

## Next Steps

After tests pass:
1. ✅ Python implementation complete
2. 🔄 Update Nim mettascope viewer to display policy information
3. 🔄 Add policy filtering in analytics dashboards
4. 🔄 Visualize multi-policy interactions in UI

## Related Documentation

- Replay spec: `packages/mettagrid/nim/mettascope/docs/replay_spec.md`
- Policy classes: `packages/mettagrid/python/src/mettagrid/policy/policy.py`
- Implementation PR: [Link to PR]

