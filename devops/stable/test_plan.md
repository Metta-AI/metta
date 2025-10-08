# Stable Release System - Test Plan

## Prerequisites

### 1. Fix Nim Compiler Issue
The `uv run` shebang in `release_stable.py` is failing due to missing Nim compiler.

**Option A**: Install Nim (recommended)
```bash
brew install nim
```

**Option B**: Run with Python directly (temporary workaround)
```bash
# Instead of ./devops/stable/release_stable.py
# Use:
python3 devops/stable/release_stable.py
```

### 2. Verify Environment
```bash
# Check Python version (should be 3.11.7)
python3 --version

# Check git is working
git status

# Check if SkyPilot is configured (for remote jobs)
sky status
```

### 3. Set Up Asana (Optional)
For automated bug checking:
```bash
export ASANA_TOKEN="your_personal_access_token"
export ASANA_PROJECT_ID="your_project_id"
```

Without this, step 2 will require manual confirmation.

---

## Test Phase 1: Dry Run (5 minutes)

Test that the script works without executing anything.

### 1.1 Show Help
```bash
./devops/stable/release_stable.py --help
# Or with workaround:
python3 devops/stable/release_stable.py --help
```

**Expected**: Help text showing all options and validation modes.

### 1.2 Check Mode - All Steps
```bash
./devops/stable/release_stable.py --all --check
```

**Expected**: Script shows what it would do for each step without making changes.

### 1.3 Check Mode - Comprehensive
```bash
./devops/stable/release_stable.py --step workflow-tests --check --comprehensive
```

**Expected**: Shows comprehensive validation plan (2B timesteps, 4 nodes).

---

## Test Phase 2: Local Validation (10 minutes)

Test that local validation works end-to-end.

### 2.1 Run Local Smoke Test Only
First, let's verify the local job runner works by creating a minimal test:

```bash
# Create a test version
export TEST_VERSION="test_$(date +%Y%m%d_%H%M%S)"
echo "Test version: $TEST_VERSION"

# Run just the local smoke test manually
python3 << 'EOF'
from devops.stable.orchestrator import run_validations
from devops.stable.models import Location, ThresholdCheck, Validation

v = Validation(
    name="quick_test",
    module="experiments.recipes.arena_basic_easy_shaped.train",
    location=Location.LOCAL,
    args=["run=stable_test", "trainer.total_timesteps=100", "wandb.enabled=false"],
    timeout_s=120,
    acceptance=[ThresholdCheck(key="sps_max", op=">=", expected=1000)],
)

state = run_validations(version="test_local", validations=[v])
print("\n=== Results ===")
print(f"Summary: {state.validation_summary}")
print(f"All passed: {state.all_validations_passed}")
EOF
```

**Expected**:
- Job runs locally for ~30-60 seconds
- Logs written to `devops/stable/logs/local/quick_test.log`
- State saved to `devops/stable/state/test_local.json`
- Metrics extracted (SPS should be present)
- Result shows PASSED or FAILED with clear reason

**Debug checklist if it fails**:
- [ ] Check log file exists and has content
- [ ] Check if SPS metric was found in logs
- [ ] Verify threshold logic (should pass if SPS >= 1000)
- [ ] Check state file was created

### 2.2 Run Step 3 with Local Only
Modify `get_release_plan()` temporarily to skip remote validation:

```python
# In release_stable.py, temporarily change get_release_plan():
validations = [
    Validation(
        name="arena_local_smoke",
        module="experiments.recipes.arena_basic_easy_shaped.train",
        location=Location.LOCAL,
        args=["run=stable.smoke", "trainer.total_timesteps=1000", "wandb.enabled=false"],
        timeout_s=600,
        acceptance=[ThresholdCheck(key="sps_max", op=">=", expected=30000)],
    ),
    # Comment out arena_remote_50k for now
]
```

Then run:
```bash
./devops/stable/release_stable.py --step workflow-tests --version local_test
```

**Expected**:
- Runs local smoke test
- Shows validation summary
- Asks for manual validation confirmation
- State saved

**Revert the change** after testing!

---

## Test Phase 3: Remote Validation (20 minutes)

Test that remote SkyPilot jobs work.

### 3.1 Verify SkyPilot Setup
```bash
# Check SkyPilot clusters
sky status

# If needed, launch a small test cluster
sky launch --gpus=1 --cloud=gcp --region=us-central1 -y -c test-cluster -- echo "test"
sky down test-cluster -y
```

### 3.2 Test Remote Job Runner Directly
```bash
python3 << 'EOF'
from devops.job_runner import run_remote

job = run_remote(
    name="test_remote",
    module="experiments.recipes.arena_basic_easy_shaped.train",
    args=["trainer.total_timesteps=1000"],
    timeout_s=600,
    log_dir="devops/stable/logs/remote",
)

print(f"Job launched: {job.job_id}")
print("Waiting for completion...")
exit_code = job.wait(timeout_s=600)
print(f"Exit code: {exit_code}")

logs = job.get_logs()
print(f"Logs (first 500 chars): {logs[:500]}")
EOF
```

**Expected**:
- SkyPilot job launches successfully
- Job ID is captured
- Logs are fetched
- Exit code is 0

**Debug checklist if it fails**:
- [ ] Check SkyPilot configuration
- [ ] Verify job appears in `sky queue`
- [ ] Check if logs are being fetched
- [ ] Verify job completes within timeout

### 3.3 Run Full Workflow Validation (Quick Mode)
```bash
./devops/stable/release_stable.py --step workflow-tests --version remote_test
```

**Expected**:
- Local smoke test runs first (1k timesteps)
- Remote 50k test launches on SkyPilot
- Both jobs complete successfully
- Metrics extracted and thresholds evaluated
- State saved with both results

---

## Test Phase 4: Full Release Workflow (30 minutes)

Test the complete 5-step release process.

### 4.1 Step 1: Prepare Branch
```bash
# First, ensure you're on main and up-to-date
git checkout main
git pull

# Run step 1
./devops/stable/release_stable.py --step prepare-branch --version test_$(date +%Y.%m.%d-%H%M)
```

**Expected**:
- Creates branch `staging/vtest_YYYY.MM.DD-HHMM-rc1`
- Pushes to origin
- Branch visible in `git branch -a`

**Cleanup after test**:
```bash
git checkout main
git branch -D staging/vtest_*-rc1
git push origin --delete staging/vtest_*-rc1
```

### 4.2 Step 2: Bug Check
```bash
./devops/stable/release_stable.py --step bug-check
```

**Expected**:
- If Asana configured: Shows bug status automatically
- If not configured: Prompts for manual confirmation
- Exits with clear pass/fail status

### 4.3 Step 3: Workflow Validation
Already tested in Phase 3.

### 4.4 Step 4: Release Instructions
```bash
./devops/stable/release_stable.py --step release --version 2025.10.07-1234
```

**Expected**:
- Prints PR template with correct branch names
- Shows git commands for tagging
- Includes key metrics section

### 4.5 Step 5: Announce
```bash
./devops/stable/release_stable.py --step announce --version 2025.10.07-1234
```

**Expected**:
- Prints Discord announcement template

### 4.6 Run All Steps Together (Dry Run)
```bash
./devops/stable/release_stable.py --all --check
```

**Expected**:
- Shows plan for all 5 steps
- No actual changes made

---

## Test Phase 5: Comprehensive Mode (Optional, 2+ hours)

Only run this if you want to test the long-running validation.

```bash
./devops/stable/release_stable.py --step workflow-tests --comprehensive --version comp_test
```

**Expected**:
- Local smoke test (quick)
- Remote 2B timestep run on 4-node cluster (long)
- Monitor with: `sky queue`
- Check logs periodically

**Warning**: This will cost cloud credits!

---

## Test Phase 6: State Persistence (10 minutes)

Test that resumability works.

### 6.1 Interrupt a Run
```bash
# Start workflow validation
./devops/stable/release_stable.py --step workflow-tests --version resume_test

# Wait for local test to complete, then Ctrl+C before remote finishes
```

### 6.2 Resume the Run
```bash
# Re-run the same command
./devops/stable/release_stable.py --step workflow-tests --version resume_test
```

**Expected**:
- Skips completed local validation
- Continues with remote validation
- Final state includes both results

---

## Success Criteria

### Must Pass
- [ ] Script runs without Python errors
- [ ] Local validation completes and extracts metrics
- [ ] Remote validation launches on SkyPilot
- [ ] State persistence works (resume after interruption)
- [ ] All 5 steps execute without errors
- [ ] Validation summary displays correctly

### Nice to Have
- [ ] Asana integration works automatically
- [ ] Comprehensive mode works (2B timesteps)
- [ ] Metrics thresholds are reasonable
- [ ] Log files are readable and useful

---

## Common Issues

### Issue: Nim compiler error
**Solution**: Install Nim with `brew install nim` or use `python3 devops/stable/release_stable.py` instead of `./devops/stable/release_stable.py`

### Issue: SkyPilot not configured
**Solution**: Run `sky check` and follow setup instructions

### Issue: Metrics not extracted
**Solution**: Check log file for presence of "SPS" or "eval_success_rate" strings. Adjust regex in `acceptance.py` if needed.

### Issue: Timeouts too short
**Solution**: Increase `timeout_s` in validation definitions

### Issue: Remote job fails to launch
**Solution**: Check `sky queue` and `sky logs <job_id>` for details

---

## Next Steps After Testing

1. **Commit changes**:
   ```bash
   git add devops/stable/release_stable.py devops/stable/README.md
   git commit -m "feat: add comprehensive validation mode and staging branch naming"
   ```

2. **Run first real release** (when ready):
   ```bash
   # Quick release
   ./devops/stable/release_stable.py --all

   # Or comprehensive (for major release)
   ./devops/stable/release_stable.py --all --comprehensive
   ```

3. **Document any adjustments needed** in DESIGN.md or testing_plan.md
