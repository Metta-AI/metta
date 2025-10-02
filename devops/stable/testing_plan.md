# Testing Plan for Stable Release System

## What Changed

Migrated from `devops/stable/runner.py` to a shared `devops/job_runner.py`:

- **New shared runner**: `devops/job_runner.py` with `run_local()` and `run_remote()` functions
- **Updated orchestrator**: `devops/stable/orchestrator.py` now uses the new job_runner
- **Removed old runner**: `devops/stable/runner.py` deleted
- **Updated docs**: `DESIGN.md` reflects new architecture

## Critical Tests Before Merging

### 1. Local Validation Test (5 minutes)

Test that local jobs run correctly:

```bash
# Run a quick local smoke test
./devops/stable/release_stable.py --step workflow-tests --version test_local --check

# Expected: Should show validation plan without executing
# Then run without --check to execute
./devops/stable/release_stable.py --step workflow-tests --version test_local
```

**Success criteria:**
- Local job launches without errors
- Logs written to `devops/stable/logs/local/`
- Exit code captured correctly
- Metrics extracted from logs
- State saved to `devops/stable/state/test_local.json`

### 2. Remote Validation Test (15 minutes)

Test that remote jobs launch on SkyPilot:

```bash
# Create a test validation with remote execution
# Edit release_stable.py temporarily to add a short remote test, or run full workflow
./devops/stable/release_stable.py --step workflow-tests --version test_remote
```

**Success criteria:**
- SkyPilot job launches successfully
- Job ID captured in state
- Logs fetched via `tail_job_log()`
- Job completes and exit code extracted
- Metrics evaluated correctly

### 3. State Persistence Test

Verify incremental state saving works:

```bash
# Start a validation run
./devops/stable/release_stable.py --step workflow-tests --version test_state

# Interrupt it (Ctrl+C) partway through
# Re-run same command
./devops/stable/release_stable.py --step workflow-tests --version test_state

# Expected: Should skip completed validations and resume
```

**Success criteria:**
- State file exists after first run
- Second run skips completed validations
- Final state includes all results

### 4. Error Handling Tests

Test failure modes:

```bash
# Test with invalid module (should fail gracefully)
# Test with missing metrics (should mark as failed with "metric missing")
# Test with timeout (should return exit code 124)
```

**Success criteria:**
- Clear error messages
- State reflects failures correctly
- Logs attached even on failure

### 5. Full Release Workflow Test

Run all 5 steps end-to-end:

```bash
export TEST_VERSION=$(date +%Y.%m.%d-%H%M)

# Step 1: Prepare branch
./devops/stable/release_stable.py --step prepare-branch --version $TEST_VERSION --check

# Step 2: Bug check
./devops/stable/release_stable.py --step bug-check --check

# Step 3: Workflow validation
./devops/stable/release_stable.py --step workflow-tests --version $TEST_VERSION

# Step 4: Release instructions
./devops/stable/release_stable.py --step release --version $TEST_VERSION --check

# Step 5: Announcement
./devops/stable/release_stable.py --step announce --version $TEST_VERSION --check
```

**Success criteria:**
- All steps complete without errors
- State file has all validation results and gate results
- Summary shows pass/fail status correctly

## Code Quality Checks

```bash
# Lint Python files
metta lint --fix

# Run type checking (if available)
metta ci
```

## Integration with Existing Systems

### Future Work (Not Required Now)

The new `devops/job_runner.py` is designed to be shared with:

1. **Sweep system** - Check `metta/sweep/` for compatibility
2. **Adaptive system** - Check `metta/adaptive/` for similar patterns
3. **General devops** - Any other job execution needs

**Next steps for integration:**
- Compare API with sweep/adaptive job execution
- Refactor sweep/adaptive to use shared job_runner if beneficial
- Document shared patterns in job_runner.py docstrings

## Known Limitations

1. **No streaming logs** - Local jobs buffer all output, only written after completion
2. **No retry logic** - Transient failures require manual re-run
3. **No regression checks** - Each validation is standalone, no comparison to previous releases
4. **Regex-only metrics** - No W&B or other metrics sources yet

## Rollback Plan

If the new job_runner has issues:

```bash
# Revert the changes
git checkout HEAD~1 -- devops/stable/orchestrator.py
git checkout HEAD~1 -- devops/stable/runner.py
rm devops/job_runner.py

# Or revert the entire commit
git revert HEAD
```

## Questions to Answer

- [ ] Does local job execution work correctly?
- [ ] Does remote job execution work correctly?
- [ ] Are logs attached in all cases (success/failure/timeout)?
- [ ] Are metrics extracted and thresholds evaluated?
- [ ] Does state persistence and resumption work?
- [ ] Are error messages clear and actionable?
- [ ] Does the full 5-step workflow complete?

## After Testing

Once all tests pass:

1. Update this testing_plan.md with actual results
2. Create a PR with the changes
3. Consider adding automated tests for job_runner.py
4. Document any issues or edge cases discovered
