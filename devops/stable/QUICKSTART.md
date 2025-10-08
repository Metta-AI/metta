# Quick Start - Stable Release

## 1. Pre-flight Check (2 minutes)

```bash
# Verify Nim compiler (needed for uv run shebang)
which nim || brew install nim

# Verify git
git status

# Verify SkyPilot (for remote jobs)
sky status

# Test the script works
./devops/stable/release.py --help
```

---

## 2. Dry Run (1 minute)

Test the full workflow without making changes:

```bash
# Full dry-run
./devops/stable/release.py --all --check

# Expected output:
# - Shows what branch would be created
# - Shows bug check process
# - Shows validation plan (QUICK mode by default)
# - Shows PR template
# - Shows announcement template
```

---

## 3. Optional: Configure Asana (2 minutes)

For automated bug checking (optional but recommended):

```bash
export ASANA_TOKEN="your_personal_access_token"
export ASANA_PROJECT_ID="your_project_id"

# Test Asana connection
./devops/stable/release.py --step bug-check --check
```

Without Asana, the script will prompt for manual confirmation.

---

## 4. First Real Release (30-60 minutes)

### Quick Mode (Recommended)
For regular releases with fast validation (50k timesteps, ~30 min):

```bash
# Run all steps
./devops/stable/release.py --all

# Or step by step:
./devops/stable/release.py --step prepare-branch
./devops/stable/release.py --step bug-check
./devops/stable/release.py --step workflow-tests  # Takes ~30 min
./devops/stable/release.py --step release
./devops/stable/release.py --step announce
```

### Comprehensive Mode (Major Releases)
For major releases with thorough validation (2B timesteps, 4 nodes, ~hours):

```bash
./devops/stable/release.py --all --comprehensive

# Note: Step 3 (workflow-tests) will take several hours
```

---

## 5. During Validation (Step 3)

While validations are running:

```bash
# Monitor state
cat devops/stable/state/release_*.json | jq

# Watch local logs
tail -f devops/stable/logs/local/*.log

# Check remote jobs
sky queue

# Monitor remote logs (if you know the job ID)
sky logs <job-id> --follow
```

**If validation fails:**
1. Check logs in `devops/stable/logs/`
2. Review metrics in the printed summary
3. Fix issues
4. Re-run step 3 (completed validations are skipped automatically)

---

## 6. After Validation Passes

The script will:
1. ✅ Print instructions for creating the release PR
2. ✅ Show the PR template to use
3. ✅ Show git commands for tagging
4. ✅ Show Discord announcement template

**You manually do:**
1. Create `devops/stable/release-notes/v{version}.md`
2. Open PR from `staging/v{version}-rc1` to `stable`
3. After approval:
   ```bash
   git checkout stable
   git pull
   git tag -a v{version} -m "Release version {version}"
   git push origin v{version}
   ```
4. Post announcement to Discord #eng-process

---

## Common Issues

### Issue: `uv run` fails with Nim compiler error
**Solution**: Install Nim (`brew install nim`) or use:
```bash
python3 devops/stable/release.py <args>
```

### Issue: SkyPilot not configured
**Solution**: Run `sky check` and follow setup instructions

### Issue: Metrics not extracted (validation fails with "metric missing")
**Solution**:
1. Check log file for "SPS" or "eval_success_rate" strings
2. If missing, the training may have failed to complete
3. Review full logs in `devops/stable/logs/`

### Issue: Timeout
**Solution**: The validation exceeded its timeout. Check logs to see if training is stuck or just slow.

### Issue: Remote job fails to launch
**Solution**: Check `sky queue` and `sky logs <job-id>` for details

---

## Resume After Interruption

If you interrupt (Ctrl+C) during step 3, you can resume:

```bash
# Re-run the same command
./devops/stable/release.py --step workflow-tests

# Completed validations are automatically skipped
# Only incomplete/failed validations will re-run
```

State is saved to: `devops/stable/state/release_{version}.json`

---

## Quick Reference

```bash
# Help
./devops/stable/release.py --help

# Full release (quick mode)
./devops/stable/release.py --all

# Full release (comprehensive mode for major releases)
./devops/stable/release.py --all --comprehensive

# Dry-run everything
./devops/stable/release.py --all --check

# Run single step
./devops/stable/release.py --step workflow-tests

# Custom version
./devops/stable/release.py --all --version 2025.10.07
```

---

## Files Created

During the release process, these files are created:

```
devops/stable/
├── state/
│   └── release_{version}.json          # State file (resumable)
├── logs/
│   ├── local/
│   │   └── arena_local_smoke.log       # Local validation logs
│   └── remote/
│       └── arena_remote_50k.log        # Remote validation logs (or arena_staging_2b.log)
└── release-notes/
    └── v{version}.md                    # You create this manually
```

---

## Next Steps

After your first successful release:

1. **Review the process** - Did anything fail? Take notes in the state file or logs
2. **Adjust thresholds** - If SPS thresholds are too strict/loose, update in `release.py`
3. **Add more validations** - Edit `get_release_plan()` to add recipe tests
4. **Automate more** - Consider adding W&B metrics extraction or regression checks

---

## Support

For questions about this release process, contact:
- Release Manager: @Robb
- Technical Lead: @Jack Heart
- Bug Triage: @Nishad Singh
