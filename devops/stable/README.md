# Stable Release Process

Automated qualification system for creating stable releases.

**Contacts:**
- Release Manager: @Robb
- Technical Lead: @Jack Heart
- Bug Triage: @Nishad Singh

## For Release Managers

### Quick Start

```bash
# Run all steps for a new release (auto-generates version from date/time)
./devops/stable/release_stable.py --all

# Or run steps individually:
./devops/stable/release_stable.py --step prepare-branch
./devops/stable/release_stable.py --step bug-check
./devops/stable/release_stable.py --step workflow-tests
./devops/stable/release_stable.py --step release
./devops/stable/release_stable.py --step announce
```

### Release Steps

#### Step 1: Prepare Staging Branch
Creates and pushes a `staging/v{version}-rc1` branch.

```bash
./devops/stable/release_stable.py --step prepare-branch --version 2025.10.02
```

#### Step 2: Bug Check
Checks Asana for blocking bugs in the "Active" section.

**Setup (first time only):**
```bash
export ASANA_TOKEN="your_personal_access_token"
export ASANA_PROJECT_ID="your_project_id"
```

If not configured, will prompt for manual confirmation.

#### Step 3: Workflow Validation
Runs automated training validations (local + remote).

**Quick mode (default)**:
- **arena_local_smoke**: 1k timesteps locally, expects SPS ≥ 30k
- **arena_remote_50k**: 50k timesteps on SkyPilot (1 node), expects SPS ≥ 40k

**Comprehensive mode** (`--comprehensive` flag, for major releases):
- **arena_local_smoke**: 1k timesteps locally, expects SPS ≥ 30k
- **arena_staging_2b**: 2B timesteps on SkyPilot (4 nodes, 16 GPUs), expects SPS ≥ 40k

State is saved to `devops/stable/state/release_{version}.json` for resumability.

```bash
# Quick validation (regular releases)
./devops/stable/release_stable.py --step workflow-tests

# Comprehensive validation (major releases)
./devops/stable/release_stable.py --step workflow-tests --comprehensive
```

**If validations fail:**
- Check logs in `devops/stable/logs/`
- Review metrics in the printed summary
- Fix issues and re-run (completed validations are skipped)

#### Step 4: Release
Creates the release PR and tags.

1. **Create release notes**: `devops/stable/release-notes/v{version}.md`
2. **Open PR**: From `staging/v{version}-rc1` to `stable`
3. **After approval**, merge and tag:
   ```bash
   git checkout stable
   git pull
   git tag -a v{version} -m "Release version {version}"
   git push origin v{version}
   ```

#### Step 5: Announce
Post to Discord #eng-process:
```
Released stable version v{version} 🎉
Release notes: devops/stable/release-notes/v{version}.md
```

### Modifying Validations

Edit `get_release_plan()` in `release_stable.py`:

```python
validations = [
    Validation(
        name="my_validation",
        module="experiments.recipes.my_recipe.train",
        location=Location.LOCAL,  # or Location.REMOTE
        args=["run=stable.test", "trainer.total_timesteps=1000"],
        timeout_s=600,
        acceptance=[
            ThresholdCheck(key="sps_max", op=">=", expected=30000),
        ],
    ),
]
```

**Supported operators**: `>=`, `>`, `<=`, `<`, `==`, `!=`

**Available metrics** (extracted from logs):
- `sps_max` - Maximum samples per second
- `sps_last` - Last SPS value
- `eval_success_rate` - Evaluation success rate

### Troubleshooting

**"metric missing" failure:**
The expected metric wasn't found in logs. Check:
- Log file location in error output
- Metric extraction regex patterns in `acceptance.py`

**Timeout:**
Validation exceeded timeout. Check logs and consider:
- Increasing `timeout_s` in validation definition
- Checking if training is stuck

**Remote launch failed:**
Check SkyPilot configuration and cluster availability.

### Check Mode

Dry-run steps without making changes:
```bash
./devops/stable/release_stable.py --step workflow-tests --check
```

Useful for:
- Verifying configuration
- Understanding what will happen
- Testing step 4 (release) and step 5 (announce) safely
