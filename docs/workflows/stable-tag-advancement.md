# Stable Tag Advancement Workflow

## Overview

The stable tag advancement workflow automatically promotes commits on the `main` branch to a `stable` tag after comprehensive testing. This provides a known-good version of the codebase that has passed all quality checks.

## How It Works

### 1. Trigger
The workflow triggers automatically when:
- A PR is merged to `main`
- Manually via workflow dispatch (with options to skip tests or adjust grace period)

### 2. Comprehensive Testing Phase
The workflow runs an extensive test suite including:

#### Full Test Suite
- All unit tests (with coverage)
- Integration tests (`pytest -m integration`)
- Expensive tests (`pytest -m expensive`)

#### Performance Benchmarks
- Python benchmarks with regression detection (fails if >10% performance degradation)
- C++ benchmarks from mettagrid

#### Config Validation
- Validates all YAML configuration files can be loaded
- Ensures no syntax errors in configs

#### Example Script Testing
- Runs `tools/train.py` with minimal settings to ensure training pipeline works

### 3. Grace Period
If all tests pass, a configurable grace period begins (default: 30 minutes):
- Allows manual intervention if issues are discovered
- Can be cancelled by pushing a commit with `[skip-stable]` in the message
- Posts notification to Discord with countdown timer

### 4. Tag Advancement
After the grace period expires:
- Updates the `stable` tag to point to the tested commit
- Updates the stability dashboard at `docs/stability/dashboard.md`
- Records history in `docs/stability/history.json`
- Generates and posts a changelog to Discord

## Usage

### Manual Trigger
You can manually trigger the workflow from the Actions tab with options:
- **skip_tests**: Skip all tests and advance tag immediately (use with extreme caution)
- **grace_period_minutes**: Adjust the grace period (default: 30 minutes)

### Cancelling Tag Advancement
During the grace period, you can cancel the advancement by:
1. Pushing a commit with `[skip-stable]` in the commit message
2. Manually cancelling the workflow run in GitHub Actions

### Monitoring
- **Discord Notifications**: Posted to the configured webhook for:
  - Grace period start (with countdown)
  - Successful tag advancement (with changelog)
  - Test failures
- **Stability Dashboard**: View at `docs/stability/dashboard.md` for:
  - Current stable release info
  - Test results
  - Recent history
  - Statistics (total releases, average commits between releases)

## Configuration

### Required Secrets
- `DISCORD_WEBHOOK_URL`: Discord webhook for notifications

### Test Markers
Ensure your tests are properly marked:
```python
@pytest.mark.integration
def test_integration_feature():
    pass

@pytest.mark.expensive
def test_expensive_operation():
    pass
```

### Benchmark Configuration
For performance regression detection:
```python
def test_performance(benchmark):
    result = benchmark(function_to_test)
    # Automatically fails if >10% regression
```

## Best Practices

1. **Regular Integration**: Merge PRs frequently to avoid large gaps between stable releases
2. **Test Coverage**: Ensure critical paths have integration tests
3. **Performance Monitoring**: Add benchmarks for performance-critical code
4. **Config Testing**: Test with various configs in integration tests
5. **Grace Period**: Use the grace period to monitor production after merges

## Troubleshooting

### Tests Pass in PR but Fail in Stable Workflow
- Check for environment differences (PR runs vs main branch)
- Look for timing-dependent tests
- Verify all dependencies are properly installed

### Tag Not Advancing
- Check workflow runs for failures
- Verify Discord webhook is configured
- Ensure no `[skip-stable]` commits were pushed

### Performance Regression Detected
- Compare benchmark results in artifacts
- Use `--benchmark-compare` locally to investigate
- Consider if regression is acceptable or needs fixing