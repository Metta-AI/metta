# GitHub Actions Workflows

This directory contains all GitHub Actions workflows for the Metta project.

## Test Workflows

### Fail-Fast Testing Strategy

The main test workflow (`checks.yml`) implements a **fail-fast testing strategy** to reduce CI feedback time from 10+ minutes to 1-2 minutes for most failures.

#### How It Works

Tests are split into two categories:

1. **Fast Tests** 🏃
   - Run first with a 2-minute timeout
   - Execute tests marked as `not slow` (`pytest -m "not slow"`)
   - Include most unit tests and quick integration tests
   - Generate separate test reports (`fast-test-results.xml`, `fast-test-report.json`)
   - If these fail, a comment is immediately posted to the PR with failure details

2. **Slow Tests** 🐢
   - Only run if fast tests pass (dependency: `needs: fast-tests`)
   - Execute with a 10-minute timeout
   - Run tests marked as `slow` (`pytest -m slow`)
   - Include heavy integration tests, C++ tests, and tests requiring builds
   - Generate separate test reports (`slow-test-results.xml`, `slow-test-report.json`)

#### Marking Tests as Slow

To mark a test as slow in pytest, use the `@pytest.mark.slow` decorator:

```python
import pytest

@pytest.mark.slow
def test_heavy_integration():
    # This test will only run in the slow-tests job
    pass

def test_quick_unit():
    # This test will run in the fast-tests job
    pass
```

You can also mark entire test classes or modules as slow:

```python
# Mark entire class
@pytest.mark.slow
class TestHeavyFeature:
    def test_method1(self):
        pass

    def test_method2(self):
        pass

# Mark entire module (in conftest.py or at module level)
pytestmark = pytest.mark.slow
```

#### Benefits

- **Faster Feedback**: Common failures are caught in 1-2 minutes instead of 10+ minutes
- **Resource Efficiency**: Expensive slow tests only run when fast tests pass
- **Clear Failure Reporting**: Separate test reports make it easy to identify which category failed
- **PR Comments**: Immediate feedback on fast test failures via GitHub comments

#### Workflow Structure

```yaml
jobs:
  fast-tests:
    timeout-minutes: 2
    steps:
      - Run pytest -m "not slow"
      - Generate fast-test-results.xml
      - Comment on PR if failed

  slow-tests:
    needs: fast-tests  # Only runs if fast-tests succeed
    timeout-minutes: 10
    steps:
      - Build C++ components
      - Run pytest -m slow
      - Run C++ tests
      - Generate slow-test-results.xml
```

## Other Workflows

### Code Quality
- `checks.yml` - Main CI workflow (linting, tests, benchmarks)
- `claude-review-*.yml` - AI-powered code review workflows

### Build & Deploy
- `build-image.yml` - Docker image builds
- `deploy-mettascope.yml` - Mettascope deployment
- `sky-train.yml` - SkyPilot training job launcher

### Maintenance
- `stale.yml` - Mark and close stale issues/PRs
- `assign-dependabot.yml` - Auto-assign Dependabot PRs
- `analyze-pr.yml` - PR size and complexity analysis

### Development Tools
- `generate-newsletter.yml` - Generate development newsletter
- `asana-integration.yml` - Sync with Asana project management

## Best Practices

1. **Test Categorization**: Be thoughtful about marking tests as slow. Only mark tests that:
   - Take more than a few seconds to run
   - Require heavy computation or I/O
   - Need complex setup/teardown
   - Depend on external services

2. **Test Reports**: Both fast and slow test jobs generate:
   - JUnit XML reports for CI visualization
   - JSON reports for detailed analysis
   - These are uploaded as artifacts and retained for 3 days

3. **Local Testing**: Developers can run test categories locally:
   ```bash
   # Run only fast tests
   pytest -m "not slow"

   # Run only slow tests
   pytest -m slow

   # Run all tests
   pytest
   ```

4. **Debugging CI Failures**: Test artifacts are available in the Actions tab:
   - Download `fast-test-results` for fast test failures
   - Download `slow-test-results` for slow test failures
   - JSON reports contain detailed failure information
