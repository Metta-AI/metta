# Stable Release System Tests

This directory contains behavioral tests for the stable release validation system. All tests use mocks and fakes to avoid real operations (no training, no network calls, no git operations).

## Test Coverage

### test_acceptance.py (6 tests)
Tests for acceptance criteria evaluation:
- ✅ All comparison operators (`>`, `>=`, `<`, `<=`, `==`, `!=`) work correctly
- ✅ Missing metrics are detected and fail appropriately
- ✅ Multiple failures are all reported
- ✅ Empty acceptance criteria always pass
- ✅ Rule normalization works correctly

### test_metrics.py (9 tests)
Tests for metrics extraction from logs:
- ✅ SPS extraction in different formats (`SPS: 1234` and `SPS=1234`)
- ✅ KSPS conversion to SPS (87.5 ksps → 87500 SPS)
- ✅ Mixed SPS/KSPS handling (picks max, tracks last)
- ✅ Evaluation success rate extraction
- ✅ W&B URL parsing (entity/project/run_id)
- ✅ W&B metric fetching (mocked API calls)
- ✅ Empty log handling
- ✅ Combined metric extraction

### test_tasks.py (9 tests)
Tests for Task execution logic:
- ✅ Timeout detection (exit code 124)
- ✅ Non-zero exit code handling
- ✅ Success case with no acceptance criteria
- ✅ Acceptance criteria passing
- ✅ Acceptance criteria failing
- ✅ Job ID capture
- ✅ Policy URI extraction from job_id
- ✅ Policy URI extraction from run= arg
- ✅ No policy_uri when no identifiers

### test_release_stable.py (8 tests)
Tests for release orchestration:
- ✅ State persistence to disk
- ✅ Version prefix handling (with/without "release_")
- ✅ Skip already-completed validations
- ✅ Retry failed validations
- ✅ Skip validations with missing dependencies
- ✅ Skip validations with failed dependencies
- ✅ Policy URI injection for EVALUATE tasks
- ✅ Exception handling during task execution

## Running Tests

### Run all stable release tests
```bash
uv run pytest tests/devops/stable/ -v
```

### Run specific test file
```bash
uv run pytest tests/devops/stable/test_acceptance.py -v
uv run pytest tests/devops/stable/test_metrics.py -v
uv run pytest tests/devops/stable/test_tasks.py -v
uv run pytest tests/devops/stable/test_release_stable.py -v
```

### Run specific test
```bash
uv run pytest tests/devops/stable/test_acceptance.py::test_all_operators_pass -v
```

### Run with coverage
```bash
uv run pytest tests/devops/stable/ --cov=devops.stable --cov-report=term-missing
```

## Test Philosophy

### No Real Operations
- **No subprocess execution**: LocalJob uses FakePopen
- **No network calls**: W&B/Asana/GitHub APIs are mocked
- **No file I/O**: Except to tmp_path fixtures
- **No git operations**: git commands are mocked
- **No training**: FakeJob provides predefined logs

### Fast Execution
All 32 tests run in ~5 seconds, making them suitable for:
- Pre-commit hooks
- CI pipelines
- TDD workflows

### Behavioral Focus
Tests verify:
- **Outcomes**, not implementation details
- **Error handling** and edge cases
- **Integration points** between components
- **State transitions** (pending → in_progress → completed/failed)

## Test Patterns

### Using tmp_path for State
```python
def test_state_persistence(tmp_path, monkeypatch):
    monkeypatch.setattr("devops.stable.release_stable.STATE_DIR", tmp_path)
    # ... test writes/reads state to tmp_path
```

### Mocking External Dependencies
```python
def test_with_wandb_mocked(monkeypatch):
    def fake_fetch(entity, project, run_id, metric_key, **kwargs):
        return 1.23

    monkeypatch.setattr(metrics_module, "fetch_wandb_metric", fake_fetch)
    # ... test uses mocked W&B
```

### Using FakePopen for LocalJob
```python
def test_localjob_success(tmp_path, monkeypatch, fake_popen):
    factory, instances = fake_popen
    output_lines = ["Starting job", "Complete", "Exit code: 0"]
    monkeypatch.setattr("subprocess.Popen", factory(output_lines=output_lines))
    # ... test runs LocalJob without real process
```

## Adding New Tests

### For new acceptance operators
Add to `test_acceptance.py`:
```python
def test_new_operator():
    metrics = {"value": 10}
    outcome, failures = evaluate_thresholds(metrics, [("value", "~=", 10)])
    assert outcome == "passed"
```

### For new metrics
Add to `test_metrics.py`:
```python
def test_extract_new_metric():
    log_text = "new_metric: 42.0"
    metrics = extract_metrics(log_text)
    assert metrics["new_metric"] == 42.0
```

### For new task types
Add to `test_tasks.py`:
```python
def test_new_task_behavior():
    task = NewTaskType(name="test", ...)
    # Mock job and verify behavior
```

### For new validation flows
Add to `test_release_stable.py`:
```python
def test_new_flow(tmp_path, monkeypatch):
    # Setup state and mocks
    # Run validation flow
    # Verify outcomes
```

## Known Test Warnings

- **SQLAlchemy MovedIn20Warning**: From SkyPilot dependencies (harmless)
- **ddtrace crypt deprecation**: From datadog tracer (harmless)
- **pytest-benchmark**: Disabled in parallel mode (expected)

These warnings do not affect test reliability and can be safely ignored.
