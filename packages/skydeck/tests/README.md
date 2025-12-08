# SkyDeck Dashboard Tests

This directory contains tests for the SkyDeck dashboard, including UI tests using Playwright.

## Setup

Tests are automatically configured when you install the project with dev dependencies:

```bash
cd packages/skydeck
uv sync
```

Playwright browsers should be installed automatically, but if needed:

```bash
python -m playwright install chromium
```

## Running Tests

### Run all tests

```bash
# From project root
cd packages/skydeck
uv run pytest

# Or from metta root
uv run pytest packages/skydeck/tests/
```

### Run specific test files

```bash
# Unit tests only (fast)
uv run pytest tests/test_models.py tests/test_database.py

# UI tests only (slower, requires running server)
uv run pytest tests/test_dashboard_ui.py

# Run with verbose output
uv run pytest -v

# Run specific test
uv run pytest tests/test_dashboard_ui.py::test_dashboard_loads -v
```

### Run with screenshots on failure

UI tests automatically take screenshots on failure and save them to `tests/screenshots/`.

### Watch mode

```bash
# Run tests on file changes
uv run pytest-watch
```

## Test Types

### Unit Tests
- `test_models.py`: Data model validation and logic
- `test_database.py`: Database operations

### UI Tests
- `test_dashboard_ui.py`: End-to-end UI tests with Playwright

UI tests:
1. Start a test server on port 8765
2. Create a temporary database with sample data
3. Use Playwright to interact with the UI
4. Take screenshots on failures for debugging

## Test Fixtures

### Database Fixtures
- `temp_db`: Empty temporary database
- `db_with_experiments`: Database pre-populated with 3 test experiments

### Playwright Fixtures
- `browser`: Chromium browser instance (session-scoped)
- `page`: New page for each test
- `dashboard`: Running dashboard server with test data

## Writing New Tests

### UI Test Example

```python
import pytest
from tests.utils import wait_for_element, screenshot_on_failure

@pytest.mark.asyncio
async def test_my_feature(dashboard, page):
    """Test description."""
    try:
        # Navigate to dashboard
        await page.goto(dashboard.base_url)

        # Wait for elements
        await wait_for_element(page, "#my-element")

        # Interact with page
        await page.click("button#my-button")

        # Assert results
        text = await page.text_content("#result")
        assert "expected" in text

    except Exception as e:
        await screenshot_on_failure(page, "test_my_feature")
        raise
```

### Database Test Example

```python
import pytest

@pytest.mark.asyncio
async def test_my_database_operation(temp_db):
    """Test description."""
    # Use temp_db for operations
    await temp_db.save_experiment(...)
    result = await temp_db.get_experiment(...)
    assert result is not None
```

## Debugging

### View screenshots
Failed UI tests save screenshots to `tests/screenshots/`.

### Run in headed mode
To see the browser during tests:

Edit `tests/conftest.py` and change:
```python
browser = await p.chromium.launch(headless=False)
```

### Slow down tests
Add delays to see what's happening:
```python
await page.wait_for_timeout(1000)  # Wait 1 second
```

### Check server logs
The test server output is captured. To see it, run with `-s`:
```bash
uv run pytest -s tests/test_dashboard_ui.py
```

## CI/CD

These tests are designed to run in CI environments:
- Headless mode by default
- Automatic cleanup of test databases
- Screenshot capture on failures
- No external dependencies beyond Playwright browsers

## Tips

- UI tests are slower than unit tests - run unit tests frequently, UI tests before commits
- Always use the `dashboard` fixture for UI tests to ensure proper server lifecycle
- Use `wait_for_element` instead of fixed timeouts when possible
- Take screenshots on failures to help debug issues
- Keep test data minimal but representative
