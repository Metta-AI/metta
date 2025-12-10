# SkyDeck Testing Guide

## Overview

SkyDeck now includes comprehensive automated testing using Playwright, allowing you (Claude) to test the dashboard UI programmatically.

## What's Installed

- **Playwright**: Browser automation framework
- **pytest-playwright**: Pytest integration for Playwright
- **Chromium**: Headless browser for testing
- **Test fixtures**: Pre-configured database and server setup

## Quick Start

```bash
# Run all tests
uv run pytest

# Run only UI tests
uv run pytest tests/test_dashboard_ui.py -v

# Run specific test
uv run pytest tests/test_dashboard_ui.py::test_dashboard_loads -v

# Run with visible browser (for debugging)
# Edit tests/conftest.py: headless=False
uv run pytest tests/test_dashboard_ui.py -v
```

## Test Structure

### Fixtures (`tests/conftest.py`)

1. **Database Fixtures**
   - `temp_db`: Empty temporary database
   - `db_with_experiments`: Pre-populated with 3 test experiments

2. **Playwright Fixtures**
   - `browser`: Chromium browser instance
   - `page`: New browser page for each test
   - `dashboard`: Running server with test database

### Test Files

- `test_models.py`: Unit tests for data models
- `test_database.py`: Database operation tests
- `test_dashboard_ui.py`: End-to-end UI tests

## Available UI Tests

1. **test_dashboard_loads**: Verifies dashboard loads and displays header
2. **test_experiments_table_renders**: Checks experiment table with data
3. **test_expand_experiment_row**: Tests expandable row functionality
4. **test_flag_columns_displayed**: Verifies dynamic flag columns
5. **test_create_experiment_modal**: Tests modal open/close
6. **test_action_buttons_present**: Verifies Start/Edit/Delete buttons
7. **test_health_status_displayed**: Checks health status display
8. **test_clusters_section_exists**: Verifies clusters section

## How Tests Work

1. **Server Lifecycle**
   ```python
   async with DashboardServer(db_path=test_db.path) as server:
       # Server runs on http://127.0.0.1:8765
       # Automatically starts/stops
   ```

2. **Database Setup**
   ```python
   # Each test gets a clean database
   # Pre-populated with test experiments
   # Automatically cleaned up after test
   ```

3. **Browser Interaction**
   ```python
   await page.goto(dashboard.base_url)
   await page.click("button#my-button")
   text = await page.text_content("#result")
   assert "expected" in text
   ```

4. **Screenshots on Failure**
   - Automatically saved to `tests/screenshots/`
   - Named after the failing test
   - Useful for debugging

## Writing New Tests

### Template

```python
import pytest
from tests.utils import wait_for_element, screenshot_on_failure

@pytest.mark.asyncio
async def test_my_feature(dashboard, page):
    """Test description."""
    try:
        # Navigate
        await page.goto(dashboard.base_url)

        # Wait for element
        await wait_for_element(page, "#my-element")

        # Interact
        await page.click("button")
        await page.fill("input#name", "test")

        # Assert
        result = await page.text_content("#result")
        assert "success" in result

    except Exception as e:
        await screenshot_on_failure(page, "test_my_feature")
        raise
```

### Best Practices

1. **Always use try/except with screenshot_on_failure**
2. **Wait for elements** before interacting: `wait_for_element()`
3. **Use specific selectors**: ID > class > tag
4. **Add delays only when necessary**: `page.wait_for_timeout(1000)`
5. **Keep test data minimal**: 2-3 experiments is usually enough

## Debugging

### View Browser

Edit `tests/conftest.py`:
```python
browser = await p.chromium.launch(headless=False, slow_mo=1000)
```

### Print Page Content

```python
content = await page.content()
print(content)
```

### Check Screenshots

```bash
open tests/screenshots/
```

### Server Logs

Run with `-s` to see output:
```bash
uv run pytest -s tests/test_dashboard_ui.py
```

## CI/CD Ready

Tests are designed for CI:
- ✅ Headless mode by default
- ✅ Automatic cleanup
- ✅ No manual setup required
- ✅ Fast parallel execution
- ✅ Clear error messages with screenshots

## Performance

- Unit tests: ~0.01s each
- UI tests: ~2-5s each (includes server startup)
- Full suite: ~30s

## State Management

All test state is stored in `~/.skydeck/` (same as production):
- Database: `~/.skydeck/skydeck.db`
- Logs: `~/.skydeck/skydeck.log`
- PID file: `~/.skydeck/skydeck.pid`

Tests use temporary databases that don't affect your production data.

## Next Steps

To add group drag-and-drop tests:

```python
@pytest.mark.asyncio
async def test_drag_experiment_to_group(dashboard, page):
    """Test dragging experiment between groups."""
    await page.goto(dashboard.base_url)

    # Get experiment row
    row = await page.query_selector("tr.main-row[data-exp-id='test_exp_1']")

    # Drag to group
    target = await page.query_selector(".group-header[data-group='new_group']")

    await row.drag_to(target)

    # Verify
    # ... check group assignment
```

## Useful Selectors

```python
# Experiments table
"#experiments-table"
"tr.main-row"
"tr.expanded-row"

# Columns
".col-id"
".col-name"
".col-flag"

# Buttons
"button:has-text('Start')"
"button:has-text('Edit')"
"button:has-text('Delete')"

# Modals
"#create-modal.show"
"#flags-modal.show"

# Status
".status-badge.running"
".status-badge.stopped"
```

## Tips for Claude

When testing the dashboard:

1. **Start with simple tests**: Load page, check elements exist
2. **Build up complexity**: Click buttons, fill forms, verify results
3. **Use screenshots**: They show exactly what's wrong
4. **Test error cases**: Missing data, failed API calls, etc.
5. **Verify responsive behavior**: Table scrolling, modal sizing

You can now test UI changes without manual browser interaction!
