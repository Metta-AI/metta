"""UI tests for SkyDeck dashboard using Playwright."""

import pytest
import pytest_asyncio

from .utils import DashboardServer, screenshot_on_failure, wait_for_element


@pytest_asyncio.fixture
async def dashboard(db_with_experiments):
    """Start dashboard server with test data."""
    async with DashboardServer(db_path=db_with_experiments.db_path) as server:
        yield server


@pytest.mark.asyncio
async def test_dashboard_loads(dashboard, page):
    """Test that the dashboard loads successfully."""
    try:
        # Navigate to dashboard
        await page.goto(dashboard.base_url)

        # Wait for the page to load
        await wait_for_element(page, "h1")

        # Check title
        title = await page.text_content("h1")
        assert "SkyDeck" in title

        # Check that the experiments table exists
        await wait_for_element(page, "#experiments-table")

        # Check that controls are present
        await wait_for_element(page, ".controls")
        new_experiment_btn = await page.query_selector('button:has-text("New Experiment")')
        assert new_experiment_btn is not None

    except Exception:
        await screenshot_on_failure(page, "test_dashboard_loads")
        raise


@pytest.mark.asyncio
async def test_experiments_table_renders(dashboard, page):
    """Test that experiments are displayed in the table."""
    try:
        await page.goto(dashboard.base_url)

        # Wait for table to load
        await wait_for_element(page, "#experiments-tbody")

        # Wait a moment for data to load
        await page.wait_for_timeout(1000)

        # Check that experiments are displayed
        rows = await page.query_selector_all("tr.main-row")
        assert len(rows) > 0, "No experiment rows found"

        # Check that we have our test experiments
        assert len(rows) == 3, f"Expected 3 experiments, found {len(rows)}"

        # Check that experiment IDs are visible
        for i, exp_id in enumerate(["test_exp_1", "test_exp_2", "test_exp_3"]):
            cell = await page.query_selector(f"tr.main-row:nth-child({(i + 1) * 2}) .col-id")
            if cell:
                text = await cell.text_content()
                assert exp_id in text, f"Expected {exp_id} in cell, got {text}"

    except Exception:
        await screenshot_on_failure(page, "test_experiments_table_renders")
        raise


@pytest.mark.asyncio
async def test_expand_experiment_row(dashboard, page):
    """Test expanding an experiment row to see details."""
    try:
        await page.goto(dashboard.base_url)

        # Wait for table to load
        await wait_for_element(page, "#experiments-tbody")
        await page.wait_for_timeout(1000)

        # Get first experiment row
        first_row = await page.query_selector("tr.main-row")
        assert first_row is not None, "No experiment rows found"

        # Click to expand
        await first_row.click()

        # Wait for expanded section to appear
        await page.wait_for_timeout(500)

        # Check that expanded row is visible
        expanded_row = await page.query_selector("tr.expanded-row.show")
        assert expanded_row is not None, "Expanded row did not appear"

        # Check that configuration section exists
        config_section = await page.query_selector(".detail-section h3:has-text('Configuration')")
        assert config_section is not None, "Configuration section not found"

        # Check that job history section exists
        jobs_section = await page.query_selector(".detail-section h3:has-text('Job History')")
        assert jobs_section is not None, "Job History section not found"

    except Exception:
        await screenshot_on_failure(page, "test_expand_experiment_row")
        raise


@pytest.mark.asyncio
async def test_flag_columns_displayed(dashboard, page):
    """Test that flag columns are dynamically created."""
    try:
        await page.goto(dashboard.base_url)

        # Wait for table to load
        await wait_for_element(page, "#experiments-thead")
        await page.wait_for_timeout(1000)

        # Check for flag columns
        # We expect columns for the flags we added in test data
        flag_headers = await page.query_selector_all("th.col-flag")
        assert len(flag_headers) > 0, "No flag columns found"

        # Check that our specific flags are present
        all_headers_text = []
        for header in flag_headers:
            text = await header.get_attribute("title")
            if text:
                all_headers_text.append(text)

        # Should have columns for ppo.enabled and core_resnet_layers
        assert any("ppo" in h.lower() for h in all_headers_text), (
            f"PPO flag column not found. Headers: {all_headers_text}"
        )
        assert any("resnet" in h.lower() for h in all_headers_text), (
            f"ResNet flag column not found. Headers: {all_headers_text}"
        )

    except Exception:
        await screenshot_on_failure(page, "test_flag_columns_displayed")
        raise


@pytest.mark.asyncio
async def test_create_experiment_modal(dashboard, page):
    """Test opening the create experiment modal."""
    try:
        await page.goto(dashboard.base_url)

        # Click new experiment button
        await page.click('button:has-text("New Experiment")')

        # Wait for modal to appear
        await wait_for_element(page, "#create-modal.show")

        # Check modal content
        modal_title = await page.text_content("#create-modal h2")
        assert "Create New Experiment" in modal_title

        # Check form fields exist
        exp_id_input = await page.query_selector("#exp-id")
        assert exp_id_input is not None, "Experiment ID input not found"

        exp_name_input = await page.query_selector("#exp-name")
        assert exp_name_input is not None, "Experiment name input not found"

        # Close modal
        await page.click("#create-modal .close")
        await page.wait_for_timeout(300)

        # Check modal is hidden
        modal = await page.query_selector("#create-modal.show")
        assert modal is None, "Modal did not close"

    except Exception:
        await screenshot_on_failure(page, "test_create_experiment_modal")
        raise


@pytest.mark.asyncio
async def test_action_buttons_present(dashboard, page):
    """Test that action buttons are present in the table."""
    try:
        await page.goto(dashboard.base_url)

        # Wait for table to load
        await wait_for_element(page, "#experiments-tbody")
        await page.wait_for_timeout(1000)

        # Get first experiment row
        first_row = await page.query_selector("tr.main-row")
        assert first_row is not None

        # Check for action buttons
        start_btn = await first_row.query_selector('button:has-text("Start")')
        edit_btn = await first_row.query_selector('button:has-text("Edit")')
        delete_btn = await first_row.query_selector('button:has-text("Delete")')

        # At least one of start/stop should be present
        assert start_btn is not None, "Start button not found"
        assert edit_btn is not None, "Edit button not found"
        assert delete_btn is not None, "Delete button not found"

    except Exception:
        await screenshot_on_failure(page, "test_action_buttons_present")
        raise


@pytest.mark.asyncio
async def test_health_status_displayed(dashboard, page):
    """Test that health status is displayed."""
    try:
        await page.goto(dashboard.base_url)

        # Wait for health status
        await wait_for_element(page, "#health-status")

        # Check status text
        status_text = await page.text_content("#status-text")
        assert status_text is not None
        assert "experiment" in status_text.lower()

    except Exception:
        await screenshot_on_failure(page, "test_health_status_displayed")
        raise


@pytest.mark.asyncio
async def test_jobs_section_exists(dashboard, page):
    """Test that SkyPilot jobs section exists."""
    try:
        await page.goto(dashboard.base_url)

        # Wait for jobs section
        await wait_for_element(page, "#jobs-section")

        # Check section title
        title = await page.query_selector("#jobs-section h2")
        title_text = await title.text_content()
        assert "SkyPilot" in title_text or "Job" in title_text

        # Check jobs table exists
        jobs_table = await page.query_selector("#jobs-table")
        assert jobs_table is not None

    except Exception:
        await screenshot_on_failure(page, "test_jobs_section_exists")
        raise
