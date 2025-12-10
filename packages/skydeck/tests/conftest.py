"""Pytest configuration and fixtures for SkyDeck tests."""

import asyncio
import os
import tempfile

import pytest
import pytest_asyncio
from playwright.async_api import async_playwright
from skydeck.database import Database
from skydeck.desired_state import DesiredStateManager
from skydeck.models import CreateExperimentRequest, DesiredState


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def temp_db():
    """Create a temporary database for testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)

    db = Database(path)
    await db.connect()

    yield db

    await db.close()
    os.unlink(path)


@pytest_asyncio.fixture
async def db_with_experiments(temp_db):
    """Create a database with sample experiments."""
    dsm = DesiredStateManager(temp_db)

    # Create sample experiments
    experiments = [
        CreateExperimentRequest(
            id="test_exp_1",
            name="Test Experiment 1",
            flags={
                "trainer.losses.ppo.enabled": True,
                "policy_architecture.core_resnet_layers": 4,
            },
            base_command="lt",
            run_name="daveey.test1",
            nodes=1,
            gpus=4,
            desired_state=DesiredState.STOPPED,
            description="Test experiment 1",
            group="test_group",
            order=0,
        ),
        CreateExperimentRequest(
            id="test_exp_2",
            name="Test Experiment 2",
            flags={
                "trainer.losses.ppo.enabled": False,
                "policy_architecture.core_resnet_layers": 16,
            },
            base_command="lt",
            run_name="daveey.test2",
            nodes=4,
            gpus=8,
            desired_state=DesiredState.STOPPED,
            description="Test experiment 2",
            group="test_group",
            order=1,
        ),
        CreateExperimentRequest(
            id="test_exp_3",
            name="Test Experiment 3",
            flags={
                "trainer.losses.ppo.enabled": True,
                "policy_architecture.core_resnet_layers": 64,
            },
            base_command="lt",
            run_name="daveey.test3",
            nodes=2,
            gpus=4,
            desired_state=DesiredState.STOPPED,
            description="Test experiment 3",
            order=0,
        ),
    ]

    for exp_request in experiments:
        await dsm.create_experiment(exp_request)

    yield temp_db


@pytest_asyncio.fixture
async def browser():
    """Create a browser instance for Playwright tests."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        yield browser
        await browser.close()


@pytest_asyncio.fixture
async def page(browser):
    """Create a new page for each test."""
    context = await browser.new_context()
    page = await context.new_page()
    yield page
    await page.close()
    await context.close()
