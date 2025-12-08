"""Test utilities for SkyDeck dashboard testing."""

import asyncio
import subprocess
import time
from pathlib import Path

import httpx


class DashboardServer:
    """Context manager for running the SkyDeck dashboard server during tests."""

    def __init__(self, db_path: str, host: str = "127.0.0.1", port: int = 8765):
        """Initialize dashboard server.

        Args:
            db_path: Path to database file
            host: Host to bind to
            port: Port to bind to (use non-standard port for testing)
        """
        self.db_path = db_path
        self.host = host
        self.port = port
        self.process = None
        self.base_url = f"http://{host}:{port}"

    async def __aenter__(self):
        """Start the dashboard server."""
        # Start server process
        self.process = subprocess.Popen(
            [
                "python",
                "-m",
                "skydeck.run",
                "--host",
                self.host,
                "--port",
                str(self.port),
                "--db-path",
                self.db_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for server to be ready
        await self._wait_for_server()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Stop the dashboard server."""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()

    async def _wait_for_server(self, timeout: int = 10):
        """Wait for server to be ready.

        Args:
            timeout: Maximum time to wait in seconds
        """
        start = time.time()
        async with httpx.AsyncClient() as client:
            while time.time() - start < timeout:
                try:
                    response = await client.get(f"{self.base_url}/api/health")
                    if response.status_code == 200:
                        return
                except (httpx.ConnectError, httpx.ReadTimeout):
                    await asyncio.sleep(0.1)
                    continue

        raise TimeoutError(f"Server did not start within {timeout} seconds")


async def wait_for_element(page, selector: str, timeout: int = 5000):
    """Wait for an element to appear on the page.

    Args:
        page: Playwright page
        selector: CSS selector
        timeout: Timeout in milliseconds
    """
    await page.wait_for_selector(selector, timeout=timeout)


async def screenshot_on_failure(page, test_name: str, screenshot_dir: Path = None):
    """Take a screenshot on test failure.

    Args:
        page: Playwright page
        test_name: Name of the test
        screenshot_dir: Directory to save screenshots
    """
    if screenshot_dir is None:
        screenshot_dir = Path(__file__).parent / "screenshots"

    screenshot_dir.mkdir(parents=True, exist_ok=True)
    screenshot_path = screenshot_dir / f"{test_name}_failure.png"

    await page.screenshot(path=str(screenshot_path))
    print(f"Screenshot saved to: {screenshot_path}")
