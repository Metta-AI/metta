"""Base class for tests requiring function-scoped async fixtures."""

from typing import AsyncGenerator

import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient

from metta.app_backend.metta_repo import MettaRepo


class BaseAsyncTest:
    """Base class for tests that require function-scoped async fixtures.

    This base class provides function-scoped fixtures for test classes that:
    - Need proper async cleanup of database connections after each test
    - Test concurrent async operations requiring fresh connections
    - Contain async tests that directly interact with the database

    Using function scope ensures test isolation and prevents connection pool issues.
    """

    @pytest_asyncio.fixture(scope="function")
    async def stats_repo(self, db_uri: str) -> AsyncGenerator[MettaRepo, None]:
        """Create a MettaRepo instance with async cleanup for the test database."""
        repo = MettaRepo(db_uri)
        yield repo
        # Ensure pool is closed gracefully
        if repo._pool is not None:
            try:
                await repo._pool.close()
            except RuntimeError:
                # Event loop might be closed, ignore
                pass

    @pytest.fixture(scope="function")
    def test_app(self, stats_repo: MettaRepo) -> FastAPI:
        """Create a test FastAPI app with dependency injection."""
        from metta.app_backend.server import create_app

        return create_app(stats_repo)

    @pytest.fixture(scope="function")
    def test_client(self, test_app: FastAPI) -> TestClient:
        """Create a test client."""
        return TestClient(test_app)
