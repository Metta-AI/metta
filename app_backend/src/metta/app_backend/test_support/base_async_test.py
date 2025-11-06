"""Base class for tests requiring function-scoped async fixtures."""

import asyncio
import typing

import fastapi
import fastapi.testclient
import pytest
import pytest_asyncio

import metta.app_backend.metta_repo


class BaseAsyncTest:
    """Base class for tests that require function-scoped async fixtures.

    This base class provides function-scoped fixtures for test classes that:
    - Need proper async cleanup of database connections after each test
    - Test concurrent async operations requiring fresh connections
    - Contain async tests that directly interact with the database

    Using function scope ensures test isolation and prevents connection pool issues.
    """

    @pytest_asyncio.fixture(scope="function")
    async def stats_repo(self, db_uri: str) -> typing.AsyncGenerator[metta.app_backend.metta_repo.MettaRepo, None]:
        """Create a MettaRepo instance with async cleanup for the test database."""
        repo = metta.app_backend.metta_repo.MettaRepo(db_uri)
        yield repo
        # Ensure pool is closed gracefully
        if repo._pool is not None:
            try:
                await repo._pool.close()
            except (RuntimeError, asyncio.CancelledError):
                # Event loop might be closed or tasks cancelled, ignore
                pass

    @pytest.fixture(scope="function")
    def test_app(self, stats_repo: metta.app_backend.metta_repo.MettaRepo) -> fastapi.FastAPI:
        """Create a test FastAPI app with dependency injection."""
        import metta.app_backend.server

        return metta.app_backend.server.create_app(stats_repo)

    @pytest.fixture(scope="function")
    def test_client(self, test_app: fastapi.FastAPI) -> fastapi.testclient.TestClient:
        """Create a test client."""
        return fastapi.testclient.TestClient(test_app)
