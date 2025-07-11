"""Integration tests for SQL route concurrency behavior."""

import asyncio
import time

import httpx
import pytest
from fastapi import FastAPI

from tests.base_async_test import BaseAsyncTest


class TestSQLConcurrency(BaseAsyncTest):
    """Tests for SQL route concurrency to validate async behavior."""

    @pytest.fixture(scope="function")
    def base_url(self, test_app: FastAPI) -> str:
        """Get the base URL for the test app."""
        # For this test, we'll use httpx.AsyncClient with the app directly
        return "http://test"

    @pytest.mark.asyncio
    async def test_sql_query_concurrency(self, test_app: FastAPI, auth_headers: dict) -> None:
        """
        Test that slow SQL queries don't block fast queries.

        This validates the async conversion by running a pg_sleep(3) query
        in parallel with a fast SELECT 1 query and ensuring the fast query
        completes immediately while the slow query takes ~3 seconds.
        """
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=test_app), base_url="http://test") as client:
            # Define queries
            slow_query = {"query": "SELECT pg_sleep(3), 'slow' as query_type"}
            fast_query = {"query": "SELECT 1 as result, 'fast' as query_type"}

            # Record start time
            start_time = time.time()

            # Create tasks for concurrent execution
            slow_task = asyncio.create_task(client.post("/sql/query", json=slow_query, headers=auth_headers))

            # Wait a tiny bit to ensure slow query starts first
            await asyncio.sleep(0.1)

            fast_task = asyncio.create_task(client.post("/sql/query", json=fast_query, headers=auth_headers))

            # Wait for the fast query to complete
            fast_response = await fast_task
            fast_completion_time = time.time()

            # Validate fast query completed quickly (< 1 second)
            fast_duration = fast_completion_time - start_time
            assert fast_duration < 1.0, f"Fast query took {fast_duration:.2f}s, should be < 1s"
            assert fast_response.status_code == 200

            fast_data = fast_response.json()
            assert fast_data["columns"] == ["result", "query_type"]
            assert fast_data["rows"] == [[1, "fast"]]
            assert fast_data["row_count"] == 1

            # Wait for slow query to complete
            slow_response = await slow_task
            slow_completion_time = time.time()

            # Validate slow query took approximately 3 seconds
            slow_duration = slow_completion_time - start_time
            assert 2.5 <= slow_duration <= 4.0, f"Slow query took {slow_duration:.2f}s, should be ~3s"
            assert slow_response.status_code == 200

            slow_data = slow_response.json()
            assert slow_data["columns"] == ["pg_sleep", "query_type"]
            assert slow_data["rows"] == [["", "slow"]]  # pg_sleep returns empty string
            assert slow_data["row_count"] == 1

            print(f"✓ Fast query completed in {fast_duration:.2f}s")
            print(f"✓ Slow query completed in {slow_duration:.2f}s")
            print(f"✓ Concurrency validated: {slow_duration / fast_duration:.1f}x difference")

    @pytest.mark.asyncio
    async def test_multiple_concurrent_queries(self, test_app: FastAPI, auth_headers: dict) -> None:
        """
        Test multiple concurrent queries to further validate async behavior.

        Runs 5 fast queries and 1 slow query concurrently to ensure
        the slow query doesn't block any of the fast queries.
        """
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=test_app), base_url="http://test") as client:
            # One slow query
            slow_query = {"query": "SELECT pg_sleep(2), 'slow' as query_type"}

            # Multiple fast queries
            fast_queries = [{"query": f"SELECT {i} as query_id, 'fast' as query_type"} for i in range(1, 6)]

            start_time = time.time()

            # Start slow query first
            slow_task = asyncio.create_task(client.post("/sql/query", json=slow_query, headers=auth_headers))

            # Wait briefly then start all fast queries
            await asyncio.sleep(0.1)

            fast_tasks = [
                asyncio.create_task(client.post("/sql/query", json=query, headers=auth_headers))
                for query in fast_queries
            ]

            # Wait for all fast queries to complete
            fast_responses = await asyncio.gather(*fast_tasks)
            fast_completion_time = time.time()

            # All fast queries should complete quickly
            fast_duration = fast_completion_time - start_time
            assert fast_duration < 1.0, f"Fast queries took {fast_duration:.2f}s, should be < 1s"

            # Validate all fast responses
            for i, response in enumerate(fast_responses):
                assert response.status_code == 200
                data = response.json()
                assert data["columns"] == ["query_id", "query_type"]
                assert data["rows"] == [[i + 1, "fast"]]

            # Wait for slow query
            slow_response = await slow_task
            slow_completion_time = time.time()

            slow_duration = slow_completion_time - start_time
            assert 1.5 <= slow_duration <= 3.0, f"Slow query took {slow_duration:.2f}s, should be ~2s"
            assert slow_response.status_code == 200

            print(f"✓ 5 fast queries completed in {fast_duration:.2f}s")
            print(f"✓ 1 slow query completed in {slow_duration:.2f}s")
            print("✓ Async behavior confirmed: fast queries not blocked")


if __name__ == "__main__":
    # Simple test runner for debugging
    pytest.main([__file__, "-v", "-s"])
