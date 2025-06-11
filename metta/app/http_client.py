from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import httpx
from fastapi.testclient import TestClient


class HttpClient(ABC):
    """Abstract HTTP client interface."""

    @abstractmethod
    async def get(self, url: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Make a GET request."""
        pass

    @abstractmethod
    async def post(self, url: str, json: Optional[Dict[str, Any]] = None) -> Any:
        """Make a POST request."""
        pass

    async def close(self):
        """Close the HTTP client. Optional method."""
        pass


class HttpxClient(HttpClient):
    """HTTP client implementation using httpx."""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()

    async def get(self, url: str, params: Optional[Dict[str, Any]] = None) -> httpx.Response:
        """Make a GET request."""
        full_url = f"{self.base_url}{url}"
        return await self._client.get(full_url, params=params)

    async def post(self, url: str, json: Optional[Dict[str, Any]] = None) -> httpx.Response:
        """Make a POST request."""
        full_url = f"{self.base_url}{url}"
        return await self._client.post(full_url, json=json)


class FastAPITestClientAdapter(HttpClient):
    """HTTP client adapter for FastAPI TestClient."""

    def __init__(self, test_client: TestClient):
        self.test_client = test_client

    async def get(self, url: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Make a GET request using TestClient."""
        return self.test_client.get(url, params=params)

    async def post(self, url: str, json: Optional[Dict[str, Any]] = None) -> Any:
        """Make a POST request using TestClient."""
        return self.test_client.post(url, json=json)
