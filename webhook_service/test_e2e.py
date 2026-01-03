#!/usr/bin/env python3
"""End-to-end test for webhook service."""

import asyncio
import json
import subprocess
from pathlib import Path

import httpx


async def test_webhook_service():
    """Test the webhook service end-to-end."""
    # Get environment variables
    import os
    env = os.environ.copy()

    # Start server
    print("Starting webhook service...")
    server_process = subprocess.Popen(
        ["uv", "run", "python", "-m", "github_webhook.app"],
        cwd=Path(__file__).parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )

    try:
        # Wait for server to start
        print("Waiting for server to start...")
        await asyncio.sleep(3)

        # Test health endpoint
        print("\n1. Testing health endpoint...")
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8000/health")
            print(f"   Status: {response.status_code}")
            print(f"   Response: {response.json()}")
            assert response.status_code == 200

        # Test ping event
        print("\n2. Testing ping event...")
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:8000/webhooks/github",
                headers={
                    "Content-Type": "application/json",
                    "X-GitHub-Event": "ping",
                    "X-GitHub-Delivery": "test-delivery-123",
                },
                json={"zen": "test"},
            )
            print(f"   Status: {response.status_code}")
            print(f"   Response: {response.json()}")
            assert response.status_code == 200
            assert response.json()["status"] == "ok"

        # Test pull_request opened event
        print("\n3. Testing pull_request opened event...")
        import os
        if os.getenv("ASANA_PAT"):
            print("   ASANA_PAT is configured - will attempt to create Asana task")
        else:
            print("   ASANA_PAT not configured - task creation will be skipped")
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "http://localhost:8000/webhooks/github",
                headers={
                    "Content-Type": "application/json",
                    "X-GitHub-Event": "pull_request",
                    "X-GitHub-Delivery": "test-delivery-456",
                },
                json={
                    "action": "opened",
                    "pull_request": {
                        "number": 9999,
                        "title": "Test PR",
                        "html_url": "https://github.com/Metta-AI/metta/pull/9999",
                        "user": {"login": "testuser"},
                        "assignee": None,
                    },
                    "repository": {"full_name": "Metta-AI/metta"},
                },
            )
            print(f"   Status: {response.status_code}")
            print(f"   Response: {json.dumps(response.json(), indent=2)}")
            assert response.status_code == 200

        print("\n✅ All tests passed!")

    finally:
        # Stop server
        print("\nStopping server...")
        server_process.terminate()
        try:
            server_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server_process.kill()
            server_process.wait()


if __name__ == "__main__":
    asyncio.run(test_webhook_service())

