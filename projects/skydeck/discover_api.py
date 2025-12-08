"""Discover SkyPilot API endpoints."""

import http.cookiejar
import json
import urllib.request

from skydeck.services import ServiceEndpoints


def load_cookies():
    """Load cookies from ~/.sky/cookies.txt."""
    jar = http.cookiejar.MozillaCookieJar()
    jar.load("/Users/daveey/.sky/cookies.txt", ignore_discard=True, ignore_expires=True)
    return jar


def try_endpoint(path):
    """Try an API endpoint and return the response."""
    url = f"{ServiceEndpoints.SKYPILOT_API}{path}"
    jar = load_cookies()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))

    try:
        req = urllib.request.Request(url)
        req.add_header("Accept", "application/json")
        response = opener.open(req, timeout=10)
        data = response.read().decode("utf-8")
        try:
            parsed = json.loads(data)
            print(f"✓ {path}")
            print(f"  Status: {response.status}")
            print(f"  Response: {json.dumps(parsed, indent=2)[:200]}...")
            return True
        except json.JSONDecodeError:
            print(f"✗ {path} - Not JSON: {data[:100]}")
            return False
    except urllib.error.HTTPError as e:
        print(f"✗ {path} - HTTP {e.code}: {e.reason}")
        return False
    except Exception as e:
        print(f"✗ {path} - Error: {e}")
        return False


if __name__ == "__main__":
    endpoints = [
        # Try different API paths
        "/api/jobs",
        "/api/v1/jobs",
        "/api/v2/jobs",
        "/api/managed_jobs",
        "/api/queue",
        "/api/status",
        "/jobs",
        "/queue",
        "/v1/jobs",
        "/v1/managed_jobs",
        # Try specific job
        "/api/jobs/9588",
        "/api/v1/jobs/9588",
        "/jobs/9588",
        # Try listing endpoints
        "/api",
        "/api/v1",
        "",
    ]

    print("Testing SkyPilot API endpoints...")
    print("=" * 60)

    for endpoint in endpoints:
        try_endpoint(endpoint)
        print()
