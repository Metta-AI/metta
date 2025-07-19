#!/usr/bin/env python3
"""
Test script to demonstrate VCR functionality
"""

import requests
import vcr

# Configure VCR
my_vcr = vcr.VCR(
    cassette_library_dir="cassettes",
    record_mode=vcr.mode.ONCE,
    match_on=["uri", "method"],
    filter_headers=["authorization"],
    decode_compressed_response=True,
)


def test_http_request():
    """Test HTTP request with VCR"""
    url = "https://httpbin.org/get"

    with my_vcr.use_cassette("test_request.yaml"):
        response = requests.get(url)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200


if __name__ == "__main__":
    print("Testing VCR functionality...")
    success = test_http_request()
    print(f"Test {'passed' if success else 'failed'}")

    # Run again to test replay
    print("\nTesting VCR replay...")
    success = test_http_request()
    print(f"Replay test {'passed' if success else 'failed'}")
