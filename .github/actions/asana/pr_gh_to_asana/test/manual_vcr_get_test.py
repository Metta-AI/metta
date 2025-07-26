#!/usr/bin/env python3
from pathlib import Path

import requests
import vcr

# Directory containing the run files
run_dir = Path("run_16393962816/http-interactions-16393962816")

# Find the cassette file
cassette_file = None
for f in run_dir.iterdir():
    if f.name.startswith("http_interactions_") and f.name.endswith(".yaml"):
        cassette_file = f
        break

if not cassette_file:
    print("No cassette file found in run_16393962816.")
    exit(1)

print(f"Using cassette: {cassette_file}")

# Set up VCR to match the test config
my_vcr = vcr.VCR(
    record_mode="none",
    filter_headers=["Authorization", "User-Agent"],
    match_on=["uri", "method"],
    filter_query_parameters=["access_token"],
)

url = "https://api.github.com/repos/Metta-AI/metta/pulls/1572"

try:
    with my_vcr.use_cassette(str(cassette_file)):
        print(f"Attempting GET {url} with VCR in replay mode...")
        r = requests.get(url)
        print(f"Status: {r.status_code}")
        print(f"Body: {r.text[:200]}...")
except Exception as e:
    print(f"Exception: {e}")
    import traceback

    traceback.print_exc()
