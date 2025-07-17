#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

"""
Upload coverage reports to Codecov using their API directly.
This avoids downloading the CLI tool.
"""

import gzip
import json
import os
import subprocess
import sys
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def get_git_info() -> dict[str, str]:
    """Get current git commit and branch information."""

    def run_git(cmd: list[str]) -> str:
        result = subprocess.run(["git"] + cmd, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else ""

    return {
        "commit": run_git(["rev-parse", "HEAD"]),
        "branch": run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "slug": os.environ.get("CODECOV_SLUG") or os.environ.get("GITHUB_REPOSITORY", ""),
    }


def create_upload_request(token: str, git_info: dict[str, str]) -> dict[str, str]:
    """Create an upload request with Codecov to get upload URL."""
    url = "https://api.codecov.io/upload/v4"

    data = {
        "commit": git_info["commit"],
        "branch": git_info["branch"],
        "slug": git_info["slug"],
        "token": token,
    }

    req = Request(
        url,
        data=json.dumps(data).encode(),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )

    try:
        with urlopen(req) as response:
            return json.loads(response.read())
    except HTTPError as e:
        print(f"❌ Failed to create upload request: {e}")
        print(f"Response: {e.read().decode()}")
        raise


def upload_coverage_file(upload_url: str, file_path: Path, flag: str, name: str) -> bool:
    """Upload a coverage file to the provided URL."""
    if not file_path.exists():
        print(f"⚠️  Skipping {name} - coverage file not found: {file_path}")
        return False

    print(f"📤 Uploading {name} coverage from {file_path}...")

    # Read and compress the coverage file
    with open(file_path, "rb") as f:
        coverage_data = f.read()

    compressed_data = gzip.compress(coverage_data)

    # Prepare the upload
    headers = {
        "Content-Type": "text/plain",
        "Content-Encoding": "gzip",
        "x-amz-meta-flag": flag,
        "x-amz-meta-name": name,
    }

    req = Request(upload_url, data=compressed_data, headers=headers, method="PUT")

    try:
        with urlopen(req) as response:
            if response.status == 200:
                print(f"✅ Successfully uploaded {name} coverage")
                return True
            else:
                print(f"❌ Failed to upload {name} coverage: HTTP {response.status}")
                return False
    except Exception as e:
        print(f"❌ Error uploading {name} coverage: {e}")
        return False


def main():
    """Main function to upload all coverage reports."""
    # Get configuration

    token = os.environ.get("CODECOV_TOKEN")
    if not token and len(sys.argv) > 1:
        token = sys.argv[1]

    if not token:
        print("❌ Error: CODECOV_TOKEN is required but missing.")
        sys.exit(1)

    print(f"✅ Using CODECOV_TOKEN from {'env' if 'CODECOV_TOKEN' in os.environ else 'argv'}")

    subpackages = os.environ.get("SUBPACKAGES", "app_backend agent mettagrid common").split()

    if not token:
        print("❌ Error: CODECOV_TOKEN is required")
        sys.exit(1)

    print("🔍 Codecov Upload Configuration:")
    print(f"   Subpackages: {', '.join(subpackages)}")
    print()

    # Get git information
    git_info = get_git_info()
    print(f"📍 Git info: {git_info['slug']} @ {git_info['commit'][:8]}")
    print()

    # Create upload request
    try:
        upload_info = create_upload_request(token, git_info)
        upload_url = upload_info["url"]
    except Exception as e:
        print(f"❌ Failed to get upload URL: {e}")
        sys.exit(1)

    # Prepare all coverage files
    coverage_files = [(Path("./coverage.xml"), "core", "core")]
    for package in subpackages:
        coverage_files.append((Path(f"./{package}/coverage.xml"), package, package))

    # Upload each file
    success_count = 0
    for file_path, flag, name in coverage_files:
        if upload_coverage_file(upload_url, file_path, flag, name):
            success_count += 1

    print()
    print(f"📊 Coverage upload summary: {success_count}/{len(coverage_files)} files uploaded")

    if success_count == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
