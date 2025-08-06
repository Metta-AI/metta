#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
Upload coverage reports to Codecov using the official uploader CLI,
automatically downloading the binary if necessary.
Each coverage file is uploaded separately with the appropriate flag.
Only supports macOS and Linux.
"""

import os
import platform
import subprocess
import sys
from pathlib import Path
from urllib.request import urlretrieve

# Configuration
CODECOV_BIN = Path(".github/scripts/codecov")
CODECOV_URL_BASE = "https://uploader.codecov.io/latest"
DEFAULT_SUBPACKAGES = ["app_backend", "agent", "mettagrid", "common"]


def get_platform_binary_path() -> tuple[Path, str]:
    """Get the appropriate Codecov binary path and download URL (macOS and Linux only)."""
    system = platform.system().lower()

    if system == "darwin":
        platform_name = "macos"
    elif system == "linux":
        platform_name = "linux"
    else:
        raise RuntimeError(f"Unsupported platform: {system}")

    binary_name = "codecov"
    binary_path = CODECOV_BIN / binary_name
    download_url = f"{CODECOV_URL_BASE}/{platform_name}/{binary_name}"

    return binary_path, download_url


def ensure_codecov_binary() -> Path:
    """Download the Codecov uploader binary if not already present."""
    CODECOV_BIN.mkdir(parents=True, exist_ok=True)

    binary_path, download_url = get_platform_binary_path()

    if not binary_path.exists():
        print(f"⬇️  Downloading Codecov uploader from {download_url}")
        urlretrieve(download_url, binary_path)
        binary_path.chmod(0o755)
    else:
        print(f"✅ Codecov uploader already exists at {binary_path}")

    return binary_path


def find_coverage_files(base_path: Path) -> list[Path]:
    """Find all coverage XML files in the given directory."""
    coverage_files = []

    # Look for any coverage*.xml files
    for pattern in ["coverage.xml", "coverage-*.xml"]:
        coverage_files.extend(base_path.glob(pattern))

    return sorted(coverage_files)  # Sort for consistent ordering


def run_codecov(binary: Path, token: str, coverage_files: list[tuple[Path, str]]):
    """Upload each coverage file with the correct flag."""
    for path, flag in coverage_files:
        if not path.exists():
            print(f"⚠️  Skipping missing coverage file: {path}")
            continue

        args = [
            str(binary),
            "-t",
            token,
            "--disable",
            "search",
            "-f",
            str(path),
            "-F",
            flag,
            "--name",
            flag,
        ]

        print(f"📤 Uploading {path} with flag '{flag}'...")
        try:
            subprocess.run(args, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Upload failed for {path}: {e}")
            sys.exit(1)


def main():
    token = os.environ.get("CODECOV_TOKEN") or (sys.argv[1] if len(sys.argv) > 1 else None)
    if not token:
        print("❌ Error: CODECOV_TOKEN is required (pass via env or argv)")
        sys.exit(1)

    subpackages = os.environ.get("SUBPACKAGES", " ".join(DEFAULT_SUBPACKAGES)).split()

    # Build list of coverage files to upload
    coverage_files = []

    for pkg in subpackages:
        # Determine base path
        base_path = Path(".") if pkg == "core" else Path(pkg)

        # Find all coverage files in this package
        found_files = find_coverage_files(base_path)

        if not found_files:
            print(f"⚠️  No coverage files found for package: {pkg}")
            continue

        # Add each found file with appropriate flag
        for coverage_file in found_files:
            # Extract suffix if present (e.g., "fast" from "coverage-fast.xml")
            if coverage_file.stem == "coverage":
                flag = pkg
            else:
                suffix = coverage_file.stem.replace("coverage-", "")
                flag = f"{pkg}-{suffix}"

            coverage_files.append((coverage_file, flag))

    if not coverage_files:
        print("❌ No coverage files found to upload")
        sys.exit(1)

    print("🔍 Codecov Upload Plan:")
    for path, flag in coverage_files:
        print(f"   - {path} → flag: {flag}")
    print()

    binary_path = ensure_codecov_binary()
    run_codecov(binary_path, token, coverage_files)


if __name__ == "__main__":
    main()
