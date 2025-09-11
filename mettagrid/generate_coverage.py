#!/usr/bin/env -S uv run
"""
generate_coverage.py - Process Bazel C++ coverage reports

Combines Bazel's coverage.dat files for upload to Codecov.
"""

import sys
from pathlib import Path

# Simple text styling functions
_use_colors = True


def use_colors(enabled: bool):
    """Enable or disable color output"""
    global _use_colors
    _use_colors = enabled


def bold(text):
    return f"\033[1m{text}\033[0m" if _use_colors else text


def cyan(text):
    return f"\033[36m{text}\033[0m" if _use_colors else text


def green(text):
    return f"\033[32m{text}\033[0m" if _use_colors else text


def magenta(text):
    return f"\033[35m{text}\033[0m" if _use_colors else text


def red(text):
    return f"\033[31m{text}\033[0m" if _use_colors else text


def yellow(text):
    return f"\033[33m{text}\033[0m" if _use_colors else text


# Configuration
BUILD_DIR = Path("build-debug")
COVERAGE_FILE = "coverage.info"


def main():
    """Main entry point"""
    # Disable colors if not in a terminal
    if not sys.stdout.isatty():
        use_colors(False)

    try:
        print(bold(green("🚀 Processing C++ coverage data...")))

        # Create output directory
        BUILD_DIR.mkdir(parents=True, exist_ok=True)
        coverage_path = BUILD_DIR / COVERAGE_FILE

        # Find all coverage.dat files
        print(cyan("🔍 Looking for Bazel coverage files..."))
        coverage_files = []

        for search_dir in ["bazel-testlogs", "bazel-out"]:
            if Path(search_dir).exists():
                coverage_files.extend(list(Path(search_dir).rglob("coverage.dat")))

        if not coverage_files:
            print(red("✗ No coverage.dat files found!"))
            print(yellow("  Make sure 'bazel coverage' was run first"))
            return 1

        print(green(f"✓ Found {len(coverage_files)} coverage.dat files"))

        # Combine all non-empty coverage files
        print(cyan("📊 Combining coverage data..."))
        combined_lines = []
        files_processed = 0

        for cov_file in coverage_files:
            try:
                if cov_file.stat().st_size > 0:
                    print(f"  Adding: {cov_file.relative_to('.')}")
                    with open(cov_file, "r") as f:
                        content = f.read()
                        if content.strip():  # Only add non-empty content
                            combined_lines.append(content)
                            files_processed += 1
            except (FileNotFoundError, PermissionError) as e:
                print(f"  Warning: Could not process {cov_file}: {e}")
        if not combined_lines:
            print(red("✗ All coverage files are empty!"))
            return 1

        # Write combined coverage
        with open(coverage_path, "w") as f:
            f.write("\n".join(combined_lines))

        # Success!
        print(bold(green(f"\n✅ Successfully combined {files_processed} coverage files!")))
        print(f"📄 Output file: {cyan(str(coverage_path))}")
        print(f"📏 File size: {magenta(f'{coverage_path.stat().st_size:,} bytes')}")

        # Quick validation - check for LCOV format markers
        with open(coverage_path, "r") as f:
            first_line = f.readline().strip()
            if first_line.startswith("TN:") or first_line.startswith("SF:"):
                print(green("✓ Valid LCOV format detected"))
            else:
                print(yellow("⚠️  Warning: File may not be in LCOV format"))

        print(yellow("\n📤 Upload to Codecov with:"))
        print(f"  codecov -f {coverage_path}")
        print("  # or")
        print(f"  bash <(curl -s https://codecov.io/bash) -f {coverage_path}")

        return 0

    except Exception as e:
        print(red(f"❌ Error: {e}"))
        return 1


if __name__ == "__main__":
    sys.exit(main())
