#!/usr/bin/env -S uv run --active
# /// script
# requires-python = ">=3.11"
# dependencies = ["gcovr>=6.0"]
# ///
"""
generate_coverage.py - Generate C++ coverage report

A cross-platform coverage generation script that supports both GCC/lcov and Clang/llvm-cov formats.
Prioritizes gcovr for simplified processing.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

# Add common package to path
script_path = Path(__file__).resolve()
common_path = script_path.parents[1] / "common"
sys.path.insert(0, str(common_path))

from metta.common.util.colorama import bold, cyan, green, magenta, red, use_colors, yellow  # noqa: E402

# Configuration
BUILD_DIR = Path("build-coverage")
COVERAGE_FILE = "coverage.info"


def run_command(cmd: List[str], cwd: Optional[Path] = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run a command and return the result"""
    print(yellow(f"→ {' '.join(cmd)}"))
    try:
        result = subprocess.run(cmd, cwd=cwd, check=check, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        if result.stderr and result.returncode != 0:
            print(red(result.stderr), file=sys.stderr)
        return result
    except subprocess.CalledProcessError as e:
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(red(e.stderr), file=sys.stderr)
        if check:
            raise
        # Return a CompletedProcess object instead of the exception
        return subprocess.CompletedProcess(args=e.cmd, returncode=e.returncode, stdout=e.stdout, stderr=e.stderr)


def check_command_exists(cmd: str) -> bool:
    """Check if a command exists in PATH"""
    return shutil.which(cmd) is not None


def ensure_gcovr() -> bool:
    """Check if gcovr is available, suggest installation if not"""
    if check_command_exists("gcovr"):
        return True

    print(red("Error: gcovr is not installed"))
    print(yellow("Install with: pip install gcovr"))
    print(yellow("Or run this script with: uv run generate_coverage.py"))
    return False


def setup_and_build() -> bool:
    """Setup build directory and build project"""
    print(bold(green("🚀 Starting C++ coverage generation...")))

    # Clean and create build directory
    print(cyan("📁 Setting up coverage build..."))
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)

    # Run cmake with coverage preset
    result = run_command(["cmake", "--preset", "coverage"], check=False)
    if result.returncode != 0:
        print(red("✗ Failed to configure project"))
        return False

    # Build the project
    print(cyan("🔨 Building project with coverage..."))
    cpu_count = os.cpu_count() or 4
    result = run_command(["cmake", "--build", str(BUILD_DIR), "-j", str(cpu_count)], check=False)
    if result.returncode != 0:
        print(red("✗ Failed to build project"))
        return False

    # Run tests (don't fail if tests fail - we still want coverage)
    print(cyan("🧪 Running tests..."))
    run_command(["ctest", "--output-on-failure"], cwd=BUILD_DIR, check=False)
    return True


def process_coverage() -> bool:
    """Process coverage data using gcovr"""
    coverage_path = BUILD_DIR / COVERAGE_FILE

    # Detect which coverage format we have
    print(cyan("🔍 Detecting coverage format..."))

    gcda_files = list(BUILD_DIR.rglob("*.gcda"))
    profraw_files = list(BUILD_DIR.rglob("*.profraw"))

    if gcda_files:
        print(green(f"✓ Found GCC coverage data ({len(gcda_files)} .gcda files)"))
    elif profraw_files:
        print(green(f"✓ Found LLVM coverage data ({len(profraw_files)} .profraw files)"))
    else:
        print(red("✗ No coverage data found!"))
        print(yellow("  Make sure your build was configured with coverage enabled"))
        return False

    # Use gcovr for both GCC and Clang coverage
    print(cyan("📊 Generating coverage report with gcovr..."))

    # Common exclude patterns
    excludes = [
        "--exclude",
        ".*test.*",
        "--exclude",
        ".*/usr/.*",
        "--exclude",
        ".*pybind11.*",
        "--exclude",
        ".*/site-packages/.*",
        "--exclude",
        ".*/benchmarks/.*",
        "--exclude",
        ".*googletest.*",
        "--exclude",
        ".*googlebenchmark.*",
    ]

    # Generate LCOV format for codecov
    cmd = ["gcovr", "--root", ".", "--lcov", str(coverage_path), "--print-summary"] + excludes
    result = run_command(cmd, check=False)

    if result.returncode != 0:
        print(red("✗ Failed to generate coverage report"))
        return False

    # Also generate a nice HTML report
    html_dir = BUILD_DIR / "coverage-html"
    html_dir.mkdir(parents=True, exist_ok=True)
    print(cyan("🌐 Generating HTML coverage report..."))

    cmd = [
        "gcovr",
        "--root",
        ".",
        "--html-details",
        str(html_dir / "index.html"),
        "--html-title",
        "Code Coverage Report",
    ] + excludes
    run_command(cmd, check=False)

    return True


def verify_and_display_results() -> bool:
    """Verify coverage file and display results"""
    coverage_path = BUILD_DIR / COVERAGE_FILE

    if not coverage_path.exists():
        print(red("✗ Coverage file was not created"))
        return False

    if coverage_path.stat().st_size == 0:
        print(red("✗ Coverage file is empty"))
        return False

    # Success!
    print(bold(green("✅ Coverage generation complete!")))
    print(f"📄 Coverage report: {cyan(str(coverage_path))}")
    print(f"📏 File size: {magenta(f'{coverage_path.stat().st_size:,} bytes')}")

    html_report = BUILD_DIR / "coverage-html" / "index.html"
    if html_report.exists():
        # Simple file URL for Unix-like systems
        html_url = f"file://{html_report.resolve()}"
        print(f"🌐 HTML report: {cyan(html_url)}")

    print(yellow("\n📤 Upload to Codecov with:"))
    print(f"  codecov -f {coverage_path}")
    print("  # or")
    print(f"  bash <(curl -s https://codecov.io/bash) -f {coverage_path}")

    return True


def main():
    """Main entry point"""
    # Disable colors if not in a terminal
    if not sys.stdout.isatty():
        use_colors(False)

    try:
        # Check for gcovr
        if not shutil.which("gcovr"):
            print(red("Error: gcovr is not installed"))
            print(yellow("Install with: pip install gcovr"))
            print(yellow("Or run this script with: uv run generate_coverage.py"))
            return 1

        # Build and run tests
        if not setup_and_build():
            return 1

        # Process coverage
        if not process_coverage():
            return 1

        # Verify and display results
        if not verify_and_display_results():
            return 1

        return 0

    except KeyboardInterrupt:
        print(red("\n⚠️  Interrupted by user"))
        return 1
    except Exception as e:
        print(red(f"❌ Error: {e}"))
        return 1


if __name__ == "__main__":
    sys.exit(main())
