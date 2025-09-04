#!/usr/bin/env -S uv run
"""
generate_coverage.py - Generate C++ coverage report

A cross-platform coverage generation script that supports both GCC/lcov and Clang/llvm-cov formats.
Prioritizes gcovr for simplified processing.
"""

import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from metta.mettagrid.util.text_styles import bold, cyan, green, magenta, red, use_colors, yellow

# Configuration
BUILD_DIR = Path("build-debug")
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

    # Clean Bazel cache for fresh coverage
    print(cyan("📁 Setting up coverage build..."))
    run_command(["bazel", "clean"], check=False)

    # Build with coverage enabled
    print(cyan("🔨 Building project with coverage..."))
    result = run_command(["bazel", "build", "--config=dbg", "--collect_code_coverage", "//:mettagrid_c"], check=False)
    if result.returncode != 0:
        print(red("✗ Failed to build project"))
        return False

    # Run tests with bazel coverage (don't fail if tests fail - we still want coverage)
    print(cyan("🧪 Running tests with coverage..."))
    test_targets = ["//tests:tests_all"]
    run_command(["bazel", "coverage", "--config=dbg"] + test_targets, check=False)
    return True


def process_coverage() -> bool:
    """Process coverage data using gcovr or lcov"""
    coverage_path = BUILD_DIR / COVERAGE_FILE
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    # Detect which coverage format we have
    print(cyan("🔍 Detecting coverage format..."))

    # Look for Bazel's coverage.dat files
    bazel_coverage_files = []
    if Path("bazel-testlogs").exists():
        bazel_coverage_files = list(Path("bazel-testlogs").rglob("coverage.dat"))

    # Also check bazel-out directory
    if Path("bazel-out").exists():
        bazel_coverage_files.extend(list(Path("bazel-out").rglob("coverage.dat")))

    if bazel_coverage_files:
        print(green(f"✓ Found Bazel LCOV coverage data ({len(bazel_coverage_files)} coverage.dat files)"))

        # Combine all coverage.dat files into one
        print(cyan("📊 Combining coverage data..."))
        combined_coverage = []
        for cov_file in bazel_coverage_files:
            if cov_file.stat().st_size > 0:
                with open(cov_file, "r") as f:
                    combined_coverage.append(f.read())

        if not combined_coverage:
            print(red("✗ All coverage files are empty!"))
            return False

        # Write combined coverage
        with open(coverage_path, "w") as f:
            f.write("\n".join(combined_coverage))

        print(green(f"✓ Combined {len(bazel_coverage_files)} coverage files into {coverage_path}"))
        return True

    # Look for traditional gcda/profraw files
    gcda_files = []
    profraw_files = []

    # Check bazel-out directory
    if Path("bazel-out").exists():
        gcda_files.extend(list(Path("bazel-out").rglob("*.gcda")))
        profraw_files.extend(list(Path("bazel-out").rglob("*.profraw")))

    # Also check BUILD_DIR for compatibility
    gcda_files.extend(list(BUILD_DIR.rglob("*.gcda")))
    profraw_files.extend(list(BUILD_DIR.rglob("*.profraw")))

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
