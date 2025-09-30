"""CLI for Metta development tasks."""

import importlib.util
import os
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Optional

import questionary
import typer
from rich.console import Console

# TODO: move the unit test group discovery to a separate function
PYTHON_TEST_FOLDERS = [
    # Core package tests
    "tests",
    # Individual agent tests
    "agent/tests",
    # Tool-specific test folders
    "app_backend/tests",
    "mcp_servers/wandb_dashboard/tests",
    # Environment-specific tests
    "common/tests",
    # Only test packages/mettagrid/tests for Python, not C++
    "packages/mettagrid/tests",
    # Test special packages that need separate environments
    # NOTE: Skipping codebot as it has dependency conflicts
    # "codebot/tests",
]

# Mapping of package names to git tag prefixes
PACKAGE_TAG_PREFIXES = {
    "mettagrid": "mettagrid-v",
}

console = Console()


def info(msg: str):
    console.print(f"[cyan]{msg}[/cyan]")


def success(msg: str):
    console.print(f"[green]{msg}[/green]")


def warning(msg: str):
    console.print(f"[yellow]{msg}[/yellow]")


def error(msg: str):
    console.print(f"[red]{msg}[/red]")


class MettaCLI:
    """CLI for Metta development tasks."""

    def __init__(self):
        # Store repo root once
        self.repo_root = self._find_repo_root()

    def _find_repo_root(self) -> Path:
        """Find the repository root by looking for .git directory or pyproject.toml."""
        current_dir = Path.cwd()
        for parent in [current_dir, *current_dir.parents]:
            if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
                return parent
        # Fallback to current directory if not found
        return current_dir


# Global instance
cli = MettaCLI()

# Create the main Typer app
app = typer.Typer(help="Metta development CLI")


@app.command(name="doctor", help="Check for common setup issues")
def cmd_doctor():
    info("Running Metta doctor checks...")
    has_errors = False

    # Check Python version
    version_info = sys.version_info
    if version_info < (3, 11):
        error(f"Python version {version_info.major}.{version_info.minor} is too old. Please use Python 3.11+")
        has_errors = True
    else:
        success(f"Python version: {version_info.major}.{version_info.minor}.{version_info.micro}")

    # Check if uv is installed
    if shutil.which("uv"):
        success("uv is installed")
    else:
        error("uv is not installed. Install from https://github.com/astral-sh/uv")
        has_errors = True

    # Check if .venv exists
    venv_path = cli.repo_root / ".venv"
    if venv_path.exists():
        success(".venv directory exists")
    else:
        warning(".venv directory not found. Run `uv sync` to create it.")

    # Check if git is installed
    if shutil.which("git"):
        success("git is installed")
    else:
        error("git is not installed")
        has_errors = True

    # Check if make is installed
    if shutil.which("make"):
        success("make is installed")
    else:
        warning("make is not installed. Some C++ build commands may not work.")

    # Check if mettagrid can be built
    mettagrid_dir = cli.repo_root / "packages" / "mettagrid"
    if (mettagrid_dir / "BUILD.bazel").exists():
        success("mettagrid BUILD.bazel found")
    else:
        error("mettagrid BUILD.bazel not found")
        has_errors = True

    # Print summary
    print()
    if has_errors:
        error("Some checks failed. Please fix the issues above.")
        raise typer.Exit(1)
    else:
        success("All checks passed!")


@app.command(name="shell", help="Set up the Python environment and launch a shell")
def cmd_shell():
    """Set up the Python environment and launch a shell."""
    shell_module_path = cli.repo_root / "metta" / "setup" / "shell.py"
    spec = importlib.util.spec_from_file_location("shell", shell_module_path)
    assert spec is not None, f"Could not load spec from {shell_module_path}"
    shell = importlib.util.module_from_spec(spec)
    assert spec.loader is not None, f"Spec loader is None for {shell_module_path}"
    spec.loader.exec_module(shell)
    sys.exit(shell.cmd_shell())


@app.command(name="setup", help="Set up Metta development environment")
def cmd_setup():
    """Set up the Metta development environment."""
    # Create all required directories
    info("Setting up Metta environment...")

    # Run install.sh if it exists
    install_sh = cli.repo_root / "install.sh"
    if install_sh.exists():
        info("Running install.sh...")
        try:
            subprocess.run([str(install_sh)], cwd=cli.repo_root, check=True)
            success("Setup complete!")
        except subprocess.CalledProcessError as e:
            error("Setup failed!")
            raise typer.Exit(e.returncode) from e
    else:
        warning("install.sh not found. Please run setup manually.")


@app.command(name="test", help="Run Python unit tests")
def cmd_test(
    folder: Annotated[
        Optional[str],
        typer.Option(help="Specific test folder or file to run (e.g. 'tests', 'tests/rl', 'tests/test_foo.py')"),
    ] = None,
    keyword: Annotated[Optional[str], typer.Option("--keyword", "-k", help="Run tests matching the keyword")] = None,
    markers: Annotated[Optional[str], typer.Option("--markers", "-m", help="Run tests matching the markers")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Verbose output")] = False,
    capture: Annotated[
        str, typer.Option("--capture", help="Capture output: 'yes' (default), 'no', or 'sys'")
    ] = "yes",
    debug: Annotated[bool, typer.Option("--debug", help="Enable pytest debug mode (-vv --log-cli-level=DEBUG)")] = False,
    pdb: Annotated[bool, typer.Option("--pdb", help="Enter debugger on test failure")] = False,
    parallel: Annotated[
        Optional[str],
        typer.Option("--parallel", "-n", help="Number of parallel workers (default: auto, use -n 1 to disable)"),
    ] = "auto",
):
    """Run Python unit tests using pytest."""
    folders_to_test = [folder] if folder else PYTHON_TEST_FOLDERS

    test_cmd = [
        "uv",
        "run",
        "pytest",
        *folders_to_test,
        "--benchmark-disable",
        "-n",
        parallel,
    ]

    if keyword:
        test_cmd.extend(["-k", keyword])

    if markers:
        test_cmd.extend(["-m", markers])

    if verbose:
        test_cmd.append("-v")

    if debug:
        test_cmd.extend(["-vv", "--log-cli-level=DEBUG"])

    if pdb:
        test_cmd.append("--pdb")

    if capture != "yes":
        test_cmd.append(f"--capture={capture}")

    info(f"Running: {' '.join(test_cmd)}")

    try:
        subprocess.run(test_cmd, cwd=cli.repo_root, check=True)
        success("Tests passed!")
    except subprocess.CalledProcessError as e:
        error("Tests failed!")
        raise typer.Exit(e.returncode) from e


@app.command(
    name="test-benchmark", help="Run Python benchmarks (tests marked with @pytest.mark.benchmark_compare_columns)"
)
def cmd_test_benchmark(
    folder: Annotated[
        Optional[str],
        typer.Option(help="Specific test folder or file to run (e.g. 'tests', 'tests/rl', 'tests/test_foo.py')"),
    ] = None,
    keyword: Annotated[Optional[str], typer.Option("--keyword", "-k", help="Run tests matching the keyword")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Verbose output")] = False,
    save: Annotated[
        Optional[str], typer.Option("--save", help="Save benchmark results with this name for later comparison")
    ] = None,
    compare: Annotated[
        Optional[str], typer.Option("--compare", help="Compare against previously saved benchmark with this name")
    ] = None,
):
    """Run Python benchmarks using pytest-benchmark."""
    folders_to_test = [folder] if folder else PYTHON_TEST_FOLDERS

    test_cmd = [
        "uv",
        "run",
        "pytest",
        *folders_to_test,
        "--benchmark-only",
        "--benchmark-compare-columns=min,max,mean,stddev",
        "-n",
        "0",  # Benchmarks must run serially
    ]

    if keyword:
        test_cmd.extend(["-k", keyword])

    if verbose:
        test_cmd.append("-v")

    if save:
        test_cmd.append(f"--benchmark-save={save}")

    if compare:
        test_cmd.append(f"--benchmark-compare={compare}")

    info(f"Running: {' '.join(test_cmd)}")

    try:
        subprocess.run(test_cmd, cwd=cli.repo_root, check=True)
        success("Benchmarks complete!")
    except subprocess.CalledProcessError as e:
        error("Benchmarks failed!")
        raise typer.Exit(e.returncode) from e


@app.command(name="test-package", help="Run tests for a specific package")
def cmd_test_package(
    package: Annotated[str, typer.Argument(help="Package name (e.g., 'mettagrid', 'agent', 'common')")],
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Verbose output")] = False,
):
    """Run tests for a specific package."""
    package_map = {
        "mettagrid": "packages/mettagrid/tests",
        "agent": "agent/tests",
        "common": "common/tests",
        "app_backend": "app_backend/tests",
    }

    if package not in package_map:
        error(f"Unknown package: {package}. Available packages: {', '.join(package_map.keys())}")
        raise typer.Exit(1)

    test_path = cli.repo_root / package_map[package]
    if not test_path.exists():
        error(f"Test directory not found: {test_path}")
        raise typer.Exit(1)

    test_cmd = ["uv", "run", "pytest", str(test_path), "--benchmark-disable", "-n", "auto"]

    if verbose:
        test_cmd.append("-v")

    info(f"Running: {' '.join(test_cmd)}")

    try:
        subprocess.run(test_cmd, cwd=cli.repo_root, check=True)
        success(f"{package} tests passed!")
    except subprocess.CalledProcessError as e:
        error(f"{package} tests failed!")
        raise typer.Exit(e.returncode) from e


@app.command(name="clean", help="Clean build artifacts and caches")
def cmd_clean(verbose: Annotated[bool, typer.Option("--verbose", help="Verbose output")] = False):
    build_dir = cli.repo_root / "build"
    if build_dir.exists():
        info("  Removing root build directory...")
        shutil.rmtree(build_dir)

    mettagrid_dir = cli.repo_root / "packages" / "mettagrid"
    for build_name in ["build-debug", "build-release"]:
        build_path = mettagrid_dir / build_name
        if build_path.exists():
            info(f"  Removing packages/mettagrid/{build_name}...")
            shutil.rmtree(build_path)

    cleanup_script = cli.repo_root / "devops" / "tools" / "cleanup_repo.py"
    if cleanup_script.exists():
        cmd = [str(cleanup_script)]
        if verbose:
            cmd.append("--verbose")
        try:
            subprocess.run(cmd, cwd=str(cli.repo_root), check=True)
        except subprocess.CalledProcessError as e:
            warning(f"  Cleanup script failed: {e}")


@app.command(name="publish", help="Create and push a release tag for a package")
def cmd_publish(
    package: Annotated[str, typer.Argument(help="Package to publish (currently only 'mettagrid')")],
    version_override: Annotated[
        Optional[str],
        typer.Option("--version", "-v", help="Explicit version to tag (digits separated by dots)"),
    ] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Preview actions without tagging")] = False,
    remote: Annotated[str, typer.Option("--remote", help="Git remote to push the tag to")] = "origin",
    force: Annotated[bool, typer.Option("--force", help="Bypass branch and clean checks")] = False,
):
    package = package.lower()
    if package not in PACKAGE_TAG_PREFIXES:
        error(f"Unsupported package '{package}'. Supported packages: {', '.join(sorted(PACKAGE_TAG_PREFIXES))}.")
        raise typer.Exit(1)

    prefix = PACKAGE_TAG_PREFIXES[package]

    try:
        info(f"Fetching tags from {remote}...")
        _run_git_command(["fetch", remote, "--tags"], capture_output=False)
    except subprocess.CalledProcessError as exc:
        error(f"Failed to fetch tags from {remote}: {exc}")
        raise typer.Exit(exc.returncode) from exc

    try:
        # Determine version string
        if version_override:
            # Validate format: must be digits separated by dots
            if not re.fullmatch(r"\d+(\.\d+)*", version_override):
                error(
                    f"Invalid version format: {version_override}. "
                    "Version must be digits separated by dots (e.g., 1.0.0)."
                )
                raise typer.Exit(1)
            version_str = version_override
        else:
            # Automatically find the next version
            version_str = _find_next_version(prefix)

        full_tag = f"{prefix}{version_str}"

        # Validate branch and clean state (unless --force)
        if not force:
            _validate_branch_and_clean_state(remote)

        # Preview summary
        info(f"\nRelease Summary:")
        info(f"  Package: {package}")
        info(f"  Tag: {full_tag}")
        info(f"  Remote: {remote}")
        info(f"  Dry run: {dry_run}")

        # Confirm with user (skip in dry run)
        if not dry_run:
            confirm = questionary.confirm("Proceed with tagging?", default=False).ask()
            if not confirm:
                warning("Tag creation cancelled.")
                raise typer.Exit(0)

        # Create and push the tag
        if dry_run:
            info(f"\nDry run: Would create tag '{full_tag}' and push to '{remote}'")
        else:
            info(f"\nCreating tag '{full_tag}'...")
            _run_git_command(["tag", full_tag])

            info(f"Pushing tag '{full_tag}' to {remote}...")
            _run_git_command(["push", remote, full_tag], capture_output=False)

            success(f"\nTag '{full_tag}' successfully created and pushed to '{remote}'.")

    except subprocess.CalledProcessError as exc:
        error(f"Command failed: {exc}")
        raise typer.Exit(exc.returncode) from exc
    except Exception as exc:
        error(f"Unexpected error: {exc}")
        raise typer.Exit(1) from exc


def _run_git_command(args: list[str], capture_output: bool = True) -> subprocess.CompletedProcess:
    """Helper to run git commands."""
    return subprocess.run(["git", *args], cwd=cli.repo_root, check=True, capture_output=capture_output, text=True)


def _find_next_version(prefix: str) -> str:
    """Find the next version by incrementing the patch number of the latest tag with the given prefix."""
    result = _run_git_command(["tag", "--list", f"{prefix}*", "--sort=-version:refname"])
    tags = result.stdout.strip().split("\n") if result.stdout.strip() else []

    if not tags:
        info(f"No existing tags with prefix '{prefix}' found. Starting at 0.0.1")
        return "0.0.1"

    latest_tag = tags[0]
    version_str = latest_tag.removeprefix(prefix)

    # Parse version
    try:
        parts = [int(p) for p in version_str.split(".")]
    except ValueError:
        error(f"Could not parse version from tag '{latest_tag}'. Expected format: {prefix}<major>.<minor>.<patch>")
        raise typer.Exit(1)

    # Increment patch version
    if len(parts) < 3:
        parts.extend([0] * (3 - len(parts)))
    parts[2] += 1

    next_version = ".".join(map(str, parts))
    info(f"Latest tag: {latest_tag} -> Next version: {next_version}")
    return next_version


def _validate_branch_and_clean_state(remote: str):
    """Ensure we're on main, synced with remote, and have no uncommitted changes."""
    # Check current branch
    result = _run_git_command(["rev-parse", "--abbrev-ref", "HEAD"])
    current_branch = result.stdout.strip()

    if current_branch != "main":
        error(f"Not on main branch (currently on '{current_branch}'). Use --force to bypass.")
        raise typer.Exit(1)

    # Check if working directory is clean
    result = _run_git_command(["status", "--porcelain"])
    if result.stdout.strip():
        error("Working directory is not clean. Commit or stash changes. Use --force to bypass.")
        raise typer.Exit(1)

    # Check if local main is in sync with remote
    _run_git_command(["fetch", remote])
    local_commit = _run_git_command(["rev-parse", "main"]).stdout.strip()
    remote_commit = _run_git_command(["rev-parse", f"{remote}/main"]).stdout.strip()

    if local_commit != remote_commit:
        error(f"Local 'main' is not in sync with '{remote}/main'. Pull or push changes. Use --force to bypass.")
        raise typer.Exit(1)

    success("Branch and state validation passed.")


@app.command(name="lint", help="Run linting and formatting")
def cmd_lint(
    files: Annotated[Optional[list[str]], typer.Argument()] = None,
    fix: Annotated[bool, typer.Option("--fix", help="Apply fixes automatically")] = False,
    staged: Annotated[bool, typer.Option("--staged", help="Only lint staged files")] = False,
):
    py_files = []
    cpp_files = []

    # Determine which files to lint
    if files:
        # Filter to Python and C++ files
        py_files = [f for f in files if f.endswith(".py")]
        cpp_files = [f for f in files if f.endswith((".cpp", ".hpp", ".h", ".cc", ".cxx"))]
    elif staged:
        # Discover staged files
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
            cwd=cli.repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        all_files = [f for f in result.stdout.strip().split("\n") if f]
        py_files = [f for f in all_files if f.endswith(".py")]
        cpp_files = [f for f in all_files if f.endswith((".cpp", ".hpp", ".h", ".cc", ".cxx"))]

    if not py_files and not cpp_files:
        if files is not None or staged:
            info("No Python or C++ files to lint")
            return

    # Run Python linting with ruff
    if py_files or not (files is not None or staged):
        check_cmd = ["uv", "run", "--active", "ruff", "check"]
        format_cmd = ["uv", "run", "--active", "ruff", "format"]

        if fix:
            check_cmd.append("--fix")
        else:
            format_cmd.append("--check")

        if py_files:
            check_cmd.extend(py_files)
            format_cmd.extend(py_files)

        # Run commands
        for cmd in [format_cmd, check_cmd]:
            try:
                info(f"Running: {' '.join(cmd)}")
                subprocess.run(cmd, cwd=cli.repo_root, check=True)
            except subprocess.CalledProcessError as e:
                raise typer.Exit(e.returncode) from e

    # Run C++ linting with cpplint.sh
    if cpp_files or not (files is not None or staged):
        cpplint_script = cli.repo_root / "packages/mettagrid/tests/cpplint.sh"
        try:
            if cpp_files:
                # For specific files, pass them to cpplint.sh
                info(f"Running cpplint.sh on {len(cpp_files)} C++ files...")
                subprocess.run([str(cpplint_script)] + cpp_files, cwd=cli.repo_root, check=True)
            else:
                # For full mode, run cpplint.sh without arguments (finds all files)
                info("Running C++ linting with cpplint.sh...")
                subprocess.run([str(cpplint_script)], cwd=cli.repo_root, check=True)
        except subprocess.CalledProcessError as e:
            raise typer.Exit(e.returncode) from e


@app.command(name="ci", help="Run all Python unit tests and all Mettagrid C++ tests")
def cmd_ci():
    info("Running Python tests...")
    python_test_cmd = [
        "uv",
        "run",
        "pytest",
        *PYTHON_TEST_FOLDERS,
        "--benchmark-disable",
        "-n",
        "auto",
    ]

    try:
        subprocess.run(python_test_cmd, cwd=cli.repo_root, check=True)
        success("Python tests passed!")
    except subprocess.CalledProcessError as e:
        error("Python tests failed!")
        raise typer.Exit(e.returncode) from e

    info("\nBuilding and running C++ tests...")
    mettagrid_dir = cli.repo_root / "packages" / "mettagrid"

    try:
        subprocess.run(["make", "test"], cwd=mettagrid_dir, check=True)
        success("C++ tests passed!")
        # Note: Benchmarks are not run in CI as they're for performance testing, not correctness
        # To run benchmarks manually, use: cd packages/mettagrid && make benchmark
    except subprocess.CalledProcessError as e:
        error("C++ tests failed!")
        raise typer.Exit(e.returncode) from e

    success("\nAll CI tests passed!")


@app.command(name="benchmark", help="Run C++ and Python benchmarks for mettagrid")
def cmd_benchmark():
    """Run performance benchmarks for the mettagrid package."""
    mettagrid_dir = cli.repo_root / "packages" / "mettagrid"

    info("Running mettagrid benchmarks...")
    info("Note: This may fail if Python environment is not properly configured.")
    info("If it fails, try running directly: cd packages/mettagrid && make benchmark")

    try:
        subprocess.run(["make", "benchmark"], cwd=mettagrid_dir, check=True)
        success("Benchmarks completed!")
    except subprocess.CalledProcessError as e:
        error("Benchmark execution failed!")
        raise typer.Exit(e.returncode) from e


@app.command(name="install-hooks", help="Install git pre-commit hooks")
def cmd_install_hooks():
    """Install git pre-commit hooks for the repository."""
    from metta.setup.components.githooks import install_hooks

    install_hooks()


@app.command(name="uninstall-hooks", help="Uninstall git pre-commit hooks")
def cmd_uninstall_hooks():
    """Uninstall git pre-commit hooks for the repository."""
    from metta.setup.components.githooks import uninstall_hooks

    uninstall_hooks()


@app.command(name="helm", help="Manage Helm installations (requires kubectl and helm)")
def cmd_helm(
    action: Annotated[str, typer.Argument(help="Action to perform: 'install' or 'uninstall'")],
    chart: Annotated[Optional[str], typer.Argument(help="Specific chart name (optional)")] = None,
):
    """Manage Helm chart installations."""
    from metta.setup.components.helm import helm_install, helm_uninstall

    if action == "install":
        helm_install(chart)
    elif action == "uninstall":
        helm_uninstall(chart)
    else:
        error(f"Unknown action: {action}. Use 'install' or 'uninstall'.")
        raise typer.Exit(1)


@app.command(name="install-datadog", help="Install Datadog agent (requires kubectl and helm)")
def cmd_install_datadog():
    """Install Datadog monitoring agent."""
    from metta.setup.components.datadog_agent import install_datadog

    install_datadog()


@app.command(name="build-mettagrid", help="Build the mettagrid C++ package")
def cmd_build_mettagrid(
    mode: Annotated[str, typer.Option("--mode", help="Build mode: 'debug' or 'release'")] = "release",
):
    """Build the mettagrid C++ package using Bazel."""
    mettagrid_dir = cli.repo_root / "packages" / "mettagrid"

    if mode == "debug":
        build_cmd = ["make", "build-debug"]
    elif mode == "release":
        build_cmd = ["make", "build"]
    else:
        error(f"Unknown build mode: {mode}. Use 'debug' or 'release'.")
        raise typer.Exit(1)

    info(f"Building mettagrid in {mode} mode...")
    try:
        subprocess.run(build_cmd, cwd=mettagrid_dir, check=True)
        success("Build completed!")
    except subprocess.CalledProcessError as e:
        error("Build failed!")
        raise typer.Exit(e.returncode) from e


@app.command(name="version", help="Print the Metta CLI version")
def cmd_version():
    """Print version information."""
    info("Metta CLI version: 1.0.0")


def main():
    """Main entry point for the CLI."""
    # Handle special case where we want to run without arguments
    if len(sys.argv) == 1:
        sys.argv.append("--help")
    app()


if __name__ == "__main__":
    main()
