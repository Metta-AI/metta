#!/usr/bin/env python3
"""Validate that local CI runner and GitHub Actions workflow run the same commands.

This script parses both metta/setup/tools/ci_runner.py and .github/workflows/checks.yml
to extract the metta CLI commands they execute, then compares them to ensure they stay in sync.

Usage:
    uv run python scripts/validate_ci_sync.py
"""

import ast
import re
import sys
from pathlib import Path

import yaml


def extract_commands_from_python(file_path: Path) -> dict[str, list[str]]:
    """Extract metta commands from ci_runner.py.

    Returns dict mapping check name to list of command strings.
    Example: {"lint": ["uv run metta lint"], "python_tests": ["uv run metta pytest --ci"]}
    """
    content = file_path.read_text()
    tree = ast.parse(content)

    commands = {}

    # Look for function definitions that run checks
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if node.name.startswith("_run_"):
                check_name = node.name.replace("_run_", "")
                check_commands = []

                # Look for list literals that look like commands
                for child in ast.walk(node):
                    if isinstance(child, ast.List):
                        # Try to extract the command from list literal
                        cmd_parts = []
                        for elt in child.elts:
                            if isinstance(elt, ast.Constant):
                                cmd_parts.append(str(elt.value))

                        # Filter for commands that start with "uv" or "metta"
                        if cmd_parts and (cmd_parts[0] in ("uv", "metta") or
                                         (len(cmd_parts) >= 3 and cmd_parts[0:2] == ["uv", "run"])):
                            # Normalize: remove verbose flags for comparison
                            normalized = [p for p in cmd_parts if p not in ("--verbose", "-v")]
                            check_commands.append(" ".join(normalized))

                if check_commands:
                    commands[check_name] = check_commands

    return commands


def extract_commands_from_yaml(file_path: Path) -> dict[str, list[str]]:
    """Extract metta commands from checks.yml.

    Returns dict mapping job name to list of command strings.
    Example: {"lint": ["uv run metta lint"], "python-tests": ["uv run metta pytest --ci"]}
    """
    content = file_path.read_text()
    data = yaml.safe_load(content)

    commands = {}

    if "jobs" not in data:
        return commands

    for job_name, job_data in data["jobs"].items():
        if not isinstance(job_data, dict) or "steps" not in job_data:
            continue

        job_commands = []

        for step in job_data["steps"]:
            if not isinstance(step, dict):
                continue

            # Look for "run" commands
            run_cmd = step.get("run", "")
            if not run_cmd:
                continue

            # Extract lines that contain "metta" commands
            for line in run_cmd.split("\n"):
                line = line.strip()

                # Skip comments and empty lines
                if not line or line.startswith("#"):
                    continue

                # Look for metta commands
                if "metta" in line and ("uv run" in line or line.startswith("metta")):
                    # Normalize: remove verbose flags and shell syntax for comparison
                    normalized = line
                    # Remove shell variable syntax
                    normalized = re.sub(r'\$\{.*?\}|\$\(.*?\)', '', normalized)
                    # Remove quotes
                    normalized = normalized.replace('"', '').replace("'", '')
                    # Remove verbose flags
                    normalized = re.sub(r'\s+--verbose\b', '', normalized)
                    # Remove conditional logic
                    normalized = re.sub(r'\s+\|\|.*$', '', normalized)
                    normalized = re.sub(r'\s+&&.*$', '', normalized)
                    # Clean up whitespace
                    normalized = " ".join(normalized.split())

                    if normalized:
                        job_commands.append(normalized)

        if job_commands:
            commands[job_name] = job_commands

    return commands


def normalize_command(cmd: str) -> str:
    """Normalize command for comparison by removing variable parts."""
    # Remove array syntax like "${ARGS[@]}"
    cmd = re.sub(r'\$\{[^}]+\}|\$\([^)]+\)', '', cmd)
    # Normalize whitespace
    cmd = " ".join(cmd.split())
    return cmd


def compare_commands(python_cmds: dict[str, list[str]], yaml_cmds: dict[str, list[str]]) -> bool:
    """Compare commands from both sources and report differences.

    Returns True if commands match, False otherwise.
    """
    print("=" * 80)
    print("CI Command Synchronization Check")
    print("=" * 80)
    print()

    # Build sets of unique commands from each source
    python_cmd_set = set()
    for check_name, cmds in python_cmds.items():
        print(f"Python CI Runner - {check_name}:")
        for cmd in cmds:
            normalized = normalize_command(cmd)
            python_cmd_set.add(normalized)
            print(f"  • {normalized}")
        print()

    yaml_cmd_set = set()
    for job_name, cmds in yaml_cmds.items():
        print(f"GitHub Workflow - {job_name}:")
        for cmd in cmds:
            normalized = normalize_command(cmd)
            yaml_cmd_set.add(normalized)
            print(f"  • {normalized}")
        print()

    # Compare the sets
    print("=" * 80)
    print("Comparison Results:")
    print("=" * 80)
    print()

    only_in_python = python_cmd_set - yaml_cmd_set
    only_in_yaml = yaml_cmd_set - python_cmd_set
    in_both = python_cmd_set & yaml_cmd_set

    if in_both:
        print("✓ Commands in both:")
        for cmd in sorted(in_both):
            print(f"  • {cmd}")
        print()

    all_match = not only_in_python and not only_in_yaml

    if only_in_python:
        print("⚠ Commands only in ci_runner.py:")
        for cmd in sorted(only_in_python):
            print(f"  • {cmd}")
        print()

    if only_in_yaml:
        print("⚠ Commands only in checks.yml:")
        for cmd in sorted(only_in_yaml):
            print(f"  • {cmd}")
        print()

    if all_match:
        print("✓ All commands match! CI runner and workflow are synchronized.")
        return True
    else:
        print("✗ Commands differ! Please update either ci_runner.py or checks.yml to match.")
        return False


def main() -> int:
    """Run the validation check."""
    repo_root = Path(__file__).parent.parent

    python_file = repo_root / "metta/setup/tools/ci_runner.py"
    yaml_file = repo_root / ".github/workflows/checks.yml"

    if not python_file.exists():
        print(f"Error: {python_file} not found", file=sys.stderr)
        return 1

    if not yaml_file.exists():
        print(f"Error: {yaml_file} not found", file=sys.stderr)
        return 1

    try:
        python_cmds = extract_commands_from_python(python_file)
        yaml_cmds = extract_commands_from_yaml(yaml_file)

        commands_match = compare_commands(python_cmds, yaml_cmds)

        return 0 if commands_match else 1

    except Exception as e:
        print(f"Error during validation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
