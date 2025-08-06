#!/usr/bin/env python3
import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


def find_cassette_file(run_dir: str) -> str | None:
    """Find the VCR cassette file in the run directory."""
    for root, _, files in os.walk(run_dir):
        for file in files:
            if file.startswith("http_interactions_") and file.endswith(".yaml"):
                return os.path.join(root, file)
    return None


def parse_env_vars_from_log(log_file: str) -> dict[str, str]:
    """Parse environment variables from the log file."""
    env_vars = {}
    # Regex to capture KEY and VALUE from "Environment variable 'KEY' = 'VALUE'"
    # It handles single quotes around the value.
    env_pattern = re.compile(r"Environment variable '([^']*)' = ('.*?')$")

    if not os.path.exists(log_file):
        print(f"Log file not found: {log_file}", file=sys.stderr)
        return {}

    with open(log_file, "r") as f:
        for line in f:
            match = env_pattern.search(line)
            if match:
                key, value = match.groups()
                # Strip single quotes from the captured value
                env_vars[key] = value.strip("'")
    return env_vars


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Replay a VCR cassette from a GitHub Action run to test pr_gh_to_asana.py."
    )
    parser.add_argument("run_directory", help="Directory containing logs and artifacts from a workflow run.")
    args = parser.parse_args()

    run_dir = Path(args.run_directory)
    if not run_dir.is_dir():
        print(f"Error: Directory not found: {run_dir}", file=sys.stderr)
        sys.exit(1)

    # Infer run ID from directory name
    if not run_dir.name.startswith("run_"):
        print(f"Error: Directory name '{run_dir.name}' does not match expected format 'run_<ID>'.", file=sys.stderr)
        sys.exit(1)
    github_run_id = run_dir.name.removeprefix("run_")
    print(f"Inferred GITHUB_RUN_ID={github_run_id} from directory name.")

    # 1. Parse environment variables from log file
    log_file = run_dir / "run_log.txt"
    print(f"Parsing environment variables from {log_file}...")
    env_vars = parse_env_vars_from_log(str(log_file))
    if not env_vars:
        print(f"Could not parse any environment variables from {log_file}. Aborting.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(env_vars)} environment variables.")

    # 2. Find and prepare the cassette
    original_cassette_path = find_cassette_file(str(run_dir))
    if not original_cassette_path:
        print(f"Error: No VCR cassette file (http_interactions_*.yaml) found in {run_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"Found cassette file: {original_cassette_path}")

    # The script under test will look for this specific cassette name in its CWD
    expected_cassette_name = f"http_interactions_{github_run_id}.yaml"

    # We copy it to the current directory for the test run
    shutil.copy(original_cassette_path, expected_cassette_name)
    print(f"Copied cassette to ./{expected_cassette_name} for replay.")

    # 3. Prepare to run the script under test
    script_dir = Path(__file__).parent.resolve()
    script_to_test = script_dir.parent / "pr_gh_to_asana.py"

    if not script_to_test.exists():
        print(f"Error: Script to test not found at {script_to_test}", file=sys.stderr)
        sys.exit(1)

    # 4. Run the script with VCR in replay mode
    test_env = os.environ.copy()
    test_env.update(env_vars)
    test_env["VCR_RECORD_MODE"] = "none"  # Fail on new HTTP requests
    test_env["PYTHONPATH"] = str(script_to_test.parent)  # Add script dir to path to find modules
    test_env["VCR_CASSETTE_PATH"] = str(Path(expected_cassette_name).resolve())

    print("\n" + "=" * 80)
    print(f"Running {script_to_test.name} in replay mode...")
    print("=" * 80)

    try:
        result = subprocess.run(
            [sys.executable, str(script_to_test)], env=test_env, check=True, capture_output=True, text=True
        )
        print("--- STDOUT ---")
        print(result.stdout)
        print("--- STDERR ---")
        print(result.stderr)
        print("\n✅ PASSED: Script executed successfully with the recorded HTTP interactions.")

    except subprocess.CalledProcessError as e:
        print("\n❌ FAILED: Script failed during replay.", file=sys.stderr)
        print(
            "This could mean there was an unexpected HTTP request or a code change that broke the replay.",
            file=sys.stderr,
        )

        print("\n--- DEBUGGING INFORMATION ---", file=sys.stderr)
        # Print the PR number that was used in the failing run
        pr_number = test_env.get("INPUT_PR_NUMBER", "Not Found")
        print(f"  PR Number used in test: {pr_number}", file=sys.stderr)

        # Print the requests recorded in the cassette
        try:
            with open(expected_cassette_name, "r") as f:
                cassette_data = yaml.safe_load(f)
                print("  Requests recorded in the cassette:", file=sys.stderr)
                if cassette_data and "interactions" in cassette_data:
                    for i, interaction in enumerate(cassette_data["interactions"]):
                        uri = interaction.get("request", {}).get("uri", "Unknown URI")
                        method = interaction.get("request", {}).get("method", "Unknown Method")
                        print(f"    {i + 1}. {method} {uri}", file=sys.stderr)
                else:
                    print("    No interactions found in cassette.", file=sys.stderr)
        except Exception as yaml_e:
            print(f"  Could not read or parse cassette file for debugging: {yaml_e}", file=sys.stderr)

        print("\n--- ORIGINAL FAILURE ---", file=sys.stderr)
        print("--- STDOUT ---", file=sys.stderr)
        print(e.stdout, file=sys.stderr)
        print("--- STDERR ---", file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        sys.exit(1)

    finally:
        # 5. Clean up
        if os.path.exists(expected_cassette_name):
            os.remove(expected_cassette_name)
            print(f"\nCleaned up cassette file: {expected_cassette_name}")


if __name__ == "__main__":
    main()
