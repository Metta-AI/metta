#!/usr/bin/env -S uv run
"""Add policy to evaluation leaderboard and update dashboard."""

import argparse
import subprocess
import sys
from typing import List

from metta.common.util.constants import METTASCOPE_REPLAY_URL
from softmax.lib.utils import exists


def run_command(cmd: List[str], description: str) -> bool:
    """
    Run a command and return success status.

    Args:
        cmd: Command as list of strings
        description: Description for logging

    Returns:
        True if command succeeded, False otherwise
    """
    print(f"Executing: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"❌ {description} failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Add policy to evaluation leaderboard and update dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
%(prog)s --run b.username.test_run --policy-uri s3://softmax-public/checkpoints/b.username.test_run/b.username.test_run:v40.pt
""",
    )

    parser.add_argument("--run", required=True, help="Your run identifier (e.g., b.$USER.test_run)")
    parser.add_argument(
        "--policy-uri",
        required=True,
        help="Fully qualified S3 checkpoint URI (e.g., s3://bucket/path/run/checkpoints/run:v10.pt)",
    )

    # Capture additional arguments for Hydra
    args, additional_args = parser.parse_known_args()

    policy_uri = args.policy_uri
    print(f"Adding policy to eval leaderboard using policy URI: {policy_uri}")

    if additional_args:
        print(f"Additional arguments: {' '.join(additional_args)}")

    # Step 1: Verify policy exists on WANDB
    if not exists(policy_uri):
        print("\n💡 Policy checkpoint not found; double-check the URI or upload it before running this tool.")
        sys.exit(1)
    print("✅ Policy verification passed")

    # Step 2: Run the simulation
    print("\n🚀 Step 2: Running simulation...")
    sim_cmd = [
        "./tools/run.py",
        "experiments.recipes.navigation.eval",
        f"policy_uri={policy_uri}",
        "+eval_db_uri=wandb://stats/navigation_db",
        "+eval_db_uri=wandb://stats/navigation_db",
    ] + additional_args

    if not run_command(sim_cmd, "Simulation"):
        print("❌ Simulation failed. Exiting.")
        sys.exit(1)

    print("✅ Simulation completed successfully")

    # Step 3: Analyze and update dashboard
    print("\n📊 Step 3: Analyzing results and updating dashboard...")
    analyze_cmd = [
        "./tools/analyze.py",
        f"run={args.run}",
        f"policy_uri={policy_uri}",
        "+eval_db_uri=wandb://stats/navigation_db",
        "+analysis.output_path=s3://softmax-public/policydash/results.html",
        '+analysis.num_output_policies="all"',
    ] + additional_args

    if not run_command(analyze_cmd, "Analysis"):
        print("❌ Analysis failed. Exiting.")
        sys.exit(1)

    print("✅ Analysis completed successfully")

    print(f"\n🎉 Successfully added policy for {policy_uri} to leaderboard and updated dashboard!")

    dashboard_url = f"""
{METTASCOPE_REPLAY_URL}/observatory/?data=https://s3.amazonaws.com/softmax-public/policydash/results.html
"""

    print(f"📈 Dashboard URL: {dashboard_url}")


if __name__ == "__main__":
    main()
