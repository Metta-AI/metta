#!/usr/bin/env -S uv run
"""Add policy to evaluation leaderboard and update dashboard."""

import argparse
import subprocess
import sys
from typing import List

import wandb
from wandb.errors import CommError

from metta.common.util.constants import METTASCOPE_REPLAY_URL


def check_policy_exists(run_name: str) -> bool:
    """
    Check if a policy exists in WANDB.

    Args:
        run_name: The run name to check

    Returns:
        True if policy exists, False otherwise
    """
    try:
        api = wandb.Api()
        run = api.run(f"metta-research/metta/{run_name}")
        print(f"✅ Policy found: {run.id} (state: {run.state})")
        return True
    except CommError as e:
        if "404" in str(e):
            print(f"❌ Policy not found: metta-research/metta/{run_name}")
        else:
            print(f"❌ WANDB API error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error checking policy: {e}")
        return False


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
%(prog)s --run b.username.test_run
""",
    )

    parser.add_argument("--run", required=True, help="Your run name (e.g., b.$USER.test_run)")

    # Capture additional arguments for Hydra
    args, additional_args = parser.parse_known_args()

    wandb_path = f"wandb://run/{args.run}"
    print(f"Adding policy to eval leaderboard with run name: {args.run}")

    if additional_args:
        print(f"Additional arguments: {' '.join(additional_args)}")

    # Step 1: Verify policy exists on WANDB
    if not check_policy_exists(args.run):
        print("\n💡 If this is expected (e.g., policy stored elsewhere), use --skip-check")
        sys.exit(1)
    print("✅ Policy verification passed")

    # Step 2: Run the simulation
    print("\n🚀 Step 2: Running simulation...")
    sim_cmd = [
        "./tools/sim.py",
        "sim=navigation",
        f"run={args.run}",
        f"policy_uri={wandb_path}",
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
        f"policy_uri={wandb_path}",
        "+eval_db_uri=wandb://stats/navigation_db",
        "+analysis.output_path=s3://softmax-public/policydash/results.html",
        '+analysis.num_output_policies="all"',
    ] + additional_args

    if not run_command(analyze_cmd, "Analysis"):
        print("❌ Analysis failed. Exiting.")
        sys.exit(1)

    print("✅ Analysis completed successfully")

    print(f"\n🎉 Successfully added policy for {wandb_path} to leaderboard and updated dashboard!")

    dashboard_url = f"""
{METTASCOPE_REPLAY_URL}/observatory/?data=https://s3.amazonaws.com/softmax-public/policydash/results.html
"""

    print(f"📈 Dashboard URL: {dashboard_url}")


if __name__ == "__main__":
    main()
