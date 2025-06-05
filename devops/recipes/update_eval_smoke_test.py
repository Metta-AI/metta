#!/usr/bin/env python
"""
Script to update the smoke test policy in GitHub variables.
This avoids the need for commits when updating the policy.

Usage:
    # Update to policy for current commit (auto-generated name)
    ./devops/recipes/update_eval_smoke_test.py

    # Update to a specific policy
    ./devops/recipes/update_eval_smoke_test.py sasmith_20250529_ablate_1_hot_types.3

    # Update to a specific policy (alternative syntax)
    ./devops/recipes/update_eval_smoke_test.py --policy sasmith_20250529_ablate_1_hot_types.3

The script will:
1. Check if the policy exists on wandb
2. Show the current policy value
3. Ask for confirmation before updating
4. Update the GitHub variable EVAL_SMOKE_TEST_POLICY
5. Verify the update was successful
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from metta.util.git import get_current_commit, get_current_repo, get_github_variable, set_github_variable


def check_wandb_policy_exists(policy_name: str) -> bool:
    """Check if a policy exists on wandb."""
    try:
        # Try using wandb CLI to check if the artifact exists
        cmd = ["wandb", "artifact", "get", f"{policy_name}:latest"]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            return True
        else:
            # Also check if it exists as a run
            cmd = ["wandb", "run", "get", policy_name]
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0
    except Exception:
        return False


def confirm_update(current_value: str | None, new_value: str) -> bool:
    """Ask user to confirm the update."""
    print("\n" + "="*60)
    print("SMOKE TEST POLICY UPDATE")
    print("="*60)
    print(f"Current value: {current_value or '<not set>'}")
    print(f"New value:     {new_value}")
    print("="*60)

    while True:
        response = input("\nDo you want to update the smoke test policy? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        else:
            print("Please enter 'y' for yes or 'n' for no.")


def check_dependencies() -> bool:
    """Check if required dependencies are installed."""
    try:
        import boto3
        return True
    except ImportError as e:
        import sys
        print(f"Debug: Failed to import boto3: {e}")
        print(f"Debug: Python: {sys.executable}")
        print(f"Debug: Python version: {sys.version}")
        return False


def main() -> None:
    """Main function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Update the GitHub variable for eval smoke test policy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use auto-generated policy name for current commit
  %(prog)s

  # Use a specific policy
  %(prog)s my_custom_policy_name
  %(prog)s --policy my_custom_policy_name
        """
    )
    parser.add_argument(
        "policy",
        nargs="?",
        help="Specific policy name to use (optional). If not provided, generates name from current commit."
    )
    parser.add_argument(
        "--policy",
        dest="policy_flag",
        help="Alternative way to specify policy name"
    )

    args = parser.parse_args()

    # Determine which policy name to use
    custom_policy = args.policy or args.policy_flag

    # Check if gh CLI is available
    try:
        subprocess.run(["gh", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: GitHub CLI (gh) is not installed or not in PATH")
        print("Please install it from: https://cli.github.com/")
        sys.exit(1)

    # Get current repository
    repo = get_current_repo()
    if not repo:
        print("Warning: Could not determine current repository")
        print("You may need to specify it manually with --repo flag")
    else:
        print(f"Repository: {repo}")

    # Determine the policy name to use
    if custom_policy:
        new_policy_name = custom_policy
        print(f"\nUsing specified policy: {new_policy_name}")
    else:
        # Get current git commit SHA
        commit_sha = get_current_commit()
        if not commit_sha:
            print("Error: Could not get current git commit SHA")
            sys.exit(1)

        # Get current date and short SHA
        date_str = datetime.now().strftime("%Y%m%d")
        short_sha = commit_sha[:7]

        # Construct policy name
        new_policy_name = f"eval_smoke_test_{date_str}_{short_sha}"

        print(f"\nCurrent commit: {commit_sha}")
        print(f"Generated policy name: {new_policy_name}")

    # Get current value first (before checking if policy exists)
    print("\nFetching current smoke test policy...")
    current_value = get_github_variable("EVAL_SMOKE_TEST_POLICY", repo)
    if current_value:
        print(f"Current policy: {current_value}")
    else:
        print("Current policy: <not set>")

    # Check if the policy exists on wandb
    print(f"\nChecking if new policy exists on wandb: {new_policy_name}")
    if not check_wandb_policy_exists(new_policy_name):
        print(f"\n⚠️  Warning: Policy '{new_policy_name}' not found on wandb!")
        print("This policy needs to be trained before it can be used for smoke tests.")

        # Only offer to launch training for auto-generated policy names
        if not custom_policy and new_policy_name.startswith("eval_smoke_test_"):
            # Check if boto3 is available before offering to launch
            if check_dependencies():
                print("\nWould you like to launch a training job for this policy?")
                response = input("Launch training job? (y/n): ").lower().strip()

                if response in ['y', 'yes']:
                    print("\nLaunching training job...")
                    cmd = [
                        "python3", "-m", "devops.aws.batch.launch_task",
                        "--cmd=train",
                        "trainer.env=env/mettagrid/simple",
                        "--timeout-minutes=180",
                        f"--run={new_policy_name}"
                    ]

                    try:
                        print(f"Command: {' '.join(cmd)}")
                        subprocess.run(cmd, check=True)
                        print(f"\n✓ Training job launched successfully!")
                        print(f"Policy name: {new_policy_name}")
                        print("\nThe training will take some time. Once complete, run this script again to update the smoke test.")
                        return
                    except subprocess.CalledProcessError as e:
                        print(f"\n✗ Failed to launch training job: {e}")

                        # Check if it's a module import error
                        if e.returncode == 1:
                            print("\nThis might be due to missing dependencies.")
                            print("Make sure you have all dependencies installed:")
                            print("  ./devops/setup_dev.sh")

                        print("\nYou may need to launch it manually:")
                        print(f"  {' '.join(cmd)}")
                        return
                else:
                    print("\nTo train this policy manually, run:")
                    print(f"  python3 -m devops.aws.batch.launch_task --cmd=train trainer.env=env/mettagrid/simple --timeout-minutes=180 --run={new_policy_name}")
            else:
                print("\nNote: Training job launch is unavailable (boto3 not installed).")
                print("To train this policy, run the following on a machine with AWS setup:")
                print(f"  python3 -m devops.aws.batch.launch_task --cmd=train trainer.env=env/mettagrid/simple --timeout-minutes=180 --run={new_policy_name}")
        else:
            print("\nNote: Auto-training is only available for auto-generated policy names.")
            print("For custom policy names, please ensure the policy exists on wandb before updating.")

        print("\nDo you want to update the GitHub variable anyway?")
        print("(Only do this if you're sure the policy exists or will exist soon)")
        response = input("Continue with update? (y/n): ").lower().strip()
        if response not in ['y', 'yes']:
            print("Update cancelled.")
            return
    else:
        print(f"✓ Policy found on wandb")

    # Check if already set to this value
    if current_value == new_policy_name:
        print(f"\n✓ The smoke test policy is already set to: {current_value}")
        print("No update needed.")
        return

    # Ask for confirmation
    if not confirm_update(current_value, new_policy_name):
        print("\nUpdate cancelled.")
        return

    # Update the GitHub variable
    print(f"\nUpdating EVAL_SMOKE_TEST_POLICY variable...")

    if set_github_variable("EVAL_SMOKE_TEST_POLICY", new_policy_name, repo):
        print("\n✓ Success! The smoke test will now use the new policy.")
        print("No commit or PR needed - the change is immediate.")

        # Verify the update
        print("\nVerifying update...")
        updated_value = get_github_variable("EVAL_SMOKE_TEST_POLICY", repo)
        if updated_value == new_policy_name:
            print(f"✓ Confirmed: EVAL_SMOKE_TEST_POLICY = {updated_value}")
        else:
            print(f"⚠ Warning: Verification failed. Current value: {updated_value}")
    else:
        print("\n✗ Failed to update the variable.")
        print("Make sure you have the necessary permissions and are authenticated with gh CLI.")
        sys.exit(1)


if __name__ == "__main__":
    main()
