#!/usr/bin/env python3
"""
Script to update the smoke test policy in GitHub variables.
This avoids the need for commits when updating the policy.

Usage:
    # Update to policy for current commit (auto-generated name)
    python3 devops/recipes/update_eval_smoke_test.py

    # Update to a specific policy
    python3 devops/recipes/update_eval_smoke_test.py sasmith_20250529_ablate_1_hot_types.3

    # Update to a specific policy (alternative syntax)
    python3 devops/recipes/update_eval_smoke_test.py --policy sasmith_20250529_ablate_1_hot_types.3

The script will:
1. Check if the policy exists on wandb
2. Show the current policy value
3. Ask for confirmation before updating
4. Update the GitHub variable EVAL_SMOKE_TEST_POLICY
5. Verify the update was successful
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import Optional, Tuple

import wandb
import wandb.errors

from metta.util.colorama import blue, bold, cyan, green, red, use_colors, yellow
from metta.util.git import get_current_commit, get_current_repo, get_github_variable, set_github_variable

WANDB_ENTITY = "metta-research"
WANDB_PROJECT = "metta"
GITHUB_VARIABLE_NAME = "EVAL_SMOKE_TEST_POLICY"


def check_wandb_credentials() -> bool:
    """Check if valid W&B credentials are available."""
    if "WANDB_API_KEY" in os.environ:
        return True
    try:
        return wandb.login(anonymous="never", timeout=10)
    except (wandb.errors.AuthenticationError, wandb.errors.CommError, TimeoutError, AttributeError):
        return False


def check_wandb_policy_exists(policy_name: str) -> Tuple[bool, Optional[str]]:
    """
    Check if a policy exists on wandb.

    Returns:
        Tuple of (exists: bool, error_message: Optional[str])
    """
    try:
        api = wandb.Api()

        # Method 1: Try as an artifact
        try:
            api.artifact(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{policy_name}:latest")
            return True, None
        except Exception:
            pass

        # Method 2: Try as a run with full path
        try:
            api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{policy_name}")
            return True, None
        except Exception:
            pass

        # Method 3: Search through runs (limited to avoid long waits)
        try:
            runs = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}", filters={"display_name": policy_name})
            # Check first 10 runs only
            for i, run in enumerate(runs):
                if i >= 10:
                    break
                if run.name == policy_name or run.id == policy_name:
                    return True, None
        except Exception:
            pass

        return False, None

    except Exception as e:
        return False, f"W&B error: {e}"


def check_github_cli() -> bool:
    """Check if GitHub CLI is available and authenticated."""
    try:
        result = subprocess.run(["gh", "auth", "status"], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def generate_policy_name() -> str:
    """Generate policy name from current commit."""
    commit_sha = get_current_commit()
    if not commit_sha:
        raise ValueError("Could not get current git commit SHA")

    date_str = datetime.now().strftime("%Y%m%d")
    short_sha = commit_sha[:7]
    return f"eval_smoke_test_{date_str}_{short_sha}"


def print_status_table(
    repo: str, current_value: Optional[str], new_value: str, policy_exists: bool, commit_sha: Optional[str] = None
):
    """Print a nicely formatted status table."""
    print(f"\n{bold('=' * 60)}")
    print(f"{bold('SMOKE TEST POLICY UPDATE')}")
    print(f"{bold('=' * 60)}")

    # Repository info
    print(f"Repository:     {cyan(repo)}")

    # Current commit (if auto-generated)
    if commit_sha:
        print(f"Current commit: {cyan(commit_sha)}")

    # Policy values
    print(f"Current policy: {yellow(current_value or '<not set>')}")
    print(f"New policy:     {blue(new_value)}")

    # Policy status
    print("Policy exists:  ", end="")
    if policy_exists:
        print(green("✓ Yes"))
    else:
        print(red("✗ No"))

    print(f"{bold('=' * 60)}")


def confirm_action(prompt: str) -> bool:
    """Get user confirmation."""
    while True:
        try:
            response = input(f"\n{prompt} (y/n): ").lower().strip()
            if response in ["y", "yes"]:
                return True
            elif response in ["n", "no"]:
                return False
            else:
                print("Please enter 'y' for yes or 'n' for no.")
        except (EOFError, KeyboardInterrupt):
            print("\nOperation cancelled by user.")
            return False


def launch_training_job(policy_name: str) -> bool:
    """Launch a training job for the policy."""
    cmd = [
        "python3",
        "-m",
        "devops.aws.batch.launch_task",
        "--cmd=train",
        "trainer.env=env/mettagrid/simple",
        "--timeout-minutes=180",
        f"--run={policy_name}",
    ]

    try:
        print(f"\n{blue('Command:')} {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n{red('✗ Failed to launch training job:')} {e}")
        if e.returncode == 1:
            print(f"\n{yellow('This might be due to missing dependencies.')}")
            print("Make sure you have all dependencies installed:")
            print("  ./devops/setup_dev.sh")
        return False
    except FileNotFoundError:
        print(f"\n{red('✗ Failed to launch training job: python3 not found')}")
        return False


def handle_missing_policy(policy_name: str, is_auto_generated: bool):
    """Handle the case where a policy doesn't exist on wandb."""
    print(f"\n{yellow('⚠️  Warning:')} Policy '{policy_name}' not found on wandb!")
    print("This policy needs to be trained before it can be used for smoke tests.")

    # Only offer auto-training for auto-generated policies
    if is_auto_generated and policy_name.startswith("eval_smoke_test_"):
        print("\nWould you like to launch a training job for this policy?")

        if confirm_action("Launch training job?"):
            print(f"\n{blue('Launching training job...')}")
            if launch_training_job(policy_name):
                print(f"\n{green('✓ Training job launched successfully!')}")
                print(f"Policy name: {cyan(policy_name)}")
                print(
                    "\nThe training will take some time. Once complete, run this script again to update the smoke test."
                )
                return False  # Don't continue with update

    else:
        print(f"\n{yellow('Note:')} Auto-training is only available for auto-generated policy names.")

    # Manual training command
    print("\nTo train this policy manually, run:")
    cmd_parts = [
        "python3 -m devops.aws.batch.launch_task",
        "--cmd=train",
        "trainer.env=env/mettagrid/simple",
        "--timeout-minutes=180",
        f"--run={policy_name}",
    ]
    print(f"  {cyan(' '.join(cmd_parts))}")

    # Ask if they want to update anyway
    if is_auto_generated:
        return False  # Don't update if auto-generated and not trained
    else:
        print("\nDo you want to update the GitHub variable anyway?")
        print("(Only do this if you're sure the policy exists or will exist soon)")
        return confirm_action("Continue with update?")


def main():
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
        """,
    )
    parser.add_argument(
        "policy",
        nargs="?",
        help="Specific policy name to use (optional). If not provided, generates name from current commit.",
    )
    parser.add_argument("--policy", dest="policy_flag", help="Alternative way to specify policy name")

    args = parser.parse_args()

    # Configure colors based on terminal
    use_colors(sys.stdout.isatty())

    try:
        # Check GitHub CLI
        if not check_github_cli():
            print(f"{red('Error:')} GitHub CLI (gh) is not installed or not authenticated")
            print("Please install from: https://cli.github.com/")
            print("Then authenticate with: gh auth login")
            return 1

        # Get repository
        repo = get_current_repo()
        if not repo:
            print(f"{yellow('Warning:')} Could not determine current repository")
            print("Make sure you're in a git repository with GitHub remote")
            return 1

        # Determine policy name
        custom_policy = args.policy or args.policy_flag
        commit_sha = None

        if custom_policy:
            policy_name = custom_policy
            is_auto_generated = False
            print(f"\nUsing specified policy: {cyan(policy_name)}")
        else:
            commit_sha = get_current_commit()
            if not commit_sha:
                print(f"{red('Error:')} Could not get current git commit SHA")
                return 1

            policy_name = generate_policy_name()
            is_auto_generated = True

        # Get current value
        print(f"\n{blue('Fetching current smoke test policy...')}")
        current_value = get_github_variable(GITHUB_VARIABLE_NAME, repo)

        # Check if wandb is configured
        if not check_wandb_credentials():
            print(f"\n{yellow('W&B is not configured. Please run:')}")
            print(f"  {cyan('wandb login')}")
            print("Or set WANDB_API_KEY environment variable")
            print("\nContinuing anyway...")

        # Check if policy exists
        print(f"\n{blue('Checking if policy exists on wandb...')}")
        policy_exists, error_msg = check_wandb_policy_exists(policy_name)

        if error_msg:
            print(f"{yellow('Warning:')} {error_msg}")

        # Print status
        print_status_table(repo, current_value, policy_name, policy_exists, commit_sha)

        # Check if already set
        if current_value == policy_name:
            print(f"\n{green('✓ The smoke test policy is already set to:')} {current_value}")
            print("No update needed.")
            return 0

        # Handle missing policy
        if not policy_exists:
            should_continue = handle_missing_policy(policy_name, is_auto_generated)
            if not should_continue:
                return 0

        # Confirm update
        if not confirm_action("Do you want to update the smoke test policy?"):
            print("Update cancelled.")
            return 0

        # Update the variable
        print(f"\n{blue('Updating')} {GITHUB_VARIABLE_NAME} {blue('variable...')}")

        if set_github_variable(GITHUB_VARIABLE_NAME, policy_name, repo):
            print(f"\n{green('✓ Success!')} The smoke test will now use the new policy.")
            print("No commit or PR needed - the change is immediate.")

            # Verify update
            print(f"\n{blue('Verifying update...')}")
            time.sleep(1)  # Give GitHub a moment to update

            updated_value = get_github_variable(GITHUB_VARIABLE_NAME, repo)
            if updated_value == policy_name:
                print(f"{green('✓ Confirmed:')} {GITHUB_VARIABLE_NAME} = {cyan(updated_value)}")
            else:
                print(f"{yellow('⚠ Warning:')} Verification failed. Current value: {updated_value}")
                return 1
        else:
            print(f"\n{red('✗ Failed to update the variable.')}")
            print("Make sure you have the necessary permissions and are authenticated with gh CLI.")
            return 1

        return 0

    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        return 1
    except Exception as e:
        print(f"\n{red('Unexpected error:')} {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
