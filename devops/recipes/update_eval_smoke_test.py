#!/usr/bin/env python3
"""
Script to update the smoke test policy in GitHub variables.
This avoids the need for commits when updating the policy.
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from git import get_current_commit, get_current_repo, get_github_variable, set_github_variable


def confirm_update(current_value, new_value):
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


def main():
    """Main function."""
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
    print(f"New policy name: {new_policy_name}")

    # Get current value
    print("\nFetching current smoke test policy...")
    current_value = get_github_variable("EVAL_SMOKE_TEST_POLICY", repo)

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
