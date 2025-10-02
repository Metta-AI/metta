#!/usr/bin/env -S uv run
"""Release qualification script for stable releases.

This script breaks down the release process into individual steps that can be
executed one at a time or all together.
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from enum import StrEnum

import asana

from metta.common.util.cli import get_user_confirmation
from metta.common.util.text_styles import green, red, yellow


class ReleaseStep(StrEnum):
    """Release process steps."""

    PREPARE_BRANCH = "prepare-branch"
    BUG_CHECK = "bug-check"
    WORKFLOW_TESTS = "workflow-tests"
    RELEASE = "release"
    ANNOUNCE = "announce"


def check_asana_bugs_automated() -> bool:
    """Check for incomplete tasks in the Active section of an Asana project using env vars.

    Returns:
        True if check passed, False if failed, None if automation not available
    """
    asana_token = os.getenv("ASANA_TOKEN")
    asana_project_id = os.getenv("ASANA_PROJECT_ID")

    if not asana_token or not asana_project_id:
        return None  # Automation not available

    print("Using automated Asana bug checking...")
    print(f"Project ID: {asana_project_id}")

    try:
        # Initialize Asana client
        configuration = asana.Configuration()
        configuration.access_token = asana_token
        api_client = asana.ApiClient(configuration)

        # Get user info to verify authentication
        users_api = asana.UsersApi(api_client)
        user = users_api.get_user("me", {})
        print(green(f"✅ Authenticated as {user['name']}"))

        # Get project sections
        sections_api = asana.SectionsApi(api_client)
        sections = sections_api.get_sections_for_project(asana_project_id, {})

        # Find the "Active" section
        active_section = None
        for section in sections:
            if section["name"].lower() == "active":
                active_section = section
                break

        if not active_section:
            print(yellow("⚠️  No 'Active' section found in project"))
            return True  # No active section means no blocking bugs

        # Get tasks in Active section
        tasks_api = asana.TasksApi(api_client)
        tasks = tasks_api.get_tasks_for_section(active_section["gid"], {"opt_fields": "name,permalink_url,completed"})

        # Filter for incomplete tasks
        incomplete_tasks = [task for task in tasks if not task.get("completed", False)]

        if len(incomplete_tasks) == 0:
            print(green("✅ No incomplete tasks found in Active section"))
            return True
        else:
            print(red(f"❌ Found {len(incomplete_tasks)} incomplete task(s) in Active section:"))
            print()
            for task in incomplete_tasks:
                print(f"  • {task['name']}")
                if task.get("permalink_url"):
                    print(f"    {task['permalink_url']}")
            print()
            return False

    except Exception as e:
        print(yellow(f"⚠️  Asana API error: {e}"))
        print("Falling back to manual check...")
        return None  # Fall back to manual


def generate_version() -> str:
    """Generate a version string based on current date and time.

    Format: YYYY.MM.DD-HHMM (e.g., 2025.10.02-1430)
    """
    return datetime.now().strftime("%Y.%m.%d-%H%M")


def run_command(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def step_1_prepare_release_branch(version: str, check_mode: bool = False) -> None:
    """Create and push the release qualification branch."""
    print(f"\n{'=' * 60}")
    print(f"STEP 1: Prepare Release Branch (v{version})")
    print(f"{'=' * 60}\n")

    branch_name = f"release-qual/{version}"

    # Check if we're in a git repo
    try:
        subprocess.run(["git", "rev-parse", "--git-dir"], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        print("Error: Not in a git repository")
        sys.exit(1)

    if check_mode:
        print(yellow(f"[CHECK MODE] Would create branch: {branch_name}"))
        print(yellow("[CHECK MODE] Would push branch to origin"))
        return

    # Create the branch
    print(f"Creating branch: {branch_name}")
    result = run_command(["git", "checkout", "-b", branch_name], check=False)

    if result.returncode != 0:
        print(red(f"Error creating branch: {result.stderr}"))
        sys.exit(1)

    # Push the branch
    print("Pushing branch to origin...")
    result = run_command(["git", "push", "-u", "origin", branch_name], check=False)

    if result.returncode != 0:
        print(red(f"Error pushing branch: {result.stderr}"))
        sys.exit(1)

    print(green(f"\n✅ Branch {branch_name} created and pushed successfully"))


def step_2_bug_status_check(version: str, check_mode: bool = False) -> None:
    """Check for blocking bugs in Asana project."""
    print(f"\n{'=' * 60}")
    print("STEP 2: Bug Status Check")
    print(f"{'=' * 60}\n")

    # Try automated check first
    result = check_asana_bugs_automated()

    if result is True:
        # Automated check passed
        print(green("\n✅ Bug check PASSED - clear for release"))
        return
    elif result is False:
        # Automated check failed
        print(red("\n❌ Bug check FAILED - resolve blocking issues before release"))
        print("\nContacts:")
        print("- Release Manager: @Robb")
        print("- Technical Lead: @Jack Heart")
        print("- Bug Triage: @Nishad Singh")
        sys.exit(1)

    # result is None - fall back to manual check
    print("Automated check not available (set ASANA_TOKEN and ASANA_PROJECT_ID to enable)")
    print("\nManual steps required:")
    print("1. Open Asana project for bug tracking")
    print("2. Verify no active/open bugs marked as blockers")
    print("3. Update bug statuses as needed in consultation with bug owners")
    print("4. Note any known issues that are acceptable for release")
    print("\nContacts:")
    print("- Release Manager: @Robb")
    print("- Technical Lead: @Jack Heart")
    print("- Bug Triage: @Nishad Singh")
    print()

    # Ask user for confirmation
    if not get_user_confirmation("Have you completed the bug status check and is it PASSED?"):
        print(red("\n❌ Bug check FAILED - user indicated issues remain"))
        sys.exit(1)

    print(green("\n✅ Bug check PASSED - user confirmed"))


def step_3_workflow_validation(version: str, check_mode: bool = False) -> None:
    """Run workflow validation."""
    print(f"\n{'=' * 60}")
    print("STEP 3: Workflow Validation")
    print(f"{'=' * 60}\n")

    print("Tool-based validation:\n")
    print("Training Tool:")
    print("  Launch validations:")
    print("    devops/validate_recipes.py launch train")
    print("  Check results:")
    print("    devops/validate_recipes.py check")
    print()
    print("Cluster Configuration:")
    print("  Launch validations:")
    print("    devops/validate_recipes.py launch cluster")
    print("  Check results:")
    print("    devops/validate_recipes.py check")
    print()
    print("\nFor each tool/recipe validation, verify:")
    print("  - Training metrics (loss, rewards, etc.)")
    print("  - Performance metrics (SPS near 40k, no significant dips)")
    print("  - Model checkpoints saved")
    print("  - Evaluation results generated")
    print("  - Replays and replay links created")
    print()
    print("Additional manual validations:")
    print("  - Play workflow: Launch play environment and verify agent navigation")
    print("  - Sweep workflow: Ask Axel via Discord to verify sweeps")
    print("  - Observatory: Ask Pasha via Discord to verify observatory")
    print("  - CI workflow: Verify CI pipeline completed successfully on main")
    print()

    # Ask user for confirmation
    if not get_user_confirmation("Have you completed all workflow validations and did they all PASS?"):
        print(red("\n❌ Workflow validation FAILED - user indicated issues remain"))
        sys.exit(1)

    print(green("\n✅ Workflow validation PASSED - user confirmed"))


def step_4_release(version: str, check_mode: bool = False) -> None:
    """Print instructions for creating the release."""
    print(f"\n{'=' * 60}")
    print("STEP 4: Release")
    print(f"{'=' * 60}\n")

    if check_mode:
        print(yellow("[CHECK MODE] ⚠️  Skipping step 4 - this step publishes tags"))
        print(yellow("Run without --check to execute this step"))
        return

    print("4.1 Prepare Release PR:")
    print(f"  1. Create release notes at: devops/stable/release-notes/v{version}.md")
    print(f"  2. Open PR from release-qual/{version} to stable")
    print("  3. Use this template for PR description:")
    print()
    print(f"## Version {version}")
    print()
    print("### Known Issues")
    print()
    print("<Notes from step 2>")
    print()
    print("### W&B Run Links")
    print()
    print("- Training: <link to job id examined in step 3.2>")
    print("- Evaluation: <link to job id examined step 3.3>")
    print()
    print("4.2 Merge Release PR and update stable tag:")
    print(f'  1. After PR approval, create annotated tag: git tag -a v{version} -m "Release version {version}"')
    print("  2. Click **Merge** on the PR")
    print(f"  3. Push tag: git push origin v{version}")
    print(f"  4. Delete qualification branch: git push origin --delete release-qual/{version}")


def step_5_announce(version: str, check_mode: bool = False) -> None:
    """Print instructions for announcing the release."""
    print(f"\n{'=' * 60}")
    print("STEP 5: Announce")
    print(f"{'=' * 60}\n")

    if check_mode:
        print(yellow("[CHECK MODE] ⚠️  Skipping step 5 - this step announces the release"))
        print(yellow("Run without --check to execute this step"))
        return

    print("Post release process completion to Discord in #eng-process")
    print("\nSuggested message:")
    print(f"Released stable version v{version} 🎉")
    print(f"Release notes: devops/stable/release-notes/v{version}.md")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Release qualification script for stable releases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--version",
        help="Version number (default: auto-generated from date YYYY.MM.DD-HHMM)",
        default=None,
    )

    parser.add_argument(
        "--step",
        type=ReleaseStep,
        choices=list(ReleaseStep),
        help="Run a specific step",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all steps (only step 1 executes, others print instructions)",
    )

    parser.add_argument(
        "--check",
        action="store_true",
        help="Check mode: skip steps that publish tags or announce releases",
    )

    args = parser.parse_args()

    # Generate or use provided version
    version = args.version if args.version else generate_version()

    if not args.step and not args.all:
        parser.print_help()
        print(f"\nAuto-generated version: {version}")
        print("\nEnvironment variables:")
        print("  ASANA_TOKEN - Personal Access Token for Asana API")
        print("  ASANA_PROJECT_ID - Asana project ID for bug tracking")
        sys.exit(0)

    # Build step kwargs
    def make_kwargs(step: ReleaseStep) -> dict:
        return {"check_mode": args.check}

    steps = {
        ReleaseStep.PREPARE_BRANCH: step_1_prepare_release_branch,
        ReleaseStep.BUG_CHECK: step_2_bug_status_check,
        ReleaseStep.WORKFLOW_TESTS: step_3_workflow_validation,
        ReleaseStep.RELEASE: step_4_release,
        ReleaseStep.ANNOUNCE: step_5_announce,
    }

    if args.all:
        for step in ReleaseStep:
            steps[step](version, **make_kwargs(step))
    elif args.step:
        steps[args.step](version, **make_kwargs(args.step))


if __name__ == "__main__":
    main()
