#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
PR Assignment Script
Assigns users, reviewers, and labels to pull requests based on configuration.
Includes special handling for Dependabot PRs to add version labels.
"""

import json
import os
import random
import re
import subprocess
import sys
from typing import List, Optional, Set, Tuple


def run_gh_command(args: List[str]) -> Tuple[bool, str]:
    """Execute GitHub CLI commands with error handling to prevent script failures from non-critical operations."""
    try:
        result = subprocess.run(["gh"] + args, capture_output=True, text=True, check=True)
        return True, result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Warning: Command failed: {' '.join(['gh'] + args)}")
        print(f"Error: {e.stderr}")
        return False, e.stderr


def get_pr_info(pr_number: str, repo: str) -> dict:
    """Fetch PR metadata to understand current state before making changes."""
    success, output = run_gh_command(
        ["pr", "view", pr_number, "--repo", repo, "--json", "author,assignees,reviewRequests,labels,title"]
    )

    if not success:
        return {}

    try:
        return json.loads(output)
    except json.JSONDecodeError:
        print("Warning: Failed to parse PR info")
        return {}


def is_empty(value: Optional[str]) -> bool:
    """Safely handle potentially None values from configuration parsing."""
    return not value or not value.strip()


def is_true(value: str) -> bool:
    """Normalize boolean string inputs from action parameters to handle case variations."""
    return value.lower() == "true"


def parse_list(value: str) -> List[str]:
    """Convert action input strings to usable lists, filtering empty values from malformed input."""
    if is_empty(value):
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def select_random(items: List[str], exclude: Optional[str] = None) -> Optional[str]:
    """Ensure fair distribution of assignments while preventing self-assignment to PR authors."""
    if not items:
        return None

    filtered_items = [item for item in items if item != exclude]

    if not filtered_items:
        return None

    return random.choice(filtered_items)


def clear_assignees(pr_number: str, repo: str, current_assignees: List[str]) -> bool:
    """Remove existing assignments to reset state before applying new assignment logic."""
    if not current_assignees:
        print("No existing assignees to clear")
        return False

    cleared = False
    for assignee in current_assignees:
        success, _ = run_gh_command(["pr", "edit", pr_number, "--remove-assignee", assignee, "--repo", repo])
        if success:
            print(f"Removed assignee: {assignee}")
            cleared = True

    return cleared


def clear_reviewers(pr_number: str, repo: str, current_reviewers: List[str]) -> bool:
    """Reset review state to prevent accumulation of stale review requests."""
    if not current_reviewers:
        print("No existing review requests to clear")
        return False

    cleared = False
    for reviewer in current_reviewers:
        success, _ = run_gh_command(["pr", "edit", pr_number, "--remove-reviewer", reviewer, "--repo", repo])
        if success:
            print(f"Removed review request from: {reviewer}")
            cleared = True

    return cleared


def clear_labels(pr_number: str, repo: str, current_labels: List[str]) -> bool:
    """Remove existing labels to ensure clean state for forced label application."""
    if not current_labels:
        print("No existing labels to clear")
        return False

    cleared = False
    for label in current_labels:
        success, _ = run_gh_command(["pr", "edit", pr_number, "--remove-label", label, "--repo", repo])
        if success:
            print(f"Removed label: {label}")
            cleared = True

    return cleared


def add_assignees(pr_number: str, repo: str, assignees: List[str]) -> Set[str]:
    """Track successful assignments to build accurate summary comments and handle partial failures."""
    added = set()
    for assignee in assignees:
        success, _ = run_gh_command(["pr", "edit", pr_number, "--add-assignee", assignee, "--repo", repo])
        if success:
            added.add(assignee)

    return added


def add_reviewers(pr_number: str, repo: str, reviewers: List[str]) -> Set[str]:
    """Track successful review requests to build accurate summary comments and handle partial failures."""
    added = set()
    for reviewer in reviewers:
        success, _ = run_gh_command(["pr", "edit", pr_number, "--add-reviewer", reviewer, "--repo", repo])
        if success:
            added.add(reviewer)

    return added


def add_labels(pr_number: str, repo: str, labels: List[str]) -> Set[str]:
    """Track successful label applications to build accurate summary comments and handle partial failures."""
    added = set()
    for label in labels:
        success, _ = run_gh_command(["pr", "edit", pr_number, "--add-label", label, "--repo", repo])
        if success:
            added.add(label)

    return added


def post_comment(pr_number: str, repo: str, comment: str) -> None:
    """Provide visibility into automated actions taken by the workflow."""
    run_gh_command(["pr", "comment", pr_number, "--body", comment, "--repo", repo])


def detect_version_change(pr_title: str) -> Optional[str]:
    """
    Detect version change type from Dependabot PR title.
    Returns 'major', 'minor', 'patch', or None.
    """
    # Common patterns in Dependabot PR titles:
    # "Bump webpack from 5.1.0 to 5.2.0"
    # "Update rails requirement from ~> 6.1.0 to ~> 7.0.0"
    # "Bump @types/node from 14.14.37 to 16.0.0"

    # Look for version patterns
    version_pattern = r"from .*?([0-9]+)\.([0-9]+)\.([0-9]+).*? to .*?([0-9]+)\.([0-9]+)\.([0-9]+)"
    match = re.search(version_pattern, pr_title)

    if match:
        from_major, from_minor, from_patch = match.groups()[:3]
        to_major, to_minor, to_patch = match.groups()[3:]

        if from_major != to_major:
            return "major"
        elif from_minor != to_minor:
            return "minor"
        elif from_patch != to_patch:
            return "patch"

    return None


def ensure_version_labels_exist(repo: str) -> None:
    """Pre-create semantic version labels for Dependabot PRs to avoid label creation failures."""
    labels = [
        ("major", "FF0000", "Major version update"),
        ("minor", "FFFF00", "Minor version update"),
        ("patch", "00FF00", "Patch version update"),
    ]

    for name, color, description in labels:
        run_gh_command(["label", "create", name, "--color", color, "--description", description, "--force"])


def main():
    """Main function to process PR assignments."""
    # Get arguments from environment variables instead of sys.argv
    pr_number = os.environ.get("PR_NUMBER", "")
    possible_assignees = os.environ.get("POSSIBLE_ASSIGNEES", "")
    possible_reviewers = os.environ.get("POSSIBLE_REVIEWERS", "")
    forced_assignees = os.environ.get("FORCED_ASSIGNEES", "")
    forced_reviewers = os.environ.get("FORCED_REVIEWERS", "")
    forced_labels = os.environ.get("FORCED_LABELS", "")
    clear_existing_assignees = os.environ.get("CLEAR_EXISTING_ASSIGNEES", "false")
    clear_existing_reviewers = os.environ.get("CLEAR_EXISTING_REVIEWERS", "false")
    clear_existing_labels = os.environ.get("CLEAR_EXISTING_LABELS", "false")

    if not pr_number:
        print("Error: PR_NUMBER environment variable not set")
        sys.exit(1)

    repo = os.environ.get("GITHUB_REPOSITORY", "")

    pr_info = get_pr_info(pr_number, repo)
    pr_author = pr_info.get("author", {}).get("login", "")
    pr_title = pr_info.get("title", "")
    current_assignees = [a["login"] for a in pr_info.get("assignees", [])]
    current_reviewers = [r["login"] for r in pr_info.get("reviewRequests", [])]
    current_labels = [label["name"] for label in pr_info.get("labels", [])]

    print(f"Processing PR #{pr_number} by @{pr_author}")

    if pr_author == "dependabot[bot]":
        print("Detected Dependabot PR, checking for version changes...")
        ensure_version_labels_exist(repo)

        version_type = detect_version_change(pr_title)
        if version_type:
            print(f"Detected {version_type} version update")
            # Check if the label already exists
            if version_type not in current_labels:
                success, _ = run_gh_command(["pr", "edit", pr_number, "--add-label", version_type, "--repo", repo])
                if success:
                    print(f"Added '{version_type}' label to PR")
                    current_labels.append(version_type)

    actions = []
    assigned = set()
    reviewed = set()
    labeled = set()

    if is_true(clear_existing_assignees):
        print("Clearing existing assignees...")
        if clear_assignees(pr_number, repo, current_assignees):
            actions.append("Cleared all existing assignees")

    if is_true(clear_existing_reviewers):
        print("Clearing existing review requests...")
        if clear_reviewers(pr_number, repo, current_reviewers):
            actions.append("Cleared all existing review requests")

    if is_true(clear_existing_labels):
        print("Clearing existing labels...")
        # For Dependabot PRs, preserve version labels
        if pr_author == "dependabot[bot]":
            labels_to_preserve = ["major", "minor", "patch"]
            labels_to_clear = [label for label in current_labels if label not in labels_to_preserve]
        else:
            labels_to_clear = current_labels

        if clear_labels(pr_number, repo, labels_to_clear):
            actions.append("Cleared all existing labels")

    forced_assignee_list = parse_list(forced_assignees)
    if forced_assignee_list:
        print(f"Adding forced assignees: {', '.join(forced_assignee_list)}")
        assigned = add_assignees(pr_number, repo, forced_assignee_list)

    possible_assignee_list = parse_list(possible_assignees)
    if possible_assignee_list:
        selected_assignee = select_random(possible_assignee_list, pr_author)
        if selected_assignee and selected_assignee not in assigned:
            print(f"Selected random assignee: {selected_assignee}")
            if add_assignees(pr_number, repo, [selected_assignee]):
                assigned.add(selected_assignee)
                print(f"Successfully assigned PR #{pr_number} to {selected_assignee}")

    forced_reviewer_list = parse_list(forced_reviewers)
    if forced_reviewer_list:
        print(f"Adding forced reviewers: {', '.join(forced_reviewer_list)}")
        reviewed = add_reviewers(pr_number, repo, forced_reviewer_list)

    possible_reviewer_list = parse_list(possible_reviewers)
    if possible_reviewer_list:
        selected_reviewer = select_random(possible_reviewer_list, pr_author)
        if selected_reviewer and selected_reviewer not in reviewed:
            print(f"Selected random reviewer: {selected_reviewer}")
            if add_reviewers(pr_number, repo, [selected_reviewer]):
                reviewed.add(selected_reviewer)
                print(f"Successfully requested review from {selected_reviewer} for PR #{pr_number}")

    forced_label_list = parse_list(forced_labels)
    if forced_label_list:
        print(f"Setting forced labels: {', '.join(forced_label_list)}")
        # Forced labels require clean slate to prevent mixing with existing labels
        if not is_true(clear_existing_labels) and current_labels:
            clear_labels(pr_number, repo, current_labels)
        labeled = add_labels(pr_number, repo, forced_label_list)

    if assigned:
        actions.append(f"Assigned to: {' '.join(f'@{a}' for a in sorted(assigned))}")

    if reviewed:
        actions.append(f"Review requested from: {' '.join(f'@{r}' for r in sorted(reviewed))}")

    if labeled:
        actions.append(f"Labels set: {' '.join(f'`{label}`' for label in sorted(labeled))}")

    if actions:
        comment = "PR automatically processed:\n" + "\n".join(f"- {action}" for action in actions)
        post_comment(pr_number, repo, comment)
    else:
        print("No actions were taken on this PR - no assignees, reviewers, or labels were set or cleared.")


if __name__ == "__main__":
    main()
