#!/usr/bin/env python3
"""
PR Assignment Script
Assigns users, reviewers, and labels to pull requests based on configuration.
"""

import json
import os
import random
import subprocess
import sys
from typing import List, Optional, Set, Tuple


def run_gh_command(args: List[str]) -> Tuple[bool, str]:
    """Run a GitHub CLI command and return success status and output."""
    try:
        result = subprocess.run(["gh"] + args, capture_output=True, text=True, check=True)
        return True, result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Warning: Command failed: {' '.join(['gh'] + args)}")
        print(f"Error: {e.stderr}")
        return False, e.stderr


def get_pr_info(pr_number: str, repo: str) -> dict:
    """Get PR information including author, assignees, reviewers, and labels."""
    success, output = run_gh_command(
        ["pr", "view", pr_number, "--repo", repo, "--json", "author,assignees,reviewRequests,labels"]
    )

    if not success:
        return {}

    try:
        return json.loads(output)
    except json.JSONDecodeError:
        print("Warning: Failed to parse PR info")
        return {}


def is_empty(value: Optional[str]) -> bool:
    """Check if a string is empty or contains only whitespace."""
    return not value or not value.strip()


def is_true(value: str) -> bool:
    """Check if a value represents true (case insensitive)."""
    return value.lower() == "true"


def parse_list(value: str) -> List[str]:
    """Parse a comma-separated list into a list of trimmed strings."""
    if is_empty(value):
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def select_random(items: List[str], exclude: Optional[str] = None) -> Optional[str]:
    """Randomly select an item from a list, optionally excluding one value."""
    if not items:
        return None

    filtered_items = [item for item in items if item != exclude]

    if not filtered_items:
        return None

    return random.choice(filtered_items)


def clear_assignees(pr_number: str, repo: str, current_assignees: List[str]) -> bool:
    """Clear all existing assignees from a PR."""
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
    """Clear all existing review requests from a PR."""
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
    """Clear all existing labels from a PR."""
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
    """Add assignees to a PR. Returns set of successfully added assignees."""
    added = set()
    for assignee in assignees:
        success, _ = run_gh_command(["pr", "edit", pr_number, "--add-assignee", assignee, "--repo", repo])
        if success:
            added.add(assignee)

    return added


def add_reviewers(pr_number: str, repo: str, reviewers: List[str]) -> Set[str]:
    """Add reviewers to a PR. Returns set of successfully added reviewers."""
    added = set()
    for reviewer in reviewers:
        success, _ = run_gh_command(["pr", "edit", pr_number, "--add-reviewer", reviewer, "--repo", repo])
        if success:
            added.add(reviewer)

    return added


def add_labels(pr_number: str, repo: str, labels: List[str]) -> Set[str]:
    """Add labels to a PR. Returns set of successfully added labels."""
    added = set()
    for label in labels:
        success, _ = run_gh_command(["pr", "edit", pr_number, "--add-label", label, "--repo", repo])
        if success:
            added.add(label)

    return added


def post_comment(pr_number: str, repo: str, comment: str) -> None:
    """Post a comment on a PR."""
    run_gh_command(["pr", "comment", pr_number, "--body", comment, "--repo", repo])


def main():
    """Main function to process PR assignments."""
    # Get arguments
    if len(sys.argv) < 10:
        print("Error: Not enough arguments provided")
        sys.exit(1)

    pr_number = sys.argv[1]
    possible_assignees = sys.argv[2]
    possible_reviewers = sys.argv[3]
    forced_assignees = sys.argv[4]
    forced_reviewers = sys.argv[5]
    forced_labels = sys.argv[6]
    clear_existing_assignees = sys.argv[7]
    clear_existing_reviewers = sys.argv[8]
    clear_existing_labels = sys.argv[9]

    repo = os.environ.get("GITHUB_REPOSITORY", "")

    # Get PR information
    pr_info = get_pr_info(pr_number, repo)
    pr_author = pr_info.get("author", {}).get("login", "")
    current_assignees = [a["login"] for a in pr_info.get("assignees", [])]
    current_reviewers = [r["login"] for r in pr_info.get("reviewRequests", [])]
    current_labels = [l["name"] for l in pr_info.get("labels", [])]

    print(f"Processing PR #{pr_number} by @{pr_author}")

    # Track actions taken
    actions = []
    assigned = set()
    reviewed = set()
    labeled = set()

    # Clear existing assignees if requested
    if is_true(clear_existing_assignees):
        print("Clearing existing assignees...")
        if clear_assignees(pr_number, repo, current_assignees):
            actions.append("Cleared all existing assignees")

    # Clear existing reviewers if requested
    if is_true(clear_existing_reviewers):
        print("Clearing existing review requests...")
        if clear_reviewers(pr_number, repo, current_reviewers):
            actions.append("Cleared all existing review requests")

    # Clear existing labels if requested
    if is_true(clear_existing_labels):
        print("Clearing existing labels...")
        if clear_labels(pr_number, repo, current_labels):
            actions.append("Cleared all existing labels")

    # Process forced assignees
    forced_assignee_list = parse_list(forced_assignees)
    if forced_assignee_list:
        print(f"Adding forced assignees: {', '.join(forced_assignee_list)}")
        assigned = add_assignees(pr_number, repo, forced_assignee_list)

    # Process random assignee
    possible_assignee_list = parse_list(possible_assignees)
    if possible_assignee_list:
        selected_assignee = select_random(possible_assignee_list, pr_author)
        if selected_assignee and selected_assignee not in assigned:
            print(f"Selected random assignee: {selected_assignee}")
            if add_assignees(pr_number, repo, [selected_assignee]):
                assigned.add(selected_assignee)
                print(f"Successfully assigned PR #{pr_number} to {selected_assignee}")

    # Process forced reviewers
    forced_reviewer_list = parse_list(forced_reviewers)
    if forced_reviewer_list:
        print(f"Adding forced reviewers: {', '.join(forced_reviewer_list)}")
        reviewed = add_reviewers(pr_number, repo, forced_reviewer_list)

    # Process random reviewer
    possible_reviewer_list = parse_list(possible_reviewers)
    if possible_reviewer_list:
        selected_reviewer = select_random(possible_reviewer_list, pr_author)
        if selected_reviewer and selected_reviewer not in reviewed:
            print(f"Selected random reviewer: {selected_reviewer}")
            if add_reviewers(pr_number, repo, [selected_reviewer]):
                reviewed.add(selected_reviewer)
                print(f"Successfully requested review from {selected_reviewer} for PR #{pr_number}")

    # Process forced labels
    forced_label_list = parse_list(forced_labels)
    if forced_label_list:
        print(f"Setting forced labels: {', '.join(forced_label_list)}")
        # Clear existing labels if not already cleared
        if not is_true(clear_existing_labels) and current_labels:
            clear_labels(pr_number, repo, current_labels)
        labeled = add_labels(pr_number, repo, forced_label_list)

    # Build and post summary comment
    if assigned:
        actions.append(f"Assigned to: {' '.join(f'@{a}' for a in sorted(assigned))}")

    if reviewed:
        actions.append(f"Review requested from: {' '.join(f'@{r}' for r in sorted(reviewed))}")

    if labeled:
        actions.append(f"Labels set: {' '.join(f'`{l}`' for l in sorted(labeled))}")

    if actions:
        comment = "PR automatically processed:\n" + "\n".join(f"- {action}" for action in actions)
        post_comment(pr_number, repo, comment)
    else:
        print("No actions were taken on this PR - no assignees, reviewers, or labels were set or cleared.")


if __name__ == "__main__":
    main()
