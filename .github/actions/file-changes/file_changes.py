#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "PyGithub>=2.1.1",
# ]
# ///
"""
Check if files matching patterns have been changed in a pull request or push.
"""

import os
import re
import sys
from typing import List

from github import Github


def parse_comma_separated(value: str) -> List[str]:
    """Parse comma-separated string into list of non-empty strings."""
    return [item.strip() for item in value.split(",") if item.strip()]


def pattern_matches(filename: str, pattern: str) -> bool:
    """Check if filename matches the given pattern."""
    if "*" in pattern:
        # Convert glob-like pattern to regex
        regex_pattern = pattern.replace(".", r"\.").replace("*", ".*")
        return bool(re.match(regex_pattern, filename))
    else:
        # Simple substring match
        return pattern in filename


def main():
    """Main entry point."""
    # Get inputs from environment
    github_token = os.environ.get("GITHUB_TOKEN")
    github_repository = os.environ.get("GITHUB_REPOSITORY")
    github_event_name = os.environ.get("GITHUB_EVENT_NAME")
    github_event_path = os.environ.get("GITHUB_EVENT_PATH")

    patterns = parse_comma_separated(os.environ.get("PATTERNS", ""))
    specific_files = parse_comma_separated(os.environ.get("SPECIFIC_FILES", ""))
    directory_paths = parse_comma_separated(os.environ.get("DIRECTORY_PATHS", ""))

    print(f"Event type: {github_event_name}")
    print(f"Parsed {len(patterns)} patterns to check: {', '.join(patterns)}")
    print(f"Parsed {len(specific_files)} specific files to check")
    print(f"Parsed {len(directory_paths)} directory paths to check")

    # Initialize GitHub client
    g = Github(github_token)
    repo = g.get_repo(github_repository)

    changed_files = []

    try:
        if github_event_name == "pull_request":
            # Load event data to get PR number
            import json

            with open(github_event_path) as f:
                event_data = json.load(f)

            pr_number = event_data["pull_request"]["number"]
            print(f"Fetching changed files for PR #{pr_number}...")

            pr = repo.get_pull(pr_number)
            changed_files = list(pr.get_files())

        elif github_event_name == "push":
            print("Push event detected. Fetching commit information...")

            # Load event data to get commit refs
            import json

            with open(github_event_path) as f:
                event_data = json.load(f)

            base_ref = event_data["before"]
            head_ref = event_data["after"]

            # Compare commits
            comparison = repo.compare(base_ref, head_ref)
            changed_files = comparison.files

        elif github_event_name == "workflow_dispatch":
            print("Manual workflow dispatch. Assuming relevant files have changed.")
            # Set output and exit
            with open(os.environ["GITHUB_OUTPUT"], "a") as f:
                f.write("has_relevant_changes=true\n")
            return

        print(f"Found {len(changed_files)} changed files in total")

        # Check each changed file
        has_relevant_changes = False

        for file in changed_files:
            filename = file.filename
            print(f"Changed file: {filename}")

            # Check if the file matches any specific file
            if any(filename == specific_file for specific_file in specific_files):
                print(f"File {filename} matched specific file check")
                has_relevant_changes = True
                break

            # Check if the file matches any directory path
            if any(directory in filename for directory in directory_paths):
                print(f"File {filename} matched directory path check")
                has_relevant_changes = True
                break

            # Check if the file matches any pattern
            for pattern in patterns:
                if pattern_matches(filename, pattern):
                    print(f"File {filename} matched pattern {pattern}")
                    has_relevant_changes = True
                    break

            if has_relevant_changes:
                break

        print(f"Has relevant changes matching pattern: {has_relevant_changes}")

        # Write output
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"has_relevant_changes={str(has_relevant_changes).lower()}\n")

    except Exception as e:
        print(f"Error occurred while checking files: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
