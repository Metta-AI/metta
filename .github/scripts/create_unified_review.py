#!/usr/bin/env python3
"""
Create a unified GitHub review from consolidated Claude analysis.

This script takes the consolidated review JSON and creates a single
GitHub PR review with all suggestions as inline comments.
"""

import json
import os
import sys
from typing import Any, Dict, List

from github import Github


def create_review_comment(suggestion: Dict[str, Any]) -> Dict[str, Any]:
    """Create a review comment from a suggestion."""
    comment = {
        "path": suggestion["file"],
        "line": suggestion.get("end_line", suggestion["start_line"]),
        "side": suggestion.get("side", "RIGHT"),
        "body": f"**{suggestion.get('severity', 'minor')}**: {suggestion['reason']}\n\n```suggestion\n{suggestion['suggested_code']}\n```",
    }

    # Add start_line for multi-line comments
    if suggestion.get("start_line") and suggestion.get("end_line"):
        if suggestion["start_line"] != suggestion["end_line"]:
            comment["start_line"] = suggestion["start_line"]
            comment["start_side"] = suggestion.get("side", "RIGHT")

    return comment


def create_compliment_comment(compliment: Dict[str, Any]) -> Dict[str, Any]:
    """Create a review comment from a compliment."""
    return {
        "path": compliment["file"],
        "line": compliment["line"],
        "side": "RIGHT",
        "body": f"✨ {compliment['comment']}",
    }


def build_review_body(analysis: Dict[str, Any], skipped_suggestions: List[str]) -> str:
    """Build the review body markdown."""
    sections = []

    # Header
    sections.append("## 🤖 Claude Unified Code Review")
    sections.append("")
    sections.append(analysis["review_summary"])
    sections.append("")

    # Overview
    sections.append("### Overview")
    sections.append(f"- **Total suggestions**: {len(analysis['suggestions'])}")
    sections.append(f"- **Total compliments**: {len(analysis['compliments'])}")
    sections.append(f"- **Review types run**: {len(analysis['review_types'])}")
    sections.append("")

    # Review type sections
    emoji_map = {"readme": "📝", "comments": "💬", "types": "🏷️", "einops": "🔄"}

    for review_type, data in analysis["review_types"].items():
        if data["suggestion_count"] > 0 or data["compliment_count"] > 0:
            emoji = emoji_map.get(review_type, "📋")
            title = review_type.capitalize()

            sections.append(f"### {emoji} {title} Review")
            sections.append(f"- **Status**: {data['status'].replace('_', ' ').lower()}")
            sections.append(f"- **Suggestions**: {data['suggestion_count']}")
            if data["compliment_count"] > 0:
                sections.append(f"- **Compliments**: {data['compliment_count']}")
            sections.append("")

    # Quick summary
    if analysis.get("tldr"):
        sections.append("### Quick Summary")
        for item in analysis["tldr"]:
            sections.append(f"- {item}")
        sections.append("")

    # Skipped suggestions
    if skipped_suggestions:
        sections.append("### ⚠️ Skipped Suggestions")
        sections.append("The following suggestions could not be added as inline comments:")
        for skip in skipped_suggestions:
            sections.append(f"- {skip}")
        sections.append("")

    # Footer
    sections.append("---")
    if analysis["suggestions"]:
        sections.append(
            '*Review the inline comments above for specific suggestions. Each suggestion can be committed directly using GitHub\'s "Commit suggestion" button.*'
        )
    else:
        sections.append("*No specific code changes suggested in this review.*")

    return "\n".join(sections)


def create_unified_review(token: str, repo: str, pr_number: int, analysis_file: str):
    """Create a unified GitHub review from the consolidated analysis."""
    # Load consolidated analysis
    with open(analysis_file, "r") as f:
        analysis = json.load(f)

    # Skip if no issues found
    if not analysis.get("suggestions") and not analysis.get("compliments"):
        print("No issues found - skipping review creation")
        return

    # Connect to GitHub
    g = Github(token)
    repo_obj = g.get_repo(repo)
    pr = repo_obj.get_pull(pr_number)

    # Get PR files
    pr_files = list(pr.get_files())
    pr_filenames = {f.filename for f in pr_files}

    print(f"PR has {len(pr_files)} files")

    # Build review comments
    comments = []
    skipped_suggestions = []

    # Process suggestions
    for suggestion in analysis.get("suggestions", []):
        if suggestion["file"] not in pr_filenames:
            skipped_suggestions.append(f"{suggestion['file']} - file not in PR diff")
            continue

        try:
            comment = create_review_comment(suggestion)
            comments.append(comment)
        except Exception as e:
            print(f"Error creating comment for {suggestion['file']}: {e}")
            skipped_suggestions.append(f"{suggestion['file']} - error creating comment")

    # Process compliments
    for compliment in analysis.get("compliments", []):
        if compliment["file"] in pr_filenames:
            try:
                comment = create_compliment_comment(compliment)
                comments.append(comment)
            except Exception as e:
                print(f"Error creating compliment for {compliment['file']}: {e}")

    print(f"Created {len(comments)} review comments")

    # Build review body
    review_body = build_review_body(analysis, skipped_suggestions)

    # Create the review
    try:
        # GitHub API doesn't accept comments parameter directly in PyGithub
        # We need to use the raw API
        review_event = analysis.get("review_status", "COMMENT")
        if review_event == "NONE":
            review_event = "COMMENT"

        # Create review with comments
        review_data = {"body": review_body, "event": review_event, "comments": comments}

        # Use the raw GitHub API
        headers = {"Authorization": f"token {token}"}
        import requests

        response = requests.post(
            f"https://api.github.com/repos/{repo}/pulls/{pr_number}/reviews", headers=headers, json=review_data
        )

        if response.status_code in [200, 201]:
            review_id = response.json()["id"]
            print(f"✅ Created unified review #{review_id} with {len(comments)} inline comments")
        else:
            print(f"❌ Failed to create review: {response.status_code}")
            print(response.json())

            # Fallback to simple comment
            pr.create_issue_comment(review_body)
            print("✅ Created fallback comment")

    except Exception as e:
        print(f"Error creating review: {e}")
        # Fallback to simple comment
        try:
            pr.create_issue_comment(review_body)
            print("✅ Created fallback comment")
        except Exception as e2:
            print(f"Failed to create fallback comment: {e2}")
            sys.exit(1)


def main():
    """Main entry point."""
    # Get environment variables
    token = os.environ.get("GITHUB_TOKEN")
    repo = os.environ.get("GITHUB_REPOSITORY")
    pr_number = int(os.environ.get("PR_NUMBER", "0"))

    if not all([token, repo, pr_number]):
        print("Missing required environment variables")
        sys.exit(1)

    # Create unified review
    create_unified_review(token, repo, pr_number, "consolidated-review.json")


if __name__ == "__main__":
    main()
