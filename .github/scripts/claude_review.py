#!/usr/bin/env python3
"""
Consolidate multiple Claude review artifacts and create a unified GitHub review.

This script:
1. Downloads artifacts from individual Claude review runs
2. Consolidates them into a unified review structure
3. Creates a GitHub PR review with all suggestions as inline comments
"""

import json
import os
import sys
import zipfile
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import requests
from github import Github


def download_artifact(token: str, repo: str, artifact_id: int) -> Optional[bytes]:
    """Download an artifact from GitHub."""
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
    url = f"https://api.github.com/repos/{repo}/actions/artifacts/{artifact_id}/zip"

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.content
    else:
        print(f"Failed to download artifact {artifact_id}: {response.status_code}")
        return None


def extract_json_from_artifact(artifact_data: bytes) -> Optional[Dict[str, Any]]:
    """Extract claude-review-analysis.json from artifact zip."""
    try:
        with zipfile.ZipFile(BytesIO(artifact_data)) as zip_file:
            for name in zip_file.namelist():
                if name == "claude-review-analysis.json":
                    with zip_file.open(name) as f:
                        return json.load(f)
    except Exception as e:
        print(f"Error extracting JSON from artifact: {e}")
    return None


def consolidate_reviews(token: str, repo: str, run_id: int) -> Tuple[Dict[str, Any], bool]:
    """Consolidate all review artifacts into a single structure."""
    # Initialize consolidated review
    consolidated = {
        "review_summary": "Consolidated review from multiple Claude analyzers",
        "review_status": "COMMENT",
        "suggestions": [],
        "compliments": [],
        "tldr": [],
        "review_types": {},
    }

    # Review types to check
    review_types = ["readme", "comments", "types", "einops"]

    # Counters
    total_suggestions = 0
    total_compliments = 0
    has_any_issues = False
    reviews_with_issues = []

    # Connect to GitHub API
    g = Github(token)
    repo_obj = g.get_repo(repo)
    run = repo_obj.get_workflow_run(run_id)

    # Get all artifacts for this run
    artifacts = list(run.get_artifacts())
    print(f"Found {len(artifacts)} artifacts")

    # Process each review type
    for review_type in review_types:
        artifact_name = f"claude-review-{review_type}-analysis"
        artifact = next((a for a in artifacts if a.name == artifact_name), None)

        if not artifact:
            print(f"No artifact found for {review_type} review - assuming no issues")
            consolidated["review_types"][review_type] = {
                "summary": "No issues found",
                "status": "NONE",
                "suggestion_count": 0,
                "compliment_count": 0,
            }
            continue

        print(f"Processing {review_type} review (artifact: {artifact.id})")

        # Download and extract artifact
        artifact_data = download_artifact(token, repo, artifact.id)
        if not artifact_data:
            consolidated["review_types"][review_type] = {
                "summary": "Error downloading artifact",
                "status": "NONE",
                "suggestion_count": 0,
                "compliment_count": 0,
            }
            continue

        review_data = extract_json_from_artifact(artifact_data)
        if not review_data:
            print(f"No JSON found in {review_type} artifact")
            consolidated["review_types"][review_type] = {
                "summary": "No issues found",
                "status": "NONE",
                "suggestion_count": 0,
                "compliment_count": 0,
            }
            continue

        # Extract data from this review
        suggestions = review_data.get("suggestions", [])
        compliments = review_data.get("compliments", [])
        suggestion_count = len(suggestions)
        compliment_count = len(compliments)

        total_suggestions += suggestion_count
        total_compliments += compliment_count

        if suggestion_count > 0:
            has_any_issues = True
            reviews_with_issues.append(review_type)

        # Add to consolidated review
        consolidated["review_types"][review_type] = {
            "summary": review_data.get("review_summary", ""),
            "status": review_data.get("review_status", "COMMENT"),
            "suggestion_count": suggestion_count,
            "compliment_count": compliment_count,
        }

        # Merge suggestions and compliments
        consolidated["suggestions"].extend(suggestions)
        consolidated["compliments"].extend(compliments)

        # Add prefixed TLDR items
        tldr_items = review_data.get("tldr", [])
        if tldr_items:
            consolidated["tldr"].extend([f"{review_type}: {item}" for item in tldr_items])

    # Update final status
    final_status = "NONE"
    if has_any_issues:
        # Check if any review requested changes
        has_changes_requested = any(rt["status"] == "CHANGES_REQUESTED" for rt in consolidated["review_types"].values())
        final_status = "CHANGES_REQUESTED" if has_changes_requested else "COMMENT"

    # Update the review summary
    consolidated["review_status"] = final_status
    if total_suggestions == 0:
        consolidated["review_summary"] = "All Claude review checks passed. No issues found across all review types."
    else:
        consolidated["review_summary"] = (
            f"Found {total_suggestions} suggestion(s) across reviews: {', '.join(reviews_with_issues)}"
        )

    # Set outputs for GitHub Actions
    if os.getenv("GITHUB_OUTPUT"):
        with open(os.getenv("GITHUB_OUTPUT"), "a") as f:
            f.write(f"has_any_issues={'true' if has_any_issues else 'false'}\n")
            f.write(f"total_suggestions={total_suggestions}\n")
            f.write(f"total_compliments={total_compliments}\n")
            f.write(f"final_status={final_status}\n")

    print("✅ Consolidation complete:")
    print(f"   Total suggestions: {total_suggestions}")
    print(f"   Total compliments: {total_compliments}")
    print(f"   Has issues: {has_any_issues}")
    print(f"   Final status: {final_status}")

    return consolidated, has_any_issues


def create_review_comment(suggestion: Dict[str, Any]) -> Dict[str, Any]:
    """Create a review comment from a suggestion."""
    comment = {
        "path": suggestion["file"],
        "line": suggestion.get("end_line", suggestion["start_line"]),
        "side": suggestion.get("side", "RIGHT"),
        "body": (
            f"**{suggestion.get('severity', 'minor')}**: {suggestion['reason']}\n\n"
            f"```suggestion\n{suggestion['suggested_code']}\n```"
        ),
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
            "*Review the inline comments above for specific suggestions. "
            'Each suggestion can be committed directly using GitHub\'s "Commit suggestion" button.*'
        )
    else:
        sections.append("*No specific code changes suggested in this review.*")

    return "\n".join(sections)


def create_unified_review(token: str, repo: str, pr_number: int, analysis: Dict[str, Any]) -> None:
    """Create a unified GitHub review from the consolidated analysis."""
    # Connect to GitHub
    g = Github(token)
    repo_obj = g.get_repo(repo)
    pr = repo_obj.get_pull(pr_number)

    # Get PR files and latest commit
    pr_files = list(pr.get_files())
    pr_filenames = {f.filename for f in pr_files}

    # Get the latest commit SHA from the PR
    latest_commit_sha = pr.head.sha

    print(f"\n📝 Creating GitHub review for PR #{pr_number}")
    print(f"   PR has {len(pr_files)} files")
    print(f"   Latest commit: {latest_commit_sha[:7]}")

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

    print(f"   Created {len(comments)} review comments")
    if skipped_suggestions:
        print(f"   Skipped {len(skipped_suggestions)} suggestions")

    # Build review body
    review_body = build_review_body(analysis, skipped_suggestions)

    # Create the review
    try:
        # Determine review event
        review_event = analysis.get("review_status", "COMMENT")
        if review_event == "NONE":
            review_event = "COMMENT"

        # Create review with comments using GitHub API
        review_data = {
            "commit_id": latest_commit_sha,
            "body": review_body,
            "event": review_event,
            "comments": comments,
        }

        headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}

        response = requests.post(
            f"https://api.github.com/repos/{repo}/pulls/{pr_number}/reviews", headers=headers, json=review_data
        )

        if response.status_code in [200, 201]:
            review_id = response.json()["id"]
            print(f"\n✅ Created unified review #{review_id} with {len(comments)} inline comments")
        else:
            print(f"\n❌ Failed to create review: {response.status_code}")
            print(response.json())

            # Fallback to simple comment
            pr.create_issue_comment(review_body)
            print("✅ Created fallback comment")

    except Exception as e:
        print(f"\nError creating review: {e}")
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
    run_id = int(os.environ.get("GITHUB_RUN_ID", "0"))
    pr_number = int(os.environ.get("PR_NUMBER", "0"))

    if not all([token, repo, run_id, pr_number]):
        print("Missing required environment variables")
        print(f"  GITHUB_TOKEN: {'✓' if token else '✗'}")
        print(f"  GITHUB_REPOSITORY: {'✓' if repo else '✗'} ({repo if repo else 'missing'})")
        print(f"  GITHUB_RUN_ID: {'✓' if run_id else '✗'} ({run_id if run_id else 'missing'})")
        print(f"  PR_NUMBER: {'✓' if pr_number else '✗'} ({pr_number if pr_number else 'missing'})")
        sys.exit(1)

    print(f"🔍 Consolidating Claude reviews for run {run_id}")
    print(f"   Repository: {repo}")
    print(f"   PR: #{pr_number}")
    print("")

    # Step 1: Consolidate reviews
    consolidated, has_issues = consolidate_reviews(token, repo, run_id)

    # Save consolidated review for debugging
    with open("consolidated-review.json", "w") as f:
        json.dump(consolidated, f, indent=2)
    print("\n💾 Saved consolidated review to consolidated-review.json")

    # Step 2: Create GitHub review if there are issues
    if has_issues:
        create_unified_review(token, repo, pr_number, consolidated)
    else:
        print("\n✨ No issues found - skipping review creation")


if __name__ == "__main__":
    main()
