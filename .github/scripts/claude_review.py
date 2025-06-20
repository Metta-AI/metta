#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "PyGithub>=2.1.1",
#   "requests>=2.31.0",
# ]
# ///
"""
Benchmarking utilities for GitHub Actions workflows.
Provides memory and time monitoring for subprocess execution.
"""

import json
import os
import re
import sys
import zipfile
from io import BytesIO
from typing import Any, Tuple

import requests
from github import Github


def download_artifact(token: str, repo: str, artifact_id: int) -> bytes | None:
    """Download an artifact from GitHub."""
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
    url = f"https://api.github.com/repos/{repo}/actions/artifacts/{artifact_id}/zip"

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.content
    else:
        print(f"Failed to download artifact {artifact_id}: {response.status_code}")
        return None


def extract_json_from_artifact(artifact_data: bytes) -> dict[str, Any] | None:
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


def consolidate_reviews(token: str, repo: str, run_id: int) -> Tuple[dict[str, Any], bool]:
    """Consolidate all review artifacts into a single structure."""
    # Initialize consolidated review
    consolidated = {
        "review_summary": "Consolidated review from multiple Claude analyzers",
        "review_status": "COMMENT",
        "suggestions": [],
        "tldr": [],
        "review_types": {},
    }

    # Review types to check
    review_types = ["readme", "comments", "types", "einops"]

    total_suggestions = 0
    has_any_issues = False
    reviews_with_issues = []

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
            }
            continue

        review_data = extract_json_from_artifact(artifact_data)
        if not review_data:
            print(f"No JSON found in {review_type} artifact")
            consolidated["review_types"][review_type] = {
                "summary": "No issues found",
                "status": "NONE",
                "suggestion_count": 0,
            }
            continue

        # Extract data from this review
        suggestions = review_data.get("suggestions", [])
        suggestion_count = len(suggestions)

        total_suggestions += suggestion_count

        if suggestion_count > 0:
            has_any_issues = True
            reviews_with_issues.append(review_type)

        # Add to consolidated review
        consolidated["review_types"][review_type] = {
            "summary": review_data.get("review_summary", ""),
            "status": review_data.get("review_status", "COMMENT"),
            "suggestion_count": suggestion_count,
        }

        consolidated["suggestions"].extend(suggestions)

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
            f.write(f"final_status={final_status}\n")

    print("✅ Consolidation complete:")
    print(f"   Total suggestions: {total_suggestions}")
    print(f"   Has issues: {has_any_issues}")
    print(f"   Final status: {final_status}")

    return consolidated, has_any_issues


def build_future_suggestions_comment(suggestions_for_future: list[dict[str, Any]]) -> str:
    """Build a separate comment for suggestions on unchanged code."""
    sections = []

    # Header
    sections.append("## 💡 Claude Found Additional Issues in Unchanged Code")
    sections.append("")
    sections.append(
        f"Claude's review found {len(suggestions_for_future)} suggestion(s) in code that wasn't modified in this PR."
    )
    sections.append("These can be addressed in a future PR or by using Claude assistant in open-PR mode.")
    sections.append("")

    # Group by review type and file
    by_type_and_file = {}

    for suggestion in suggestions_for_future:
        # Extract review type from the suggestion if available
        review_type = "general"
        full_suggestion = suggestion.get("suggestion", {})

        # Try to determine review type from the reason
        reason = full_suggestion.get("reason", "")
        if "comment" in reason.lower() or "restate" in reason.lower():
            review_type = "comments"
        elif "type" in reason.lower() or "annotation" in reason.lower():
            review_type = "types"
        elif "einops" in reason.lower():
            review_type = "einops"
        elif "readme" in reason.lower():
            review_type = "readme"

        if review_type not in by_type_and_file:
            by_type_and_file[review_type] = {}

        file = suggestion["file"]
        if file not in by_type_and_file[review_type]:
            by_type_and_file[review_type][file] = []

        by_type_and_file[review_type][file].append(suggestion)

    # Create sections for each review type
    emoji_map = {"readme": "📝", "comments": "💬", "types": "🏷️", "einops": "🔄", "general": "📋"}

    for review_type, files in by_type_and_file.items():
        emoji = emoji_map.get(review_type, "📋")
        title = review_type.capitalize()

        sections.append(f"### {emoji} {title} Suggestions")
        sections.append("")

        for file, suggestions in files.items():
            sections.append(f"**{file}**")
            for s in suggestions:
                full_suggestion = s.get("suggestion", {})
                line_info = f"Line {s.get('line', full_suggestion.get('start_line', 'unknown'))}"
                reason = full_suggestion.get("reason", "Issue found")
                severity = full_suggestion.get("severity", "minor")

                sections.append(f"- {line_info} ({severity}): {reason}")

                # Include the suggested code if it's substantial
                if full_suggestion.get("suggested_code"):
                    sections.append("  ```suggestion")
                    sections.append(f"  {full_suggestion['suggested_code']}")
                    sections.append("  ```")

            sections.append("")

    # Footer
    sections.append("---")
    sections.append(
        "*These suggestions are for code not changed in this PR. Consider creating a follow-up PR to address them.*"
    )

    return "\n".join(sections)


def create_review_comment(suggestion: dict[str, Any]) -> dict[str, Any]:
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


def build_review_body(analysis: dict[str, Any], skipped_suggestions: list[str]) -> str:
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
    sections.append(f"- **Review types run**: {len(analysis['review_types'])}")
    sections.append("")

    # Review type sections
    emoji_map = {"readme": "📝", "comments": "💬", "types": "🏷️", "einops": "🔄"}

    for review_type, data in analysis["review_types"].items():
        if data["suggestion_count"] > 0:
            emoji = emoji_map.get(review_type, "📋")
            title = review_type.capitalize()

            sections.append(f"### {emoji} {title} Review")
            sections.append(f"- **Status**: {data['status'].replace('_', ' ').lower()}")
            sections.append(f"- **Suggestions**: {data['suggestion_count']}")
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


def create_unified_review(token: str, repo: str, pr_number: int, analysis: dict[str, Any]) -> None:
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

    # Build a mapping of files to their patches and content
    file_patches = {}
    file_diff_lines = {}

    for pr_file in pr_files:
        file_patches[pr_file.filename] = pr_file.patch or ""
        file_diff_lines[pr_file.filename] = {}

        if pr_file.patch:
            lines = pr_file.patch.split("\n")
            current_line = 0

            for _, line in enumerate(lines):
                if line.startswith("@@"):
                    # Parse the line number from the diff header
                    match = re.search(r"\+(\d+)", line)
                    if match:
                        current_line = int(match.group(1)) - 1
                elif line.startswith("+") and not line.startswith("+++"):
                    current_line += 1
                    # Store the actual content and its line number in the new file
                    content = line[1:]  # Remove the '+' prefix
                    file_diff_lines[pr_file.filename][current_line] = content
                elif not line.startswith("-"):
                    current_line += 1

    # Build review comments
    comments = []
    skipped_suggestions = []
    suggestions_for_future = []

    # Process suggestions
    for suggestion in analysis.get("suggestions", []):
        if suggestion["file"] not in pr_filenames:
            skipped_suggestions.append(
                {"file": suggestion["file"], "reason": "File not in PR", "suggestion": suggestion}
            )
            continue

        # Try to find the original code in the diff
        original_code = suggestion.get("original_code", "").strip()
        if not original_code:
            skipped_suggestions.append(
                {"file": suggestion["file"], "reason": "No original code to match", "suggestion": suggestion}
            )
            continue

        # Search for the original code in the file's diff lines
        found_line = None
        diff_lines = file_diff_lines.get(suggestion["file"], {})

        # Try exact match first
        for line_num, content in diff_lines.items():
            if original_code in content or content.strip() == original_code:
                found_line = line_num
                break

        # If not found, try fuzzy matching (remove extra whitespace)
        if not found_line:
            normalized_original = " ".join(original_code.split())
            for line_num, content in diff_lines.items():
                normalized_content = " ".join(content.split())
                if normalized_original in normalized_content:
                    found_line = line_num
                    break

        if not found_line:
            # This suggestion is for code not changed in this PR
            suggestions_for_future.append(
                {
                    "file": suggestion["file"],
                    "line": suggestion.get("start_line", "unknown"),
                    "reason": suggestion["reason"],
                    "suggestion": suggestion,
                }
            )
            continue

        try:
            # Create comment with the found line
            comment = {
                "path": suggestion["file"],
                "line": found_line,
                "side": "RIGHT",
                "body": (
                    f"**{suggestion.get('severity', 'minor')}**: {suggestion['reason']}\n\n"
                    f"```suggestion\n{suggestion['suggested_code']}\n```"
                ),
            }
            comments.append(comment)
        except Exception as e:
            print(f"Error creating comment for {suggestion['file']}: {e}")
            skipped_suggestions.append(
                {"file": suggestion["file"], "reason": f"Error: {str(e)}", "suggestion": suggestion}
            )

    print(f"   Created {len(comments)} review comments")
    print(f"   Found {len(suggestions_for_future)} suggestions for unchanged code")
    if skipped_suggestions:
        print(f"   Skipped {len(skipped_suggestions)} suggestions due to errors")

    # Build review body
    review_body = build_review_body(analysis, [s["file"] + " - " + s["reason"] for s in skipped_suggestions])

    # Create the review
    try:
        # Determine review event
        review_event = analysis.get("review_status", "COMMENT")
        if review_event == "NONE":
            review_event = "COMMENT"

        review_created = False

        if comments:
            # Only create a review if we have valid inline comments
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
                review_created = True
            else:
                print(f"\n❌ Failed to create review: {response.status_code}")
                print(response.json())

                # Fallback to simple comment
                pr.create_issue_comment(review_body)
                print("✅ Created fallback comment")
                review_created = True
        else:
            # No valid inline comments, just create a regular comment
            pr.create_issue_comment(review_body)
            print("\n✅ Created summary comment (no inline comments were applicable to the diff)")
            review_created = True

        # Create a separate comment for future suggestions if we have any
        if review_created and suggestions_for_future:
            try:
                future_comment_body = build_future_suggestions_comment(suggestions_for_future)
                pr.create_issue_comment(future_comment_body)
                print(
                    f"\n📋 Created separate comment with {len(suggestions_for_future)} suggestions for unchanged code"
                )
            except Exception as e:
                print(f"\n⚠️  Failed to create future suggestions comment: {e}")

    except Exception as e:
        print(f"\nError creating review: {e}")
        # Fallback to simple comment
        try:
            pr.create_issue_comment(review_body)
            print("✅ Created fallback comment")

            # Still try to create future suggestions comment
            if suggestions_for_future:
                try:
                    future_comment_body = build_future_suggestions_comment(suggestions_for_future)
                    pr.create_issue_comment(future_comment_body)
                    print(f"\n📋 Created separate comment with {len(suggestions_for_future)} suggestions")
                except Exception as e2:
                    print(f"\n⚠️  Failed to create future suggestions comment: {e2}")

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
