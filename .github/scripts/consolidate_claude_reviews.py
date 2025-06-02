#!/usr/bin/env python3
"""
Consolidate multiple Claude review artifacts into a single review.

This script downloads artifacts from individual Claude review runs and
combines them into a unified review structure.
"""

import json
import os
import sys
import zipfile
from io import BytesIO
from typing import Any, Dict, Optional

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


def consolidate_reviews(token: str, repo: str, run_id: int) -> Dict[str, Any]:
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

    return consolidated


def main():
    """Main entry point."""
    # Get environment variables
    token = os.environ.get("GITHUB_TOKEN")
    repo = os.environ.get("GITHUB_REPOSITORY")
    run_id = int(os.environ.get("GITHUB_RUN_ID", "0"))

    if not all([token, repo, run_id]):
        print("Missing required environment variables")
        sys.exit(1)

    # Consolidate reviews
    consolidated = consolidate_reviews(token, repo, run_id)

    # Save consolidated review
    with open("consolidated-review.json", "w") as f:
        json.dump(consolidated, f, indent=2)

    print("Consolidated review saved to consolidated-review.json")


if __name__ == "__main__":
    main()
