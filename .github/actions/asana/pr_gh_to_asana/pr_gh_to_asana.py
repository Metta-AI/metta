#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "requests>=2.31.0",
# ]
# ///

import json
import os
import re
import sys
from datetime import datetime
from typing import Any, Dict, List

import requests
from github_asana_mapping import GithubAsanaMapping

ASANA_GITHUB_ATTACHMENT_ACTION_URL = "https://github.integrations.asana.plus/custom/v1/actions/widget"


def extract_asana_urls_from_description(description: str) -> list[str]:
    """Extract Asana task URLs from the description text."""
    if not description:
        return []

    # Pattern to match Asana task URLs in the format used by update-pr-description
    # Matches: [Asana Task](https://app.asana.com/0/123456789/123456789)
    asana_pattern = r"\[Asana Task\]\((https://app\.asana\.com/\d+/\d+/\d+(?:\?[^\s\)]*)?)\)"

    urls = re.findall(asana_pattern, description)
    return urls


def extract_asana_gid_from_url(task_url: str) -> str:
    # Try Format 1: https://app.asana.com/0/project_id/task_id
    match = re.search(r"https://app\.asana\.com/\d+/\d+/(\d+)(?:/|$)", task_url)
    if match:
        return match.group(1)

    # Try Format 2: https://app.asana.com/1/workspace_id/project/project_id/task/task_id
    match = re.search(r"https://app\.asana\.com/\d+/\d+/project/\d+/task/(\d+)(?:/|$)", task_url)
    if match:
        return match.group(1)

    raise Exception(f"Could not extract task ID from URL: {task_url}")


def validate_asana_task_url(
    task_url: str, project_id: str, github_url: str, github_url_field_id: str, asana_token: str
) -> dict | None:
    """Validate that an Asana task URL exists, belongs to the specified project, and has the expected GitHub URL."""
    print("[validate_asana_task_url] Called with:")
    print(f"  task_url: {task_url}")
    print(f"  project_id: {project_id}")
    print(f"  github_url: {github_url}")
    print(f"  github_url_field_id: {github_url_field_id}")
    print(f"  asana_token: {'set' if asana_token else 'not set'}")

    # Extract task GID from URL
    task_gid = extract_asana_gid_from_url(task_url)
    if not task_gid:
        print(f"[validate_asana_task_url] Invalid Asana task URL format: {task_url}")
        return None

    print(f"[validate_asana_task_url] Extracted task_gid: {task_gid}")

    # Fetch task details from Asana API
    url = f"https://app.asana.com/api/1.0/tasks/{task_gid}"
    headers = {
        "Authorization": f"Bearer {asana_token}",
        "Content-Type": "application/json",
    }
    params = {
        "opt_fields": "permalink_url,custom_fields,name,notes,modified_at,completed,"
        "assignee.email,followers.email,projects.gid",
    }

    try:
        response = requests.get(url, headers=headers, params=params, timeout=30)
        if response.status_code != 200:
            print(f"Asana API Error validating task {task_url}: {response.status_code} - {response.text}")
            return None

        task_data = response.json()["data"]

        # Check if task belongs to the specified project
        task_projects = [project["gid"] for project in task_data.get("projects", [])]
        if project_id not in task_projects:
            return None

        # Check if the GitHub URL custom field matches the expected GitHub URL
        custom_fields = task_data.get("custom_fields", [])
        task_github_url = None
        for field in custom_fields:
            if field.get("gid") == github_url_field_id:
                task_github_url = field.get("text_value")
                print(f"Found github_url_field_id: {github_url_field_id}, value: {task_github_url}")
                break

        if task_github_url != github_url:
            print(f"Task {task_url} has GitHub URL '{task_github_url}' but expected '{github_url}'")
            return None

        return task_data

    except Exception as e:
        print(f"[validate_asana_task_url] Error validating Asana task {task_url}: {e}")
        return None


def search_asana_tasks(
    github_url: str,
    project_id: str,
    workspace_id: str,
    github_url_field_id: str,
    asana_token: str,
) -> dict | None:
    """Search for existing Asana tasks with the given GitHub URL in the specified project."""
    # Use workspace search endpoint which supports custom field filtering
    url = f"https://app.asana.com/api/1.0/workspaces/{workspace_id}/tasks/search"
    headers = {
        "Authorization": f"Bearer {asana_token}",
        "Content-Type": "application/json",
    }

    # Use workspace search with custom field filtering
    params = {
        "opt_fields": "permalink_url,custom_fields,name,notes,modified_at,completed,assignee.email,followers.email",
        "limit": 100,  # Maximum allowed by Asana API
        "sort_by": "created_at",
        "projects.any": project_id,  # Filter to specific project
        f"custom_fields.{github_url_field_id}.value": github_url,  # Filter by custom field
    }

    response = requests.get(url, headers=headers, params=params, timeout=30)
    if response.status_code != 200:
        print(f"Asana API Error (search_asana_tasks): {response.status_code} - {response.text}")
        sys.exit(1)

    data = response.json()
    tasks = data["data"]
    print(f"Found {len(tasks)} tasks matching GitHub URL: {github_url}")

    # Return the first matching task (should be the most recently created due to sorting)
    if tasks:
        task = tasks[0]
        print(f"Found existing Asana task via search: {task['permalink_url']}")
        return task

    print("No tasks found with matching GitHub URL")
    return None


def create_asana_task(
    title: str,
    description: str,
    completed: bool,
    assignee: str | None,
    collaborators: list[str],
    project_id: str,
    workspace_id: str,
    github_url: str,
    github_url_field_id: str,
    asana_token: str,
    pr_author_field_id: str,
    pr_author_asana: str | None,
    asana_attachment_secret: str,
) -> str:
    """Create a new Asana task with the GitHub URL field populated."""
    url = "https://app.asana.com/api/1.0/tasks"
    headers = {
        "Authorization": f"Bearer {asana_token}",
        "Content-Type": "application/json",
    }

    payload = {
        "data": {
            "name": title,
            "notes": description,
            "workspace": workspace_id,
            "projects": [project_id],
            "followers": collaborators,
        }
    }
    if assignee:
        payload["data"]["assignee"] = assignee

    custom_fields = {}
    # Add the GitHub URL custom field
    custom_fields[github_url_field_id] = github_url

    # Add the PR author custom field
    custom_fields[pr_author_field_id] = pr_author_asana

    payload["data"]["custom_fields"] = custom_fields

    # Set completion status if PR is closed
    if completed:
        payload["data"]["completed"] = True

    print(f"Creating task with payload: {payload}")

    response = requests.post(url, json=payload, headers=headers, timeout=30)
    if response.status_code == 201:
        task_url = response.json()["data"]["permalink_url"]
        # For the most part, create_asana_task should do the same work as update_asana_task. This
        # is a specific exception, since it's an extra call and this should be effectively immutable.
        ensure_github_url_in_asana_task(asana_attachment_secret, project_id, task_url, title, github_url)
        return task_url
    else:
        print(f"Asana API Error (create_asana_task): {response.status_code} - {response.text}")
        sys.exit(1)


def update_asana_task(
    task_gid: str,
    title: str,
    description: str,
    completed: bool,
    assignee: str | None,
    asana_token: str,
    github_url: str,
    github_url_field_id: str,
    pr_author_field_id: str,
    pr_author_asana: str | None,
):
    """Update an existing Asana task with new title and description."""
    url = f"https://app.asana.com/api/1.0/tasks/{task_gid}"
    headers = {
        "Authorization": f"Bearer {asana_token}",
        "Content-Type": "application/json",
    }

    payload = {"data": {"name": title, "notes": description, "completed": completed}}

    # Add assignee if provided
    if assignee:
        payload["data"]["assignee"] = assignee

    # Add custom fields if provided
    custom_fields = {}
    custom_fields[github_url_field_id] = github_url
    custom_fields[pr_author_field_id] = pr_author_asana

    if custom_fields:
        payload["data"]["custom_fields"] = custom_fields

    print(f"Updating task {task_gid} with payload: {payload}")

    response = requests.put(url, json=payload, headers=headers, timeout=30)
    if response.status_code == 200:
        print(f"Successfully updated task {task_gid}")
    else:
        print(f"Asana API Error (update_asana_task): {response.status_code} - {response.text}")


def update_task_if_needed(
    task_data: dict,
    title: str,
    description: str,
    task_completed: bool,
    asana_assignee: str | None,
    asana_token: str,
    github_url: str,
    github_url_field_id: str,
    pr_author_field_id: str,
    pr_author_asana: str | None,
) -> None:
    """Update a task if its current data differs from the provided data."""
    current_title = task_data.get("name") or ""
    current_notes = task_data.get("notes") or ""
    current_completed = task_data.get("completed") or False
    current_assignee = (task_data.get("assignee") or {}).get("email") or ""

    # Update title and description if needed
    if (
        current_title != title
        or current_notes != description
        or current_completed != task_completed
        or current_assignee != asana_assignee
    ):
        print("Task needs updates")
        update_asana_task(
            task_data["gid"],
            title,
            description,
            task_completed,
            asana_assignee,
            asana_token,
            github_url,
            github_url_field_id,
            pr_author_field_id,
            pr_author_asana,
        )


def ensure_asana_task(
    title: str,
    description: str,
    task_completed: bool,
    asana_assignee: str | None,
    asana_collaborators: list[str],
    project_id: str,
    workspace_id: str,
    github_url: str,
    github_url_field_id: str,
    asana_token: str,
    pr_author_field_id: str,
    pr_author_asana: str | None,
    asana_attachment_secret: str,
) -> str:
    """Ensure an Asana task exists with the given GitHub URL. Return existing or create new."""

    existing_task = None

    # First, check if there are existing Asana URLs in the description. Doing this (vs using Asana search)
    # avoids a race condition in Asana search indexing, and makes duplicate tasks less likely.
    existing_asana_urls = extract_asana_urls_from_description(description)
    if existing_asana_urls:
        print(f"Found {len(existing_asana_urls)} Asana URLs in description: {existing_asana_urls}")

        # Validate each URL to see if it's a valid task in our project
        for asana_url in existing_asana_urls:
            validated_task = validate_asana_task_url(
                asana_url, project_id, github_url, github_url_field_id, asana_token
            )
            if validated_task:
                print(f"Using existing Asana task from description: {asana_url}")
                existing_task = validated_task
                break

        print("None of the existing Asana URLs in description are valid for this project")

    # If no valid existing URLs found in description, search for existing task with this GitHub URL. If we need
    # to do this, the github description probably became malformed.
    if not existing_task:
        existing_task = search_asana_tasks(
            project_id=project_id,
            github_url=github_url,
            workspace_id=workspace_id,
            github_url_field_id=github_url_field_id,
            asana_token=asana_token,
        )

    if existing_task:
        update_task_if_needed(
            existing_task,
            title,
            description,
            task_completed,
            asana_assignee,
            asana_token,
            github_url,
            github_url_field_id,
            pr_author_field_id,
            pr_author_asana,
        )
        return existing_task["permalink_url"]

    # If no existing task found, create a new one
    print(f"No existing task found with GitHub URL: {github_url}")
    new_task_url = create_asana_task(
        title,
        description,
        task_completed,
        asana_assignee,
        asana_collaborators,
        project_id,
        workspace_id,
        github_url,
        github_url_field_id,
        asana_token,
        pr_author_field_id,
        pr_author_asana,
        asana_attachment_secret,
    )
    print(f"Created new Asana task: {new_task_url}")
    return new_task_url


def ensure_github_url_in_asana_task(
    asana_attachment_secret: str,
    project_id: str,
    task_url: str,
    title: str,
    github_url: str,
) -> dict | None:
    """Ensure the GitHub URL is in the Asana task.

    Asana provides this via https://github.com/Asana/create-app-attachment-github-action, but their
    workflow only runs in limited contexts. In particular, we don't trust it to pick up the task url
    from the description when we've added it within the same workflow. So we'll just do it manually.
    """
    github_url_number = github_url.split("pull/")[-1]
    if not github_url_number.isdigit():
        print(f"Invalid GitHub URL: {github_url}")
        return None

    headers = {
        "Authorization": f"Bearer {asana_attachment_secret}",
        "Content-Type": "application/json",
    }

    payload = {
        "allowedProjects": [project_id],
        "blockedProjects": [],
        # This is used in the created attachment story.
        "pullRequestName": title,
        # This we fake, since we want Asana to find the right task in the description.
        "pullRequestDescription": task_url,
        "pullRequestNumber": int(github_url_number),
        "pullRequestURL": github_url,
    }
    response = requests.post(ASANA_GITHUB_ATTACHMENT_ACTION_URL, json=payload, headers=headers, timeout=30)
    if response.status_code == 201:
        return response.json()
    else:
        print(f"Asana API Error (ensure_github_url_in_asana_task): {response.status_code} - {response.text}")
        return None


def get_pull_request_from_github(repo, pr_number, github_token):
    """
    Get pull request details from GitHub API

    Args:
        owner (str): Repository owner
        repo (str): Repository name
        pr_number (int): Pull request number
        github_token (str): GitHub personal access token

    Returns:
        dict: Pull request data, including comments under the 'comments' key
    """
    url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}"

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {github_token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()  # Raises an exception for bad status codes

    d = response.json()

    # Fetch PR comments (issue comments)
    comments_url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments"
    comments_response = requests.get(comments_url, headers=headers)
    if comments_response.status_code == 200:
        d["retrieved_comments"] = comments_response.json()

    else:
        print(f"Failed to fetch PR comments: {comments_response.status_code} - {comments_response.text}")
        d["retrieved_comments"] = []

    # Fetch PR reviews
    reviews_url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}/reviews"
    reviews_response = requests.get(reviews_url, headers=headers)
    if reviews_response.status_code == 200:
        d["retrieved_reviews"] = reviews_response.json()
    else:
        print(f"Failed to fetch PR reviews: {reviews_response.status_code} - {reviews_response.text}")
        d["retrieved_reviews"] = []

    # Fetch PR timeline
    timeline_url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/timeline"
    timeline_response = requests.get(timeline_url, headers=headers)
    if timeline_response.status_code == 200:
        d["retrieved_timeline"] = timeline_response.json()
    else:
        print(f"Failed to fetch PR timeline: {timeline_response.status_code} - {timeline_response.text}")
        d["retrieved_timeline"] = []

    print(f"Pull request retrieved from GitHub: {json.dumps(d, indent=2)}")
    return d


def get_asana_task_comments(asana_token: str, task_id: str) -> List[Dict[str, Any]]:
    """
    Fetches comments for an Asana task

    Args:
        asana_token (str): Asana Personal Access Token
        task_id (str): Asana task ID

    Returns:
        List[Dict]: List of comment dictionaries

    Raises:
        requests.RequestException: If API request fails
        ValueError: If response data is invalid
    """
    try:
        url = f"https://app.asana.com/api/1.0/tasks/{task_id}/stories"
        headers = {"Authorization": f"Bearer {asana_token}", "Content-Type": "application/json"}
        params = {
            "opt_fields": "text,html_text,created_by.name,created_by.email,created_at,type,resource_subtype,is_pinned"
        }

        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()

        data = response.json()

        # Filter for actual comments only (exclude system stories)
        comments = [
            story
            for story in data["data"]
            if story.get("type") == "comment" or story.get("resource_subtype") == "comment_added"
        ]

        # Transform to more usable format
        return [
            {
                "id": comment.get("gid"),
                "text": comment.get("text", ""),
                "html_text": comment.get("html_text", ""),
                "author": {
                    "name": comment.get("created_by", {}).get("name", "Unknown"),
                    "email": comment.get("created_by", {}).get("email"),
                },
                "created_at": datetime.fromisoformat(comment.get("created_at", "").replace("Z", "+00:00")),
                "is_pinned": comment.get("is_pinned", False),
                "review_id": extract_github_review_id(comment.get("text", "")),
            }
            for comment in comments
        ]

    except requests.RequestException as e:
        print(f"Error fetching Asana task comments: {e}")
        raise
    except (KeyError, ValueError) as e:
        print(f"Error parsing response data: {e}")
        raise


def format_github_review_body_for_asana(review_body, github_user, review_state, review_id, github_timestamp):
    """
    Format GitHub review body comment for Asana

    Args:
        review_body: The review's body text (markdown string)
        github_user: GitHub username of the reviewer
        review_state: Review state (APPROVED, CHANGES_REQUESTED, COMMENTED)
        review_id: GitHub review ID number
        github_timestamp: When the review was submitted
    """
    # Choose emoji based on review state
    emoji = {"APPROVED": "✅", "CHANGES_REQUESTED": "❌", "COMMENTED": "💬"}.get(review_state, "📝")

    # Format header with review ID
    header = f"<strong>{emoji} GitHub Review from @{github_user} (Review #{review_id})</strong><br/>"
    header += f"<em>Status: {review_state.replace('_', ' ').title()} | {github_timestamp}</em><br/><br/>"

    header = f"<strong>GitHub Review from @{github_user} ({review_id})</strong>"
    header += f"\n<em>Status: {review_state.replace('_', ' ').title()} | {github_timestamp}</em><br/><br/>"

    # Convert basic markdown in body
    formatted_body = convert_basic_markdown(review_body) if review_body else "(No comment)"
    # formatted_body = review_body if review_body else "(No comment)"

    return "<body>" + header + formatted_body + "</body>"


def extract_github_review_id(asana_comment_text):
    """
    Extract GitHub review ID from Asana comment text

    Args:
        asana_comment_text: The text content of an Asana comment

    Returns:
        int: GitHub review ID if found, None otherwise
    """
    import re

    if not asana_comment_text:
        return None

    # Look for pattern: "Review #123456789" or "(Review #123456789)"
    # This matches the format we created in format_github_review_body_for_asana
    pattern = r"Review #(\d+)"

    match = re.search(pattern, asana_comment_text)

    if match:
        return int(match.group(1))

    return None


def convert_basic_markdown(text):
    """Convert basic markdown to Asana HTML"""
    import re

    # Bold: **text** -> <strong>text</strong>
    text = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", text)

    # Italic: *text* -> <em>text</em>
    text = re.sub(r"\*(.*?)\*", r"<em>\1</em>", text)

    # Inline code: `code` -> <code>code</code>
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)

    # Line breaks
    text = text.replace("\n", "<br/>")

    return text


def synchronize_comments_in_asana_as_single_block(
    asana_token: str, task_url: str, sync_comments: list[dict], events_str: str
) -> None:
    """
    Synchronize review comments in Asana. If sync_comments is empty, add a comment with events_str and return.
    Otherwise, update the first comment to events_str if needed, and delete any additional comments.
    sync_comments is the set of comments already pulled from Asana that might match what we have pushed.
    Args:
        asana_token (str): Asana Personal Access Token
        task_url (str): Asana task URL (permalink)
        sync_comments (list[dict]): List of current review comments (id, text) already pulled from Asana
        events_str (str): The string representing the current review events
    """
    if not (events_str):
        return

    task_gid = extract_asana_gid_from_url(task_url)
    api_url = f"https://app.asana.com/api/1.0/tasks/{task_gid}"
    headers = {"Authorization": f"Bearer {asana_token}", "Content-Type": "application/json"}

    comment_body = f"GitHub Review Timeline:\n{events_str}" if events_str.strip() else ""
    print(f"new comment body: {comment_body}")

    if not sync_comments:
        # No comments, add new
        if comment_body:
            url = f"{api_url}/stories"
            payload = {"data": {"text": comment_body}}
            try:
                response = requests.post(url, headers=headers, json=payload)
                response.raise_for_status()
                print(f"Added review events comment to Asana: {comment_body}")
            except requests.exceptions.RequestException as e:
                print(f"Error adding review events comment to Asana: {e}")
        return

    # There is at least one comment
    first_comment = sync_comments[0]
    if first_comment["text"] != comment_body:
        # Update the first comment to comment_body
        story_id = first_comment["id"]
        url = f"https://app.asana.com/api/1.0/stories/{story_id}"
        payload = {"data": {"text": comment_body}}
        try:
            response = requests.put(url, headers=headers, json=payload)
            if response.status_code == 200:
                print(f"Updated first Asana comment {story_id} to: {comment_body}")
            else:
                print(f"Failed to update Asana comment {story_id}: {response.status_code} - {response.text}")
        except requests.exceptions.RequestException as e:
            print(f"Error updating Asana comment {story_id}: {e}")

    # Delete any additional comments
    for comment in sync_comments[1:]:
        story_id = comment["id"]
        del_url = f"https://app.asana.com/api/1.0/stories/{story_id}"
        try:
            response = requests.delete(del_url, headers=headers)
            if response.status_code == 200:
                print(f"Deleted duplicate Asana comment: {story_id}")
            else:
                print(f"Failed to delete Asana comment {story_id}: {response.status_code} - {response.text}")
        except requests.exceptions.RequestException as e:
            print(f"Error deleting Asana comment {story_id}: {e}")


def synchronize_comments_in_asana_as_multiple_blocks(
    asana_token: str, task_url: str, asana_comments_with_links: list[dict], comments_from_github: list[dict]
) -> None:
    """
    Synchronize review comments in Asana as multiple blocks.

    Args:
        asana_token (str): Asana Personal Access Token
        task_url (str): Asana task URL (permalink)
        asana_comments_with_links (list[dict]): List of current Asana comments that have review links
        comments_from_github (list[dict]): List of review comments from GitHub
    """
    print("[synchronize_comments_in_asana_as_multiple_blocks] Starting with:")
    print(f"  asana_comments_with_links: {len(asana_comments_with_links)} comments")
    print(f"  comments_from_github: {len(comments_from_github)} reviews")

    if not comments_from_github:
        print("[synchronize_comments_in_asana_as_multiple_blocks] No GitHub comments to process")
        return

    task_gid = extract_asana_gid_from_url(task_url)
    api_url = f"https://app.asana.com/api/1.0/tasks/{task_gid}"
    headers = {"Authorization": f"Bearer {asana_token}", "Content-Type": "application/json"}

    # Create a map of review IDs to existing Asana comments
    existing_comments_by_review_id = {
        comment["review_id"]: comment for comment in asana_comments_with_links if comment["review_id"] is not None
    }
    print(f"[s] Existing comments by review ID: {list(existing_comments_by_review_id.keys())}")

    # Process each GitHub review
    for github_review in comments_from_github:
        review_id = github_review["id"]
        review_body = github_review.get("text", "")
        github_user = github_review["user"]
        review_state = github_review["action"]
        github_timestamp = github_review["timestamp"]

        print(f"[s] Processing review {review_id} from {github_user} ({review_state})")

        # Format the review for Asana
        formatted_comment = format_github_review_body_for_asana(
            review_body, github_user, review_state, review_id, github_timestamp
        )

        if review_id in existing_comments_by_review_id:
            print(f"[synchronize_comments_in_asana_as_multiple_blocks] Review {review_id} has existing Asana comment")
            # Update existing comment if content differs
            existing_comment = existing_comments_by_review_id[review_id]
            if existing_comment["text"] != formatted_comment:
                print(f"[s] Updating existing comment for review {review_id}")
                story_id = existing_comment["id"]
                url = f"https://app.asana.com/api/1.0/stories/{story_id}"
                payload = {"data": {"html_text": formatted_comment}}
                try:
                    print(payload)
                    response = requests.put(url, headers=headers, json=payload)
                    if response.status_code == 200:
                        print(f"Updated Asana comment {story_id} for review {review_id}")
                    else:
                        print(f"Failed to update Asana comment {story_id}: {response.status_code} - {response.text}")
                except requests.exceptions.RequestException as e:
                    print(f"Error updating Asana comment {story_id}: {e}")
            else:
                print(f"[synchronize_comments_in_asana_as_multiple_blocks] Review {review_id} comment is up to date")
        else:
            print(f"[s] Review {review_id} has no existing Asana comment")
            # Check if we should add this comment (don't add out of order)
            # Find the last existing comment that matches a GitHub review
            last_matching_review_id = None
            for existing_comment in asana_comments_with_links:
                if existing_comment["review_id"] is not None:
                    if existing_comment["review_id"] in {r["id"] for r in comments_from_github}:
                        last_matching_review_id = existing_comment["review_id"]

            # Check if this review comes before the last matching review
            should_add = True
            if last_matching_review_id is not None:
                # Find the position of the last matching review in the GitHub reviews list
                last_matching_index = -1
                for i, review in enumerate(comments_from_github):
                    if review["id"] == last_matching_review_id:
                        last_matching_index = i
                        break

                # Find the position of current review in the GitHub reviews list
                current_review_index = -1
                for i, review in enumerate(comments_from_github):
                    if review["id"] == review_id:
                        current_review_index = i
                        break

                # Only add if current review comes after the last matching review
                if current_review_index <= last_matching_index:
                    print(f"[s] Review {review_id} comes before or at last  review {last_matching_review_id}, skipping")
                    should_add = False

            if should_add:
                print(f"[s] Adding new comment for review {review_id}")
                # Create new comment
                url = f"{api_url}/stories"
                payload = {"data": {"html_text": formatted_comment, "type": "comment"}}
                print(payload)

                try:
                    response = requests.post(url, headers=headers, json=payload)
                    response.raise_for_status()
                    print(f"Added new Asana comment for review {review_id}")
                except requests.exceptions.RequestException as e:
                    print(f"Error adding Asana comment for review {review_id}: {e}")
                    print(payload)
            else:
                print(f"[s] Skipped adding comment for review {review_id} due to ordering constraint")

    # Remove any existing comments that don't correspond to current GitHub reviews
    # current_review_ids = {review["id"] for review in comments_from_github}
    # for comment in asana_comments_with_links:
    #     if comment["review_id"] is not None and comment["review_id"] not in current_review_ids:
    #         story_id = comment["id"]
    #         del_url = f"https://app.asana.com/api/1.0/stories/{story_id}"
    #         try:
    #             response = requests.delete(del_url, headers=headers)
    #             if response.status_code == 200:
    #                 print(f"Deleted outdated Asana comment: {story_id}")
    #             else:
    #                 print(f"Failed to delete Asana comment {story_id}: {response.status_code} - {response.text}")
    #         except requests.exceptions.RequestException as e:
    #             print(f"Error deleting Asana comment {story_id}: {e}")


def getenv_or_bust(key: str) -> str:
    """
    Get environment variable value, throwing exception if not found

    Args:
        key (str): Environment variable name

    Returns:
        str: Environment variable value (guaranteed non-None)

    Raises:
        Exception: If environment variable is not set or is empty
    """
    value = os.getenv(key)
    if value is None:
        raise Exception(f"Environment variable '{key}' is not set")

    # Optional: Also check for empty strings
    if value == "":
        raise Exception(f"Environment variable '{key}' is empty")

    return value


def is_embedded_review_comment(key: str) -> bool:
    """
    Return True if the comment text is an embedded review comment (starts with 'GitHub Review Timeline:').
    """
    return key.lstrip().startswith("GitHub Review Timeline:")


if __name__ == "__main__":
    import traceback

    try:
        # Inputs from the Action
        project_id = getenv_or_bust("INPUT_PROJECT_ID")
        workspace_id = getenv_or_bust("INPUT_WORKSPACE_ID")
        asana_token = getenv_or_bust("INPUT_ASANA_TOKEN")
        github_url = getenv_or_bust("INPUT_GITHUB_URL")
        github_url_field_id = getenv_or_bust("INPUT_GITHUB_URL_FIELD_ID")
        gh_login_field_id = getenv_or_bust("INPUT_GH_LOGIN_FIELD_ID")
        asana_email_field_id = getenv_or_bust("INPUT_ASANA_EMAIL_FIELD_ID")
        roster_project_id = getenv_or_bust("INPUT_ROSTER_PROJECT_ID")
        pr_author_field_id = getenv_or_bust("INPUT_PR_AUTHOR_FIELD_ID")
        asana_attachment_secret = getenv_or_bust("INPUT_ASANA_ATTACHMENT_SECRET")
        pr_number = getenv_or_bust("INPUT_PR_NUMBER")
        github_repo = getenv_or_bust("INPUT_GITHUB_REPO")
        github_token = getenv_or_bust("INPUT_GITHUB_TOKEN")

        pr = get_pull_request_from_github(github_repo, pr_number, github_token)
        description = pr.get("body", "") or ""
        title = pr.get("title", "") or ""
        author = (pr.get("user", "") or {}).get("login", "") or ""
        assignees = [assignee["login"] for assignee in pr.get("assignees", [])] or []

        # designated assignee is not the author and also not random
        assignee = next((a for a in sorted(assignees) if a != author), author)

        reviewers = [
            review["user"]["login"]
            for review in pr.get("retrieved_reviews", [])
            if review.get("user") and review["user"].get("login")
        ]
        commenters = [
            review["user"]["login"]
            for review in pr.get("retrieved_comments", [])
            if review.get("user") and review["user"].get("login")
        ]

        github_logins = set(assignees + reviewers + commenters + [author])
        mapping = GithubAsanaMapping(
            github_logins,
            roster_project_id,
            gh_login_field_id,
            asana_email_field_id,
            asana_token,
        )
        github_login_to_asana_email = mapping.github_login_to_asana_email

        retrieved_reviews = pr.get("retrieved_reviews", [])
        retrieved_timeline = pr.get("retrieved_timeline", [])
        reviews = [
            {
                "type": "review",
                "timestamp": r["submitted_at"],
                "user": r["user"]["login"],
                "text": ": " + r["body"],
                "action": r["state"].lower(),
                "id": r["id"],
            }
            for r in retrieved_reviews
            if r.get("state") in ["APPROVED", "CHANGES_REQUESTED"]
        ]
        review_requested = [
            {
                "type": "review_requested",
                "timestamp": e["created_at"],
                "user": e["actor"]["login"],
                "requested_reviewer": e["requested_reviewer"]["login"],
                "action": "re-requested",
                "text": f" review from {e['requested_reviewer']['login']}",
                "id": e["id"],
            }
            for e in retrieved_timeline
            if e.get("event") == "review_requested"
        ]
        events = sorted(reviews + review_requested, key=lambda x: x["timestamp"])

        seen_review_users = set()
        filtered_events = []
        for event in events:
            if event["type"] == "review":
                seen_review_users.add(event["user"])
                filtered_events.append(event)
            elif event["type"] == "review_requested" and event["requested_reviewer"] in seen_review_users:
                filtered_events.append(event)
        events = filtered_events

        last_event = events[-1] if events else None

        is_draft = pr.get("draft", False)
        is_open = (pr.get("state", "") or "") == "open"
        task_completed = not (is_open)

        """
        asana task assignment (asana owner) should be the PR author or PR assignee dep who is responsible at this time
         - as a story, we say that the PR author is doing the coding, the PR assignee is doing the admin behind the PR
         - we switch from author <=> assignee while the review process is ongoing
         -   specifically we look at the last event
         -      (an event is either a passing or failing review or a review_requested)
         -    if there is no event, there has been no review, so the assignee has work to do and is the asana owner
         -    if the last event is a review (whether passing or failing the PR), the PR author is the asana owner
         -    if the last event is a review_request (meaning the author requested re-review), goes back to the assignee
         - note that this is only for active PRs
         -   if the PR is closed or merged (ie not open), or is a draft, the PR author is the asana owner
         - simplifying this logic:
         -   asana owner is designee when last_event is None or review_requested
         - ideally we would want to incorporate mergeability
         -   if the PR cannot be synced because of a merge issue with the PR destination this would go to the PR author.
         -     however, this is not easy to implement because mergeability is computed async and there is no hook
        """
        print(f"event stream: {events}")
        print(f"last event: {last_event}")

        asana_owner_is_assignee = not (last_event) or last_event["type"] == "review_requested"
        asana_owner = assignee if asana_owner_is_assignee else author

        # github allows multiple assignees. Asana doesn't. So we'll just use the first one.
        asana_assignee = github_login_to_asana_email.get(asana_owner) if assignees else None
        asana_collaborators = [
            github_login_to_asana_email[login] for login in github_logins if login in github_login_to_asana_email
        ]

        # Get the author's Asana ID for the custom field
        pr_author_asana = github_login_to_asana_email.get(author)

        # for now
        print(f"assignees: {assignees}")
        print(f"author: {author}")
        print(f"reviewers: {reviewers}")
        print(f"commenters: {commenters}")
        print(f"github_logins: {github_logins}")
        print(f"pr_author_asana: {pr_author_asana}")
        print(f"asana_owner_is_assignee: {asana_owner_is_assignee}")
        print(f"asana_assignee: {asana_assignee}")

        # Ensure task exists and output URL
        task_url = ensure_asana_task(
            title,
            description,
            task_completed,
            asana_assignee,
            asana_collaborators,
            project_id,
            workspace_id,
            github_url,
            github_url_field_id,
            asana_token,
            pr_author_field_id,
            pr_author_asana,
            asana_attachment_secret,
        )

        comments = get_asana_task_comments(asana_token, extract_asana_gid_from_url(task_url))
        print(f"comments in asana: {comments}")

        """
        # Everything in one line per item

        sync_comments = list(
            [
                {"id": comment["id"], "text": comment["text"]}
                for comment in comments
                if comment["author"]["email"] == "stemaidaemon@gmail.com"
                and is_embedded_review_comment(comment["text"])
            ]
        )

        events_str = "\n".join(
            [f"At {item['timestamp']}, user {item['user']} {item['action']}: {item['text']}" for item in events]
        )

        synchronize_comments_in_asana_as_single_block(asana_token, task_url, sync_comments, events_str)
        """

        asana_comments_with_links = list([comment for comment in comments if comment["review_id"] is not None])
        comments_from_github = list([e for e in events if event["type"] == "review"])

        synchronize_comments_in_asana_as_multiple_blocks(
            asana_token, task_url, asana_comments_with_links, comments_from_github
        )

        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"task_url={task_url}\n")
    except Exception:
        traceback.print_exc()
        raise
