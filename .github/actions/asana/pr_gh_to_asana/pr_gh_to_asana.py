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
    """Extract the Asana task GID from a task URL, or raise an exception if not found."""
    match = re.search(r"https://app\.asana\.com/\d+/\d+/(\d+)", task_url)
    if match:
        return match.group(1)
    raise ValueError(f"Invalid Asana task URL format: {task_url}")


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
            }
            for comment in comments
        ]

    except requests.RequestException as e:
        print(f"Error fetching Asana task comments: {e}")
        raise
    except (KeyError, ValueError) as e:
        print(f"Error parsing response data: {e}")
        raise


def getenv(key: str) -> str:
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


if __name__ == "__main__":
    import traceback

    try:
        # Inputs from the Action
        project_id = getenv("INPUT_PROJECT_ID")
        workspace_id = getenv("INPUT_WORKSPACE_ID")
        asana_token = getenv("INPUT_ASANA_TOKEN")
        github_url = getenv("INPUT_GITHUB_URL")
        github_url_field_id = getenv("INPUT_GITHUB_URL_FIELD_ID")
        gh_login_field_id = getenv("INPUT_GH_LOGIN_FIELD_ID")
        asana_email_field_id = getenv("INPUT_ASANA_EMAIL_FIELD_ID")
        roster_project_id = getenv("INPUT_ROSTER_PROJECT_ID")
        pr_author_field_id = getenv("INPUT_PR_AUTHOR_FIELD_ID")
        asana_attachment_secret = getenv("INPUT_ASANA_ATTACHMENT_SECRET")
        pr_number = getenv("INPUT_PR_NUMBER")
        github_repo = getenv("INPUT_GITHUB_REPO")
        github_token = getenv("INPUT_GITHUB_TOKEN")

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
                "state": r["state"],
                "data": r,
            }
            for r in retrieved_reviews
            if r.get("state") in ["APPROVED", "CHANGES_REQUESTED"]
        ]
        review_requested = [
            {
                "type": "review_requested",
                "timestamp": e["created_at"],
                "user": e["actor"]["login"],
                "state": None,
                "data": e,
            }
            for e in retrieved_timeline
            if e.get("event") == "review_requested"
        ]
        events = sorted(reviews + review_requested, key=lambda x: x["timestamp"])
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
        asana_owner_is_assignee = not (last_event) or last_event["type"] == "review_requested"
        asana_owner = assignee if asana_owner_is_assignee else author

        # github allows multiple assignees. Asana doesn't. So we'll just use the first one.
        asana_assignee = github_login_to_asana_email.get(assignees[0]) if assignees else None
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
        print(f"comments: {comments}")

        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"task_url={task_url}\n")
    except Exception:
        traceback.print_exc()
        raise
