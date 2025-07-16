#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "requests>=2.31.0",
# ]
# ///

import os
import re
import sys
from typing import Generator

import requests

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


def validate_asana_task_url(
    task_url: str, project_id: str, github_url: str, github_url_field_id: str, asana_token: str
) -> dict | None:
    """Validate that an Asana task URL exists, belongs to the specified project, and has the expected GitHub URL."""
    # Extract task GID from URL
    # URL format: https://app.asana.com/0/123456789/123456789
    match = re.search(r"https://app\.asana\.com/\d+/\d+/(\d+)", task_url)
    if not match:
        print(f"Invalid Asana task URL format: {task_url}")
        return None

    task_gid = match.group(1)

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
            print(f"Task {task_url} does not belong to project {project_id}")
            return None

        # Check if the GitHub URL custom field matches the expected GitHub URL
        custom_fields = task_data.get("custom_fields", [])
        task_github_url = None
        for field in custom_fields:
            if field.get("gid") == github_url_field_id:
                task_github_url = field.get("text_value")
                break

        if task_github_url != github_url:
            print(f"Task {task_url} has GitHub URL '{task_github_url}' but expected '{github_url}'")
            return None

        print(f"Validated Asana task: {task_url}")
        return task_data

    except Exception as e:
        print(f"Error validating Asana task {task_url}: {e}")
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
        print(f"Asana API Error: {response.status_code} - {response.text}")
        sys.exit(1)

    data = response.json()
    tasks = data["data"]
    print(f"Found {len(tasks)} tasks matching GitHub URL: {github_url}")

    # Return the first matching task (should be the most recently created due to sorting)
    if tasks:
        task = tasks[0]
        return task

    print("No tasks found with matching GitHub URL")
    return None


def _get_task_custom_fields_from_project(
    project_id: str,
    asana_token: str,
) -> Generator[dict[str, str], None, None]:
    """Get the custom fields for all tasks in the given project."""
    url = f"https://app.asana.com/api/1.0/projects/{project_id}/tasks"
    headers = {
        "Authorization": f"Bearer {asana_token}",
        "Content-Type": "application/json",
    }
    params = {
        "opt_fields": "custom_fields",
        "limit": 100,
    }
    while True:
        response = requests.get(url, headers=headers, params=params, timeout=30)
        if response.status_code != 200:
            print(f"Asana API Error: {response.status_code} - {response.text}")
            sys.exit(1)
        data = response.json()
        tasks = data["data"]
        for task in tasks:
            yield task
        if not data.get("next_page"):
            break
        params["offset"] = data["next_page"]["offset"]


def get_asana_users_by_github_logins(
    github_logins: set[str],
    roster_project_id: str,
    gh_login_field_id: str,
    asana_email_field_id: str,
    asana_token: str,
) -> dict[str, str]:
    """Get the Asana user IDs for the given GitHub logins."""
    github_login_to_asana_email = {}
    # Paginate through all tasks in the roster project, since we can't search for multiple github logins at once
    for task in _get_task_custom_fields_from_project(roster_project_id, asana_token):
        custom_fields = task.get("custom_fields") or {}
        gh_login = None
        asana_email = None
        for field in custom_fields:
            if field.get("gid") == gh_login_field_id:
                gh_login = field.get("text_value")
                if gh_login not in github_logins:
                    # This isn't the user we're looking for.
                    break
            if field.get("gid") == asana_email_field_id:
                asana_email = field.get("text_value")
            if gh_login and asana_email:
                github_login_to_asana_email[gh_login] = asana_email
                break
        if len(github_login_to_asana_email) == len(github_logins):
            break
    return github_login_to_asana_email


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
        print(f"Asana API Error: {response.status_code} - {response.text}")
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
        print(f"Asana API Error updating task: {response.status_code} - {response.text}")


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


def find_and_validate_task(
    project_id: str,
    github_url: str,
    github_url_field_id: str,
    asana_token: str,
    workspace_id: str | None = None,
) -> dict | None:
    """Find a task that matches the project and GitHub URL requirements."""
    # First try to find by searching Asana tasks with the GitHub URL
    if workspace_id:
        existing_task = search_asana_tasks(github_url, project_id, workspace_id, github_url_field_id, asana_token)
        if existing_task:
            print(f"Found existing Asana task via search: {existing_task['permalink_url']}")
            return existing_task

    print("No existing task found with matching GitHub URL")
    return None


def ensure_asana_task_exists(
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
        existing_task = find_and_validate_task(project_id, github_url, github_url_field_id, asana_token, workspace_id)

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
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Asana API Error: {response.status_code} - {response.text}")
        return None


if __name__ == "__main__":
    # Inputs from the Action
    title = os.getenv("INPUT_TITLE")
    description = os.getenv("INPUT_DESCRIPTION")
    pr_state = os.getenv("INPUT_PR_STATE")
    author = os.getenv("INPUT_AUTHOR")
    assignees = os.getenv("INPUT_ASSIGNEES").split(",") if os.getenv("INPUT_ASSIGNEES") else []
    reviewers = os.getenv("INPUT_REVIEWERS").split(",") if os.getenv("INPUT_REVIEWERS") else []
    project_id = os.getenv("INPUT_PROJECT_ID")
    workspace_id = os.getenv("INPUT_WORKSPACE_ID")
    asana_token = os.getenv("INPUT_ASANA_TOKEN")
    github_url = os.getenv("INPUT_GITHUB_URL")
    github_url_field_id = os.getenv("INPUT_GITHUB_URL_FIELD_ID")
    gh_login_field_id = os.getenv("INPUT_GH_LOGIN_FIELD_ID")
    asana_email_field_id = os.getenv("INPUT_ASANA_EMAIL_FIELD_ID")
    roster_project_id = os.getenv("INPUT_ROSTER_PROJECT_ID")
    pr_author_field_id = os.getenv("INPUT_PR_AUTHOR_FIELD_ID")
    asana_attachment_secret = os.getenv("INPUT_ASANA_ATTACHMENT_SECRET")

    github_logins = set(assignees + reviewers + [author])
    github_login_to_asana_email = get_asana_users_by_github_logins(
        github_logins,
        roster_project_id,
        gh_login_field_id,
        asana_email_field_id,
        asana_token,
    )
    print(f"github_login_to_asana_email: {github_login_to_asana_email}")

    # github allows multiple assignees. Asana doesn't. So we'll just use the first one.
    asana_assignee = github_login_to_asana_email.get(assignees[0]) if assignees else None
    asana_collaborators = [
        github_login_to_asana_email[login] for login in github_logins if login in github_login_to_asana_email
    ]

    # Get the author's Asana ID for the custom field
    pr_author_asana = github_login_to_asana_email.get(author)

    task_completed = pr_state == "closed"

    # Ensure task exists and output URL
    task_url = ensure_asana_task_exists(
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

    with open(os.environ["GITHUB_OUTPUT"], "a") as f:
        f.write(f"task_url={task_url}\n")
