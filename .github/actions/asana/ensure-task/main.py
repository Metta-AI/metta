#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "requests>=2.31.0",
# ]
# ///

import os
import sys
from typing import Generator

import requests


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

    # Add the GitHub URL custom field
    if github_url_field_id:
        payload["data"]["custom_fields"] = {github_url_field_id: github_url}

    # Set completion status if PR is closed
    if completed:
        payload["data"]["completed"] = True

    print(f"Creating task with payload: {payload}")

    response = requests.post(url, json=payload, headers=headers, timeout=30)
    if response.status_code == 201:
        return response.json()["data"]["permalink_url"]
    else:
        print(f"Asana API Error: {response.status_code} - {response.text}")
        sys.exit(1)


def update_asana_task(
    task_gid: str, title: str, description: str, completed: bool, assignee: str | None, asana_token: str
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

    print(f"Updating task {task_gid} with payload: {payload}")

    response = requests.put(url, json=payload, headers=headers, timeout=30)
    if response.status_code == 200:
        print(f"Successfully updated task {task_gid}")
    else:
        print(f"Asana API Error updating task: {response.status_code} - {response.text}")


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
) -> str:
    """Ensure an Asana task exists with the given GitHub URL. Return existing or create new."""
    # First, search for existing task with this GitHub URL
    existing_task = search_asana_tasks(github_url, project_id, workspace_id, github_url_field_id, asana_token)
    if existing_task:
        print(f"Found existing Asana task: {existing_task['permalink_url']}")

        # Check if the task needs updates using the data from search
        if existing_task:
            current_title = existing_task.get("name") or ""
            current_notes = existing_task.get("notes") or ""
            current_completed = existing_task.get("completed") or False
            current_assignee = (existing_task.get("assignee") or {}).get("email") or ""
            # TODO: update followers -- probably only add.

            # Update title and description if needed
            if (
                current_title != title
                or current_notes != description
                or current_completed != task_completed
                or current_assignee != asana_assignee
            ):
                print("Task needs updates")
                update_asana_task(
                    existing_task["gid"],
                    title,
                    description,
                    task_completed,
                    asana_assignee,
                    asana_token,
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
    )
    print(f"Created new Asana task: {new_task_url}")
    return new_task_url


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
    )

    with open(os.environ["GITHUB_OUTPUT"], "a") as f:
        f.write(f"task_url={task_url}\n")
