import os
import sys

import requests


def search_asana_tasks(github_url, project_id, workspace_id, github_url_field_id, asana_token):
    """Search for existing Asana tasks with the given GitHub URL in the specified project."""
    # Use workspace search endpoint which supports custom field filtering
    url = f"https://app.asana.com/api/1.0/workspaces/{workspace_id}/tasks/search"
    headers = {
        "Authorization": f"Bearer {asana_token}",
        "Content-Type": "application/json",
    }

    # Use workspace search with custom field filtering
    params = {
        "opt_fields": "permalink_url,custom_fields,name,modified_at",
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
        print(f"Found existing Asana task: {task['permalink_url']}")
        return task["permalink_url"]

    print("No tasks found with matching GitHub URL")
    return None


def create_asana_task(title, description, project_id, workspace_id, github_url, github_url_field_id, asana_token):
    """Create a new Asana task with the GitHub URL field populated."""
    url = "https://app.asana.com/api/1.0/tasks"
    headers = {
        "Authorization": f"Bearer {asana_token}",
        "Content-Type": "application/json",
    }

    payload = {"data": {
        "name": title,
        "notes": description,
        "workspace": workspace_id,
        "projects": [project_id],
    }}

    # Add the GitHub URL custom field
    if github_url_field_id:
        payload["data"]["custom_fields"] = {github_url_field_id: github_url}

    print(f"Creating task with payload: {payload}")

    response = requests.post(url, json=payload, headers=headers, timeout=30)
    if response.status_code == 201:
        return response.json()["data"]["permalink_url"]
    else:
        print(f"Asana API Error: {response.status_code} - {response.text}")
        sys.exit(1)


def ensure_asana_task_exists(title, description, project_id, workspace_id, github_url, github_url_field_id, asana_token):
    """Ensure an Asana task exists with the given GitHub URL. Return existing or create new."""
    # First, search for existing task with this GitHub URL
    existing_task_url = search_asana_tasks(github_url, project_id, workspace_id, github_url_field_id, asana_token)
    if existing_task_url:
        print(f"Found existing Asana task: {existing_task_url}")
        return existing_task_url

    # If no existing task found, create a new one
    print(f"No existing task found with GitHub URL: {github_url}")
    new_task_url = create_asana_task(title, description, project_id, workspace_id, github_url, github_url_field_id, asana_token)
    print(f"Created new Asana task: {new_task_url}")
    return new_task_url


if __name__ == "__main__":
    # Inputs from the Action
    title = os.getenv("INPUT_TITLE")
    description = os.getenv("INPUT_DESCRIPTION")
    project_id = os.getenv("INPUT_PROJECT_ID")
    workspace_id = os.getenv("INPUT_WORKSPACE_ID")
    asana_token = os.getenv("INPUT_ASANA_TOKEN")
    github_url = os.getenv("INPUT_GITHUB_URL")
    github_url_field_id = os.getenv("INPUT_GITHUB_URL_FIELD_ID")

    # Debug: Print all inputs
    print(f"DEBUG - title: {title}")
    print(f"DEBUG - description: {description[:50] if description else None}...")
    print(f"DEBUG - project_id: {project_id}")
    print(f"DEBUG - workspace_id: {workspace_id}")
    print(f"DEBUG - asana_token: {asana_token[:10] if asana_token else None}...")
    print(f"DEBUG - github_url: {github_url}")
    print(f"DEBUG - github_url_field_id: {github_url_field_id}")

    # Ensure task exists and output URL
    task_url = ensure_asana_task_exists(title, description, project_id, workspace_id, github_url, github_url_field_id, asana_token)
    with open(os.environ["GITHUB_OUTPUT"], "a") as f:
        f.write(f"task_url={task_url}\n")
