import os
import sys
import requests


def search_asana_tasks(github_url, project_id, github_url_field_id, asana_token):
    """Search for existing Asana tasks with the given GitHub URL in the specified project."""
    url = "https://app.asana.com/api/1.0/tasks"
    headers = {
        "Authorization": f"Bearer {asana_token}",
        "Content-Type": "application/json",
    }
    params = {"project": project_id, "opt_fields": "permalink_url,custom_fields"}

    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        print(f"Asana API Error: {response.status_code} - {response.text}")
        sys.exit(1)

    tasks = response.json()["data"]

    # Look for tasks with a GitHub URL custom field matching our URL
    for task in tasks:
        if "custom_fields" in task:
            for field in task["custom_fields"]:
                if field.get("gid") == github_url_field_id and field.get("text_value") == github_url:
                    return task["permalink_url"]

    return None


def create_asana_task(title, description, project_id, github_url, github_url_field_id, asana_token):
    """Create a new Asana task with the GitHub URL field populated."""
    url = "https://app.asana.com/api/1.0/tasks"
    headers = {
        "Authorization": f"Bearer {asana_token}",
        "Content-Type": "application/json",
    }

    payload = {
        "name": title,
        "notes": description,
        "projects": [project_id],
    }

    # Add the GitHub URL custom field
    if github_url_field_id:
        payload["custom_fields"] = {github_url_field_id: github_url}

    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 201:
        return response.json()["data"]["permalink_url"]
    else:
        print(f"Asana API Error: {response.status_code} - {response.text}")
        sys.exit(1)


def ensure_asana_task_exists(title, description, project_id, github_url, github_url_field_id, asana_token):
    """Ensure an Asana task exists with the given GitHub URL. Return existing or create new."""
    # First, search for existing task with this GitHub URL
    existing_task_url = search_asana_tasks(github_url, project_id, github_url_field_id, asana_token)
    if existing_task_url:
        print(f"Found existing Asana task: {existing_task_url}")
        return existing_task_url

    # If no existing task found, create a new one
    print(f"No existing task found with GitHub URL: {github_url}")
    new_task_url = create_asana_task(title, description, project_id, github_url, github_url_field_id, asana_token)
    print(f"Created new Asana task: {new_task_url}")
    return new_task_url


if __name__ == "__main__":
    # Inputs from the Action
    title = os.getenv("INPUT_TITLE")
    description = os.getenv("INPUT_DESCRIPTION")
    project_id = os.getenv("INPUT_PROJECT_ID")
    asana_token = os.getenv("INPUT_ASANA_TOKEN")
    github_url = os.getenv("INPUT_GITHUB_URL")
    github_url_field_id = os.getenv("INPUT_GITHUB_URL_FIELD_ID")

    # Ensure task exists and output URL
    task_url = ensure_asana_task_exists(title, description, project_id, github_url, github_url_field_id, asana_token)
    print(f"::set-output name=task_url::{task_url}")
