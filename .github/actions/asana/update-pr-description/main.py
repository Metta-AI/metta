import os
import sys

import requests


def update_pr_description(repo, pr_number, task_url, token):
    url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
    }

    # Fetch the current PR description
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"GitHub API Error: {response.status_code} - {response.text}")
        sys.exit(1)
    current_body = response.json().get("body", "")

    # Update the description
    new_body = f"{current_body}\n\nLinked Asana Task: {task_url}"
    payload = {"body": new_body}
    response = requests.patch(url, json=payload, headers=headers)
    if response.status_code != 200:
        print(f"GitHub API Error: {response.status_code} - {response.text}")
        sys.exit(1)


if __name__ == "__main__":
    # Inputs from the Action
    repo = os.getenv("INPUT_REPO")
    pr_number = os.getenv("INPUT_PR_NUMBER")
    task_url = os.getenv("INPUT_TASK_URL")
    token = os.getenv("INPUT_TOKEN")

    # Update the PR description
    update_pr_description(repo, pr_number, task_url, token)
