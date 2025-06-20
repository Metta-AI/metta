#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "requests>=2.31.0",
# ]
# ///

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
    # The body can be None -- we want to treat that as an empty string
    current_body = response.json().get("body") or ""

    # Update the description
    new_body = current_body
    if task_url not in current_body:
        new_body = f"{current_body}\n\n[Asana Task]({task_url})"
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

    print(f"Updating PR description for {repo} #{pr_number} with task {task_url}")

    # Update the PR description
    update_pr_description(repo, pr_number, task_url, token)
