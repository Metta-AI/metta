import os
import re

import requests


def add_comment_to_asana_task(task_url, comment_text, access_token):
    """
    Add a comment to an Asana task

    Args:
        task_id (str): The Asana task ID
        comment_text (str): The comment text to add
        access_token (str): Your Asana Personal Access Token

    Returns:
        dict: Response from Asana API
    """
    url = f"{task_url}/stories"

    headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}

    payload = {"data": {"text": comment_text}}

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error adding comment: {e}")
        return None


if __name__ == "__main__":
    # Inputs from the Action
    task_url = os.getenv("INPUT_TASK_URL")

    task_match = re.search(r"/task/(\d+)", task_url)
    if task_match:
        task_url = task_match.group(1)
    else:
        raise ValueError("Could not find task ID in URL")

    state = os.getenv("INPUT_STATE")
    author = os.getenv("INPUT_AUTHOR")
    text = os.getenv("INPUT_COMMENT")
    asana_token = os.getenv("INPUT_ASANA_TOKEN")

    comment_text = f"{author}\n{state}\n{text}"
    add_comment_to_asana_task(task_url, comment_text, asana_token)
