import requests
from pr_gh_to_asana import extract_asana_gid_from_url

""" just keeping this around awhile in case we want to push comments to Asana as a single comment with
the timeline in it, rather than individual comments
"""

# --- DEAD CODE: Commented reference for future use ---
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


def is_embedded_review_comment(key: str) -> bool:
    """
    Return True if the comment text is an embedded review comment (starts with 'GitHub Review Timeline:').
    """
    return key.lstrip().startswith("GitHub Review Timeline:")
