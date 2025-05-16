import os
import json
from asana import Client

# Load environment variables
GITHUB_EVENT_PATH = os.getenv("GITHUB_EVENT_PATH")
ASANA_ACCESS_TOKEN = os.getenv("ASANA_ACCESS_TOKEN")
ASANA_WORKSPACE_ID = os.getenv("ASANA_WORKSPACE_ID")
ASANA_PROJECT_ID = os.getenv("ASANA_PROJECT_ID")

# Initialize Asana client
client = Client.access_token(ASANA_ACCESS_TOKEN)

def parse_github_event(event_path):
    """Parse the GitHub event payload."""
    with open(event_path, 'r') as f:
        return json.load(f)

def create_asana_task(pr_data):
    """Create a new Asana task for the PR."""
    task = client.tasks.create_in_workspace(
        ASANA_WORKSPACE_ID,
        {
            "name": f"PR: {pr_data['title']}",
            "notes": pr_data['body'],
            "projects": [ASANA_PROJECT_ID],
            "external": {"data": pr_data['html_url']}
        }
    )
    return task

def update_asana_task(task_id, updates):
    """Update an existing Asana task."""
    client.tasks.update_task(task_id, updates)

def sync_reviewers_and_assignee(task_id, pr_data):
    """Sync GitHub reviewers and assignee to Asana followers and assignee."""
    followers = [user['email'] for user in pr_data.get('reviewers', [])]
    assignee = pr_data.get('assignee', {}).get('email')

    if followers:
        client.tasks.add_followers(task_id, {"followers": followers})
    if assignee:
        client.tasks.update_task(task_id, {"assignee": assignee})

def sync_comments(task_id, comments):
    """Sync GitHub comments to Asana task."""
    for comment in comments:
        client.tasks.add_comment(task_id, {"text": comment['body']})

def close_asana_task(task_id):
    """Mark the Asana task as complete."""
    client.tasks.update_task(task_id, {"completed": True})

def main():
    # Parse GitHub event
    event = parse_github_event(GITHUB_EVENT_PATH)
    action = event['action']
    pr_data = event.get('pull_request')
    
    if not pr_data:
        print("No pull request data found, exiting.")
        return
    
    # Get existing Asana task (if any)
    task_id = None  # Retrieve this from Asana if you have a mapping system

    if action in ['opened', 'edited']:
        if not task_id:
            task = create_asana_task(pr_data)
            task_id = task['gid']
        else:
            update_asana_task(task_id, {"name": f"PR: {pr_data['title']}"})
        sync_reviewers_and_assignee(task_id, pr_data)

    elif action == 'closed':
        close_asana_task(task_id)

    elif action == 'commented':
        comments = event.get('comment', [])
        sync_comments(task_id, comments)

if __name__ == "__main__":
    main()
