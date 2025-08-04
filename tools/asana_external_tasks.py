#!/usr/bin/env python3
"""
Fetch and implement tasks from Asana's External Projects.
"""

import os
import json
import argparse
import requests
from typing import List, Dict, Optional
from datetime import datetime


class AsanaExternalProjects:
    """Handle fetching and managing Asana External Projects tasks."""
    
    def __init__(self, token: str, workspace_id: str = None, project_id: str = None):
        self.token = token
        self.workspace_id = workspace_id
        self.project_id = project_id
        self.base_url = "https://app.asana.com/api/1.0"
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
    
    def _make_request(self, endpoint: str, method: str = "GET", data: Dict = None) -> Dict:
        """Make an API request to Asana."""
        url = f"{self.base_url}/{endpoint}"
        
        try:
            if method == "GET":
                response = requests.get(url, headers=self.headers, timeout=30)
            elif method == "POST":
                response = requests.post(url, headers=self.headers, json=data, timeout=30)
            elif method == "PUT":
                response = requests.put(url, headers=self.headers, json=data, timeout=30)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error making request to {url}: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response content: {e.response.text}")
            raise
    
    def get_workspaces(self) -> List[Dict]:
        """Get all workspaces accessible by the user."""
        result = self._make_request("workspaces")
        return result.get("data", [])
    
    def get_projects(self, workspace_id: str = None) -> List[Dict]:
        """Get all projects in a workspace."""
        workspace_id = workspace_id or self.workspace_id
        if not workspace_id:
            raise ValueError("Workspace ID is required")
        
        result = self._make_request(f"workspaces/{workspace_id}/projects")
        return result.get("data", [])
    
    def find_external_project(self) -> Optional[str]:
        """Find the External Projects project."""
        projects = self.get_projects()
        
        for project in projects:
            if "external" in project["name"].lower():
                print(f"Found External Projects: {project['name']} (GID: {project['gid']})")
                return project["gid"]
        
        # If not found by name, list all projects and let user choose
        print("\nAvailable projects:")
        for i, project in enumerate(projects):
            print(f"{i + 1}. {project['name']} (GID: {project['gid']})")
        
        return None
    
    def get_tasks(self, project_id: str = None, completed_since: str = None) -> List[Dict]:
        """Get tasks from a project."""
        project_id = project_id or self.project_id
        if not project_id:
            raise ValueError("Project ID is required")
        
        params = []
        if completed_since:
            params.append(f"completed_since={completed_since}")
        
        query_string = "&".join(params) if params else ""
        endpoint = f"projects/{project_id}/tasks"
        if query_string:
            endpoint += f"?{query_string}"
        
        result = self._make_request(endpoint)
        tasks = result.get("data", [])
        
        # Get full task details for each task
        detailed_tasks = []
        for task in tasks:
            try:
                task_detail = self._make_request(f"tasks/{task['gid']}")
                detailed_tasks.append(task_detail.get("data", task))
            except Exception as e:
                print(f"Error fetching task details for {task['gid']}: {e}")
                detailed_tasks.append(task)
        
        return detailed_tasks
    
    def display_tasks(self, tasks: List[Dict]) -> None:
        """Display tasks in a readable format."""
        print(f"\nFound {len(tasks)} tasks:")
        print("-" * 80)
        
        for i, task in enumerate(tasks):
            print(f"\n{i + 1}. {task.get('name', 'Unnamed Task')}")
            print(f"   GID: {task.get('gid', 'N/A')}")
            print(f"   Completed: {task.get('completed', False)}")
            
            if task.get('assignee'):
                print(f"   Assignee: {task['assignee'].get('name', 'Unknown')}")
            
            if task.get('due_on'):
                print(f"   Due: {task['due_on']}")
            
            if task.get('notes'):
                notes = task['notes'][:200] + "..." if len(task['notes']) > 200 else task['notes']
                print(f"   Notes: {notes}")
            
            if task.get('tags'):
                tag_names = [tag.get('name', '') for tag in task['tags']]
                print(f"   Tags: {', '.join(tag_names)}")
    
    def select_task(self, tasks: List[Dict]) -> Optional[Dict]:
        """Allow user to select a task from the list."""
        if not tasks:
            print("No tasks available to select.")
            return None
        
        while True:
            try:
                selection = input("\nEnter the number of the task to implement (or 'q' to quit): ")
                if selection.lower() == 'q':
                    return None
                
                index = int(selection) - 1
                if 0 <= index < len(tasks):
                    return tasks[index]
                else:
                    print(f"Please enter a number between 1 and {len(tasks)}")
            except ValueError:
                print("Please enter a valid number or 'q' to quit")


def main():
    parser = argparse.ArgumentParser(description="Fetch and implement Asana External Projects tasks")
    parser.add_argument("--token", help="Asana API token (or set ASANA_API_TOKEN env var)")
    parser.add_argument("--workspace-id", help="Asana workspace ID (or set ASANA_WORKSPACE_ID env var)")
    parser.add_argument("--project-id", help="Asana project ID (or set ASANA_PROJECT_ID env var)")
    parser.add_argument("--show-completed", action="store_true", help="Show completed tasks")
    parser.add_argument("--list-projects", action="store_true", help="List all projects in workspace")
    
    args = parser.parse_args()
    
    # Get credentials from environment or arguments
    token = args.token or os.getenv("ASANA_API_TOKEN")
    workspace_id = args.workspace_id or os.getenv("ASANA_WORKSPACE_ID")
    project_id = args.project_id or os.getenv("ASANA_PROJECT_ID")
    
    if not token:
        print("Error: Asana API token is required. Set ASANA_API_TOKEN environment variable or use --token")
        return 1
    
    client = AsanaExternalProjects(token, workspace_id, project_id)
    
    # If no workspace ID, get the first available workspace
    if not workspace_id:
        workspaces = client.get_workspaces()
        if workspaces:
            workspace_id = workspaces[0]["gid"]
            client.workspace_id = workspace_id
            print(f"Using workspace: {workspaces[0]['name']} (GID: {workspace_id})")
        else:
            print("Error: No workspaces found")
            return 1
    
    # If listing projects
    if args.list_projects:
        projects = client.get_projects()
        print("\nAvailable projects:")
        for project in projects:
            print(f"- {project['name']} (GID: {project['gid']})")
        return 0
    
    # If no project ID, try to find External Projects
    if not project_id:
        project_id = client.find_external_project()
        if not project_id:
            choice = input("\nEnter project number or GID: ")
            try:
                projects = client.get_projects()
                index = int(choice) - 1
                if 0 <= index < len(projects):
                    project_id = projects[index]["gid"]
                else:
                    project_id = choice  # Assume it's a GID
            except ValueError:
                project_id = choice  # Assume it's a GID
        
        client.project_id = project_id
    
    # Fetch tasks
    completed_since = None if args.show_completed else "now"
    tasks = client.get_tasks(project_id, completed_since)
    
    if not tasks:
        print("No tasks found in the project")
        return 0
    
    # Display and select task
    client.display_tasks(tasks)
    selected_task = client.select_task(tasks)
    
    if selected_task:
        print(f"\nSelected task: {selected_task.get('name', 'Unnamed Task')}")
        print(f"GID: {selected_task.get('gid')}")
        print("\nTask details:")
        print(json.dumps(selected_task, indent=2, default=str))
        
        # Create implementation file
        task_name = selected_task.get('name', 'unnamed_task')
        safe_name = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in task_name)
        safe_name = safe_name.replace(' ', '_').lower()[:50]
        
        filename = f"asana_task_{safe_name}.py"
        with open(filename, 'w') as f:
            f.write(f'''#!/usr/bin/env python3
"""
Implementation for Asana task: {task_name}
Task GID: {selected_task.get('gid')}
Created: {datetime.now().isoformat()}

Task Notes:
{selected_task.get('notes', 'No notes provided')}
"""

# TODO: Implement the task here

def main():
    """Main implementation for the task."""
    print("Task implementation placeholder")
    # Add your implementation here
    pass


if __name__ == "__main__":
    main()
''')
        
        print(f"\nCreated implementation file: {filename}")
        print("You can now implement the task in this file.")
    
    return 0


if __name__ == "__main__":
    exit(main())