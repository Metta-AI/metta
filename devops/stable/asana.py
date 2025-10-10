"""Asana integration for checking blocking bugs.

This module provides optional Asana integration to automatically check
for blocking bugs before creating a stable release.
"""

from __future__ import annotations

import os
from typing import Optional

import asana


def check_blockers() -> Optional[bool]:
    """Check for blocking bugs in Asana Active section.

    Returns:
        True if no blockers (clear to ship)
        False if blockers exist
        None if check is inconclusive (Asana unavailable)
    """
    token = os.getenv("ASANA_TOKEN")
    project_id = os.getenv("ASANA_PROJECT_ID")

    if not (token and project_id):
        return None  # Asana not configured

    try:
        # Initialize Asana client
        config = asana.Configuration()
        config.access_token = token
        client = asana.ApiClient(config)

        # Get user info to verify auth
        users_api = asana.UsersApi(client)
        user = users_api.get_user("me", {})
        print(f"✓ Authenticated as {user.get('name', '?')}")

        # Get project sections
        sections_api = asana.SectionsApi(client)
        sections = sections_api.get_sections_for_project(project_id, {})

        # Find "Active" section
        active_section = next((s for s in sections if s["name"].lower() == "active"), None)
        if not active_section:
            print("No 'Active' section found in Asana project")
            return None

        # Get tasks in Active section
        tasks_api = asana.TasksApi(client)
        tasks = tasks_api.get_tasks_for_section(
            active_section["gid"],
            {"opt_fields": "name,completed,permalink_url"},
        )

        # Filter for incomplete tasks
        open_tasks = [t for t in tasks if not t.get("completed", False)]

        if not open_tasks:
            print("✅ No blocking bugs in Asana Active section")
            return True

        print(f"❌ Found {len(open_tasks)} blocking task(s) in Active:")
        for task in open_tasks[:10]:  # Show first 10
            print(f"  • {task['name']}")
            if task.get("permalink_url"):
                print(f"    {task['permalink_url']}")
        return False

    except Exception as e:
        print(f"Asana API error: {e}")
        return None
