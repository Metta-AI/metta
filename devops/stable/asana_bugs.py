"""Asana integration for checking blocking bugs.

This module provides optional Asana integration to automatically check
for blocking bugs before creating a stable release.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import asana as asana_sdk


def _get_asana_credentials() -> tuple[Optional[str], Optional[str]]:
    """Get Asana credentials from AWS Secrets Manager or environment variables.

    Returns:
        Tuple of (token, project_id) or (None, None) if not available
    """
    # Try AWS Secrets Manager first
    try:
        from softmax.aws.secrets_manager import get_secretsmanager_secret

        secret_str = get_secretsmanager_secret("asana/atlas_app", require_exists=False)
        if secret_str:
            secret_data = json.loads(secret_str)

            # Try various field names for token and project_id
            token = (
                secret_data.get("token")
                or secret_data.get("ASANA_TOKEN")
                or secret_data.get("access_token")
                or secret_data.get("personal_access_token")
            )
            project_id = (
                secret_data.get("project_id")
                or secret_data.get("ASANA_PROJECT_ID")
                or secret_data.get("atlas_project_id")
            )

            if token and project_id:
                print("✓ Using Asana credentials from AWS Secrets Manager")
                return token, project_id
            elif secret_str:
                print("⚠️  Asana secret found but missing token or project_id fields")
                print(f"    Available fields: {', '.join(secret_data.keys())}")
    except ImportError:
        # softmax module not available - will fall back to env vars
        pass
    except Exception as e:
        print(f"⚠️  Could not fetch Asana credentials from AWS Secrets Manager: {e}")

    # Fall back to environment variables
    token = os.getenv("ASANA_TOKEN")
    project_id = os.getenv("ASANA_PROJECT_ID")

    if token and project_id:
        print("✓ Using Asana credentials from environment variables")

    return token, project_id


def check_blockers() -> Optional[bool]:
    """Check for blocking bugs in Asana Active section.

    Returns:
        True if no blockers (clear to ship)
        False if blockers exist
        None if check is inconclusive (Asana unavailable or SDK missing)
    """

    token, project_id = _get_asana_credentials()

    if not (token and project_id):
        return None  # Asana not configured

    try:
        # Initialize Asana client
        config = asana_sdk.Configuration()
        config.access_token = token
        client = asana_sdk.ApiClient(config)

        # Get user info to verify auth
        users_api = asana_sdk.UsersApi(client)
        user = users_api.get_user("me", {})
        print(f"✓ Authenticated as {user.get('name', '?')}")

        # Get project sections
        sections_api = asana_sdk.SectionsApi(client)
        sections = sections_api.get_sections_for_project(project_id, {})

        # Find "Active" section
        active_section = next((s for s in sections if s["name"].lower() == "active"), None)
        if not active_section:
            print("No 'Active' section found in Asana project")
            return None

        # Get tasks in Active section
        tasks_api = asana_sdk.TasksApi(client)
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
