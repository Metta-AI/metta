"""Asana integration for creating tasks from GitHub PRs."""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

import asana
import httpx

from metta.app_backend.github_webhook.config import settings
from metta.app_backend.github_webhook.metrics import metrics
from metta.app_backend.github_webhook.retry import RetryExhausted, retry_with_backoff

logger = logging.getLogger(__name__)


def _get_asana_access_token() -> str:
    """
    Get Asana access token.

    Note: Asana's OAuth doesn't support client_credentials grant type for service-to-service auth.
    The atlas_app (client_id/client_secret) can only be used with authorization_code or refresh_token flows,
    which require initial user interaction. For automated services, a Personal Access Token (PAT) is required.

    This function:
    1. Tries OAuth client_credentials (will fail - Asana limitation)
    2. Falls back to PAT if available
    """
    # Try OAuth client_credentials (will fail, but we try for completeness)
    if settings.ASANA_CLIENT_ID and settings.ASANA_CLIENT_SECRET:
        try:
            token_url = "https://app.asana.com/-/oauth_token"
            data = {
                "grant_type": "client_credentials",
                "client_id": settings.ASANA_CLIENT_ID,
                "client_secret": settings.ASANA_CLIENT_SECRET,
            }

            with httpx.Client(timeout=30.0) as client:
                response = client.post(token_url, data=data)
                if response.status_code == 200:
                    token_data = response.json()
                    access_token = token_data.get("access_token")
                    if access_token:
                        logger.info("Successfully obtained access token using atlas_app credentials")
                        return access_token
                else:
                    # Expected: Asana doesn't support client_credentials grant
                    logger.debug(
                        f"OAuth client_credentials not supported by Asana ({response.status_code}). "
                        f"Falling back to PAT."
                    )
        except Exception as e:
            logger.debug(f"OAuth attempt failed, falling back to PAT: {e}")

    # Use PAT (required for service-to-service authentication)
    if settings.ASANA_PAT:
        return settings.ASANA_PAT

    raise ValueError(
        "Asana authentication not configured. "
        "Asana's OAuth doesn't support client_credentials for service auth. "
        "Please configure ASANA_PAT (Personal Access Token) in AWS Secrets Manager. "
        "The atlas_app (client_id/client_secret) can only be used with authorization_code/refresh_token flows."
    )


def _get_asana_client() -> asana.ApiClient:
    """Get configured Asana API client."""
    access_token = _get_asana_access_token()
    config = asana.Configuration()
    config.access_token = access_token
    return asana.ApiClient(config)


async def _get_github_to_asana_mapping(github_logins: set[str]) -> dict[str, str]:
    """
    Get GitHub login to Asana email mapping from roster project.

    This replicates the logic from .github/actions/asana/pr_gh_to_asana/github_asana_mapping.py
    """
    try:
        access_token = _get_asana_access_token()
    except ValueError:
        return {}

    # Get roster project ID from settings
    roster_project_gid = settings.ASANA_ROSTER_PROJECT_GID
    gh_login_field_gid = settings.ASANA_GH_LOGIN_FIELD_GID
    asana_email_field_gid = settings.ASANA_EMAIL_FIELD_GID

    mapping = {}

    try:
        import requests

        access_token = _get_asana_access_token()
        url = f"https://app.asana.com/api/1.0/projects/{roster_project_gid}/tasks"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }
        params = {
            "opt_fields": "custom_fields",
            "limit": 100,
        }

        # Run blocking requests.get in executor to avoid blocking event loop
        def _fetch_page(page_url: str, page_params: dict) -> tuple[dict, str | None]:
            """Fetch a single page of roster tasks."""
            resp = requests.get(page_url, headers=headers, params=page_params, timeout=30)
            if resp.status_code != 200:
                logger.warning(f"Failed to fetch roster tasks: {resp.status_code}")
                return {}, None
            data = resp.json()
            next_page = data.get("next_page")
            return data, next_page

        loop = asyncio.get_event_loop()
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            while url:
                data, next_page = await loop.run_in_executor(executor, _fetch_page, url, params)
                if not data:
                    break

                tasks = data.get("data", [])

                for task in tasks:
                    custom_fields = task.get("custom_fields", [])
                    gh_login = None
                    asana_email = None

                    for field in custom_fields:
                        if isinstance(field, dict):
                            if field.get("gid") == gh_login_field_gid:
                                value = field.get("text_value")
                                gh_login = value.strip() if value else None
                                if gh_login and gh_login not in github_logins:
                                    break
                            if field.get("gid") == asana_email_field_gid:
                                asana_email = field.get("text_value")
                                if asana_email and "," in asana_email:
                                    asana_email = asana_email.split(",")[0].strip()

                            if gh_login and asana_email and gh_login in github_logins:
                                mapping[gh_login] = asana_email
                                break

                    if len(mapping) == len(github_logins):
                        break

                # Check for next page
                if next_page:
                    url = f"https://app.asana.com/api/1.0/projects/{roster_project_gid}/tasks"
                    params["offset"] = next_page.get("offset")
                else:
                    break
        finally:
            executor.shutdown(wait=False)

    except Exception as e:
        logger.warning(f"Failed to fetch GitHub→Asana mapping from roster: {e}")

    logger.info(f"GitHub→Asana mapping: {mapping}")
    return mapping


async def _resolve_assignee_to_gid(github_login: str, all_logins: Optional[set[str]] = None) -> Optional[str]:
    """
    Map GitHub login to Asana user GID using roster project and workspace users.

    This resolves to user GID instead of email to avoid stale email issues.
    Falls back to unassigned if mapping fails.

    Args:
        github_login: GitHub username to map
        all_logins: All GitHub logins involved (for efficient batch lookup)

    Returns:
        Asana user GID or None if mapping fails
    """
    if not github_login:
        logger.warning("Empty github_login provided to _resolve_assignee_to_gid")
        return None

    # Get email mapping from roster project
    logins_to_lookup = all_logins if all_logins else {github_login}
    email_mapping = await _get_github_to_asana_mapping(logins_to_lookup)
    asana_email = email_mapping.get(github_login)

    if not asana_email:
        logger.warning(
            f"Could not map GitHub login '{github_login}' to Asana email - "
            f"task will be created unassigned"
        )
        return None

    # Resolve email to user GID by looking up workspace users
    try:
        api_client = _get_asana_client()
        users_api = asana.UsersApi(api_client)

        # Get all users in the workspace
        workspace_gid = settings.ASANA_WORKSPACE_GID
        if not workspace_gid:
            logger.warning("ASANA_WORKSPACE_GID not set - cannot resolve user GID")
            return None

        users_response = users_api.get_users_for_workspace(workspace_gid, {"opt_fields": "email,gid"})
        # Handle Asana API response - can be dict or generator
        users: list[dict[str, Any]] = []
        if isinstance(users_response, dict):
            users = users_response.get("data", []) or []
        elif users_response:
            # If it's a generator or other iterable, convert to list
            try:
                users = [item for item in users_response if isinstance(item, dict)]
            except (TypeError, AttributeError):
                users = []

        # Find user by email (case-insensitive)
        asana_email_lower = asana_email.lower()
        for user in users:
            user_email = user.get("email", "").lower()
            if user_email == asana_email_lower:
                user_gid = user.get("gid")
                if user_gid:
                    logger.info(
                        f"Mapped GitHub login '{github_login}' to Asana user GID '{user_gid}' "
                        f"(email: {asana_email})"
                    )
                    return user_gid

        logger.warning(
            f"Could not find Asana user GID for email '{asana_email}' in workspace - "
            f"task will be created unassigned"
        )
        return None

    except Exception as e:
        logger.error(f"Failed to resolve Asana user GID for email '{asana_email}': {e}", exc_info=True)
        return None


# Keep old function name for backward compatibility during migration
async def _resolve_assignee(github_login: str, all_logins: Optional[set[str]] = None) -> Optional[str]:
    """
    Deprecated: Use _resolve_assignee_to_gid instead.
    
    This function now returns user GID instead of email to avoid stale email issues.
    """
    return await _resolve_assignee_to_gid(github_login, all_logins)


async def create_task_from_pr(
    repo_full_name: str,
    pr_number: int,
    pr_title: str,
    pr_url: str,
    author_login: str,
    assignee_login: Optional[str],
) -> Optional[Dict[str, Any]]:
    """
    Create an Asana task for a GitHub pull request.

    Args:
        repo_full_name: Full repository name (e.g., "Metta-AI/metta")
        pr_number: Pull request number
        pr_title: Pull request title
        pr_url: Pull request URL
        author_login: GitHub username of PR author
        assignee_login: GitHub username of assignee (if any)

    Returns:
        Asana task dict with 'gid', 'permalink_url', etc., or None if creation failed
    """
    logger.info(f"Creating Asana task for PR #{pr_number} in {repo_full_name}")

    # Validate required config
    if not settings.ASANA_PAT:
        logger.error("ASANA_PAT not configured - cannot create Asana task")
        return None

    if not settings.ASANA_WORKSPACE_GID:
        logger.error("ASANA_WORKSPACE_GID not configured - cannot create Asana task")
        return None

    # Compute external_id
    external_id = f"{repo_full_name}#{pr_number}"

    # Determine who to assign the task to
    # Use assignee if present, otherwise fallback to author
    assigned_login = assignee_login if assignee_login else author_login

    # Resolve GitHub login to Asana user GID (pass all logins for efficient batch lookup)
    all_logins = {author_login}
    if assignee_login:
        all_logins.add(assignee_login)

    assignee_gid = await _resolve_assignee_to_gid(assigned_login, all_logins)
    if not assignee_gid:
        metrics.increment_counter("github_asana.mapping_failures", {"github_login": assigned_login})
        logger.warning(
            f"Could not resolve assignee for GitHub login '{assigned_login}' - "
            f"task will be created without an assignee"
        )

    # Construct task name and notes
    task_name = f"PR #{pr_number} – {pr_title}"

    task_notes = f"""Repo: {repo_full_name}
URL: {pr_url}
Author: {author_login}"""

    # Build task data
    # Use settings.ASANA_GITHUB_URL_FIELD_ID (supports both ID and GID formats)
    github_url_field_gid = settings.ASANA_GITHUB_URL_FIELD_ID
    task_data = {
        "name": task_name,
        "notes": task_notes,
        "workspace": settings.ASANA_WORKSPACE_GID,
        "custom_fields": {},
    }

    # Add GitHub URL to custom field if configured
    if github_url_field_gid:
        task_data["custom_fields"][github_url_field_gid] = pr_url

    # Add to project if configured
    if settings.ASANA_PROJECT_GID:
        task_data["projects"] = [settings.ASANA_PROJECT_GID]

    # Try to add assignee if resolved, but create task unassigned if it fails
    try_with_assignee = assignee_gid is not None

    if try_with_assignee:
        task_data["assignee"] = assignee_gid

    # Add external_id as a custom field if configured
    # Note: external_id is typically a custom field in Asana for deduplication
    # For Phase 1, we'll skip custom fields and rely on task notes
    # Future phases can add custom field support

    try:
        with metrics.timed("github_asana.sync.latency_ms", {"operation": "create_task"}):
            api_client = _get_asana_client()
            tasks_api = asana.TasksApi(api_client)

            logger.info(f"Calling Asana API to create task: {task_name}")

            def _create_task():
                """Blocking Asana API call - runs in executor."""
                try:
                    return tasks_api.create_task({"data": task_data}, {})
                except Exception as assignee_error:
                    if try_with_assignee and "assignee" in str(assignee_error):
                        logger.warning(
                            f"Failed to assign task to user GID {assignee_gid}, retrying without assignee: {assignee_error}"
                        )
                        task_data.pop("assignee", None)
                        return tasks_api.create_task({"data": task_data}, {})
                    else:
                        raise

            # Run blocking Asana SDK calls in executor to avoid blocking event loop
            loop = asyncio.get_event_loop()
            executor = ThreadPoolExecutor(max_workers=1)
            try:
                async def _create_task_async():
                    """Async wrapper to run blocking call in executor."""
                    return await loop.run_in_executor(executor, _create_task)

                try:
                    result = await retry_with_backoff(
                        _create_task_async,
                        max_retries=settings.ASANA_RETRY_MAX_ATTEMPTS,
                        initial_delay_ms=settings.ASANA_RETRY_INITIAL_DELAY_MS,
                        max_delay_ms=settings.ASANA_RETRY_MAX_DELAY_MS,
                        operation_name=f"create_task_{external_id}",
                    )

                    if isinstance(result, dict):
                        task_gid = result.get("gid")
                        task_url = result.get("permalink_url")
                        logger.info(f"Successfully created Asana task {task_gid}: {task_url}")
                        return result
                    else:
                        logger.error(f"Unexpected result type from Asana API: {type(result)}")
                        return None

                except RetryExhausted as e:
                    logger.error(f"Failed to create Asana task for {external_id} after retries: {e.last_exception}")
                    metrics.increment_counter(
                        "github_asana.dead_letter.count",
                        {"operation": "create_task", "external_id": external_id},
                    )
                    logger.error(
                        f"DEAD_LETTER: {external_id} - max retry exceeded for task creation",
                        extra={"kind": "dead_letter", "externalId": external_id, "reason": "max_retry_exceeded"},
                    )
                    return None
            finally:
                executor.shutdown(wait=False)

    except Exception as e:
        logger.error(f"Failed to create Asana task for {external_id}: {e}", exc_info=True)
        return None


async def find_task_by_github_url(pr_url: str) -> Optional[Dict[str, Any]]:
    """
    Find existing Asana task by GitHub PR URL.

    Searches using custom field if configured, otherwise searches task notes.

    Args:
        pr_url: GitHub PR URL (e.g., "https://github.com/Metta-AI/metta/pull/123")

    Returns:
        Asana task dict or None if not found
    """
    if not settings.ASANA_PAT or not settings.ASANA_WORKSPACE_GID:
        return None

    github_url_field_gid = settings.ASANA_GITHUB_URL_FIELD_ID

    try:
        api_client = _get_asana_client()
        tasks_api = asana.TasksApi(api_client)

        # Run blocking Asana SDK calls in executor to avoid blocking event loop
        loop = asyncio.get_event_loop()
        executor = ThreadPoolExecutor(max_workers=1)

        def _search_by_custom_field():
            """Blocking search by custom field."""
            if github_url_field_gid and settings.ASANA_PROJECT_GID:
                opt_fields = (
                    "permalink_url,custom_fields,name,notes,modified_at,completed,"
                    "assignee.email,followers.email"
                )
                opts = {
                    "opt_fields": opt_fields,
                    "projects.any": settings.ASANA_PROJECT_GID,
                    f"custom_fields.{github_url_field_gid}.value": pr_url,
                }
                results_generator = tasks_api.search_tasks_for_workspace(
                    settings.ASANA_WORKSPACE_GID, opts, item_limit=1
                )
                if results_generator:
                    try:
                        results = [r for r in results_generator if isinstance(r, dict)]  # type: ignore
                        if results:
                            return results[0]
                    except Exception as e:
                        logger.warning(f"Error iterating search results: {e}")
            return None

        def _search_by_notes():
            """Blocking search by notes."""
            if settings.ASANA_PROJECT_GID:
                project_tasks_generator = tasks_api.get_tasks_for_project(
                    settings.ASANA_PROJECT_GID,
                    {"opt_fields": "permalink_url,name,notes", "limit": 100},
                )
                if project_tasks_generator:
                    try:
                        project_tasks = [t for t in project_tasks_generator if isinstance(t, dict)]  # type: ignore
                        for task in project_tasks:
                            notes = task.get("notes", "") or ""
                            if pr_url in notes:
                                return task
                    except Exception as e:
                        logger.warning(f"Error iterating project tasks: {e}")
            return None

        try:
            # Try custom field search first
            task = await loop.run_in_executor(executor, _search_by_custom_field)
            if task:
                logger.info(f"Found Asana task via custom field search: {task.get('permalink_url')}")
                return task

            # Fallback to notes search
            task = await loop.run_in_executor(executor, _search_by_notes)
            if task:
                logger.info(f"Found Asana task via notes search: {task.get('permalink_url')}")
                return task
        finally:
            executor.shutdown(wait=False)

        logger.warning(f"Could not find Asana task for PR URL: {pr_url}")
        return None

    except Exception as e:
        logger.error(f"Failed to search for Asana task: {e}", exc_info=True)
        return None


async def update_task_assignee(task_gid: str, assignee_gid: Optional[str]) -> bool:
    """
    Update Asana task assignee using user GID.

    Args:
        task_gid: Asana task GID
        assignee_gid: Asana user GID (None to unassign)

    Returns:
        True if update succeeded, False otherwise
    """
    if not settings.ASANA_PAT:
        logger.error("ASANA_PAT not configured")
        return False

    try:
        with metrics.timed("github_asana.sync.latency_ms", {"operation": "update_assignee"}):
            api_client = _get_asana_client()
            tasks_api = asana.TasksApi(api_client)

            task_data = {}
            if assignee_gid:
                task_data["assignee"] = assignee_gid
            else:
                task_data["assignee"] = None

            def _update_assignee():
                """Blocking Asana API call - runs in executor."""
                tasks_api.update_task({"data": task_data}, task_gid, {})

            # Run blocking Asana SDK calls in executor to avoid blocking event loop
            loop = asyncio.get_event_loop()
            executor = ThreadPoolExecutor(max_workers=1)
            try:
                async def _update_assignee_async():
                    """Async wrapper to run blocking call in executor."""
                    return await loop.run_in_executor(executor, _update_assignee)

                try:
                    await retry_with_backoff(
                        _update_assignee_async,
                        max_retries=settings.ASANA_RETRY_MAX_ATTEMPTS,
                        initial_delay_ms=settings.ASANA_RETRY_INITIAL_DELAY_MS,
                        max_delay_ms=settings.ASANA_RETRY_MAX_DELAY_MS,
                        operation_name=f"update_assignee_{task_gid}",
                    )
                    logger.info(f"Updated task {task_gid} assignee to user GID: {assignee_gid or 'unassigned'}")
                    return True

                except RetryExhausted as e:
                    logger.error(f"Failed to update task assignee after retries: {e.last_exception}")
                    metrics.increment_counter(
                        "github_asana.dead_letter.count",
                        {"operation": "update_assignee", "task_gid": task_gid},
                    )
                    logger.error(
                        f"DEAD_LETTER: task {task_gid} - max retry exceeded for assignee update",
                        extra={"kind": "dead_letter", "taskGid": task_gid, "reason": "max_retry_exceeded"},
                    )
                    return False
            finally:
                executor.shutdown(wait=False)

    except Exception as e:
        logger.error(f"Failed to update task assignee: {e}", exc_info=True)
        return False


async def update_task_completed(task_gid: str, completed: bool) -> bool:
    """
    Update Asana task completed status.

    Args:
        task_gid: Asana task GID
        completed: True to mark as completed, False to mark as incomplete

    Returns:
        True if update succeeded, False otherwise
    """
    if not settings.ASANA_PAT:
        logger.error("ASANA_PAT not configured")
        return False

    try:
        with metrics.timed("github_asana.sync.latency_ms", {"operation": "update_completed"}):
            api_client = _get_asana_client()
            tasks_api = asana.TasksApi(api_client)

            task_data = {"completed": completed}

            def _update_completed():
                """Blocking Asana API call - runs in executor."""
                tasks_api.update_task({"data": task_data}, task_gid, {})

            # Run blocking Asana SDK calls in executor to avoid blocking event loop
            loop = asyncio.get_event_loop()
            executor = ThreadPoolExecutor(max_workers=1)
            try:
                async def _update_completed_async():
                    """Async wrapper to run blocking call in executor."""
                    return await loop.run_in_executor(executor, _update_completed)

                try:
                    await retry_with_backoff(
                        _update_completed_async,
                        max_retries=settings.ASANA_RETRY_MAX_ATTEMPTS,
                        initial_delay_ms=settings.ASANA_RETRY_INITIAL_DELAY_MS,
                        max_delay_ms=settings.ASANA_RETRY_MAX_DELAY_MS,
                        operation_name=f"update_completed_{task_gid}",
                    )
                    logger.info(f"Updated task {task_gid} completed status to: {completed}")
                    return True

                except RetryExhausted as e:
                    logger.error(f"Failed to update task completed status after retries: {e.last_exception}")
                    metrics.increment_counter(
                        "github_asana.dead_letter.count",
                        {"operation": "update_completed", "task_gid": task_gid},
                    )
                    logger.error(
                        f"DEAD_LETTER: task {task_gid} - max retry exceeded for completed update",
                        extra={"kind": "dead_letter", "taskGid": task_gid, "reason": "max_retry_exceeded"},
                    )
                    return False
            finally:
                executor.shutdown(wait=False)

    except Exception as e:
        logger.error(f"Failed to update task completed status: {e}", exc_info=True)
        return False
