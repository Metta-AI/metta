"""Handler for GitHub pull_request webhook events."""

import logging
from typing import Any, Dict, Optional

from metta.app_backend.github_webhook.asana_integration import (
    _resolve_assignee_to_gid,
    create_task_from_pr,
    find_task_by_github_url,
    update_task_assignee,
    update_task_completed,
)
from metta.app_backend.github_webhook.github_integration import update_pr_description_with_asana_task
from metta.app_backend.github_webhook.metrics import metrics

logger = logging.getLogger(__name__)


async def handle_pull_request_event(
    payload: Dict[str, Any],
    delivery_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Handle a pull_request webhook event from GitHub.

    Supports all PR actions:
    - opened: create Asana task (with deduplication)
    - assigned/unassigned/edited: sync task assignee
    - closed: mark task complete (merged or not)
    - reopened: reopen task
    - synchronize: no-op (new commits pushed)
    - other actions: no-op with logging

    Args:
        payload: The webhook payload from GitHub
        delivery_id: GitHub delivery ID for tracking

    Returns:
        A structured plan dict describing what was done
    """
    action = payload.get("action")
    pr = payload.get("pull_request", {})
    repository = payload.get("repository", {})

    # Extract PR metadata
    repo_full_name = repository.get("full_name", "unknown/unknown")
    pr_number = pr.get("number")
    pr_title = pr.get("title", "")
    pr_url = pr.get("html_url", "")
    author_login = pr.get("user", {}).get("login", "")
    assignee_login = pr.get("assignee", {}).get("login") if pr.get("assignee") else None

    log_context = {
        "source": "github_asana_sync",
        "delivery": delivery_id,
        "event": "pull_request",
        "action": action,
        "repo": repo_full_name,
        "pr_number": pr_number,
    }

    logger.info(f"Processing pull_request event: {log_context}")

    external_id = f"{repo_full_name}#{pr_number}"

    if action == "opened":
        existing_task = await find_task_by_github_url(pr_url)
        if existing_task:
            plan = {"kind": "noop", "reason": "task exists"}
            metrics.increment_counter("github_asana.noops", {"reason": "task_exists"})
            logger.info(f"Task already exists for PR: {log_context | {'plan': plan}}")
            return plan

        asana_task = await create_task_from_pr(
            repo_full_name=repo_full_name,
            pr_number=pr_number,
            pr_title=pr_title,
            pr_url=pr_url,
            author_login=author_login,
            assignee_login=assignee_login,
        )

        assigned_to = assignee_login if assignee_login else author_login
        asana_task_gid = asana_task.get("gid") if asana_task else None
        asana_task_url = asana_task.get("permalink_url") if asana_task else None

        # If task creation failed, return error plan
        if not asana_task or not asana_task_gid:
            metrics.increment_counter("github_asana.task_creation_failed", {"external_id": external_id})
            plan = {
                "kind": "error",
                "externalId": external_id,
                "reason": "task_creation_failed",
                "assignedTo": assigned_to,
            }
            logger.error(f"Failed to create Asana task: {log_context | {'plan': plan}}")
            return plan

        if asana_task_url:
            await update_pr_description_with_asana_task(
                repo_full_name=repo_full_name,
                pr_number=pr_number,
                asana_task_url=asana_task_url,
            )

        plan = {
            "kind": "created",
            "externalId": external_id,
            "assignedTo": assigned_to,
            "asanaTaskGid": asana_task_gid,
        }

        metrics.increment_counter("github_asana.tasks_created")
        logger.info(f"Created Asana task: {log_context | {'plan': plan}}")
        return plan

    if action == "assigned":
        assignee_login = payload.get("assignee", {}).get("login") if payload.get("assignee") else None
        if not assignee_login:
            plan = {"kind": "noop", "reason": "no assignee in payload"}
            metrics.increment_counter("github_asana.noops", {"reason": "no_assignee"})
            logger.info(f"No action taken: {log_context | {'plan': plan}}")
            return plan

        asana_task = await find_task_by_github_url(pr_url)
        if not asana_task:
            plan = {"kind": "noop", "reason": "Asana task not found"}
            metrics.increment_counter("github_asana.noops", {"reason": "task_not_found"})
            logger.warning(f"Could not find Asana task for PR: {log_context}")
            return plan

        assignee_gid = _resolve_assignee_to_gid(assignee_login, {assignee_login})
        if not assignee_gid:
            plan = {
                "kind": "noop",
                "reason": f"No Asana mapping for GitHub login: {assignee_login}",
            }
            metrics.increment_counter("github_asana.mapping_failures", {"github_login": assignee_login})
            metrics.increment_counter("github_asana.noops", {"reason": "mapping_failure"})
            logger.warning(f"Could not map GitHub login to Asana user GID: {log_context | {'assignee': assignee_login}}")
            return plan

        task_gid = asana_task.get("gid")
        if not task_gid:
            plan = {"kind": "noop", "reason": "task GID not found"}
            return plan
        success = await update_task_assignee(task_gid, assignee_gid)

        plan = {
            "kind": "assign",
            "externalId": external_id,
            "assignTo": assignee_login,
            "source": "assigned",
            "success": success,
        }

        if success:
            metrics.increment_counter("github_asana.assign_updates")
        logger.info(f"Updated Asana task assignee: {log_context | {'plan': plan}}")
        return plan

    if action == "unassigned":
        asana_task = await find_task_by_github_url(pr_url)
        if not asana_task:
            plan = {"kind": "noop", "reason": "Asana task not found"}
            metrics.increment_counter("github_asana.noops", {"reason": "task_not_found"})
            logger.warning(f"Could not find Asana task for PR: {log_context}")
            return plan

        assignee_gid = _resolve_assignee_to_gid(author_login, {author_login})
        if not assignee_gid:
            plan = {
                "kind": "noop",
                "reason": f"No Asana mapping for GitHub login: {author_login}",
            }
            metrics.increment_counter("github_asana.mapping_failures", {"github_login": author_login})
            metrics.increment_counter("github_asana.noops", {"reason": "mapping_failure"})
            logger.warning(f"Could not map GitHub login to Asana user GID: {log_context | {'assignee': author_login}}")
            return plan

        task_gid = asana_task.get("gid")
        if not task_gid:
            plan = {"kind": "noop", "reason": "task GID not found"}
            return plan
        success = await update_task_assignee(task_gid, assignee_gid)

        plan = {
            "kind": "assign",
            "externalId": external_id,
            "assignTo": author_login,
            "source": "unassigned->author",
            "success": success,
        }

        if success:
            metrics.increment_counter("github_asana.assign_updates")
        logger.info(f"Reassigned Asana task to author: {log_context | {'plan': plan}}")
        return plan

    if action == "edited":
        asana_task = await find_task_by_github_url(pr_url)
        if not asana_task:
            plan = {"kind": "noop", "reason": "Asana task not found"}
            metrics.increment_counter("github_asana.noops", {"reason": "task_not_found"})
            logger.warning(f"Could not find Asana task for PR: {log_context}")
            return plan

        current_assignee_login = pr.get("assignee", {}).get("login") if pr.get("assignee") else None
        assignee_to_use = current_assignee_login if current_assignee_login else author_login

        assignee_gid = _resolve_assignee_to_gid(assignee_to_use, {assignee_to_use})
        if not assignee_gid:
            plan = {
                "kind": "noop",
                "reason": f"No Asana mapping for GitHub login: {assignee_to_use}",
            }
            metrics.increment_counter("github_asana.mapping_failures", {"github_login": assignee_to_use})
            metrics.increment_counter("github_asana.noops", {"reason": "mapping_failure"})
            logger.warning(f"Could not map GitHub login to Asana user GID: {log_context | {'assignee': assignee_to_use}}")
            return plan

        task_gid = asana_task.get("gid")
        if not task_gid:
            plan = {"kind": "noop", "reason": "task GID not found"}
            return plan
        success = await update_task_assignee(task_gid, assignee_gid)

        plan = {
            "kind": "assign",
            "externalId": external_id,
            "assignTo": assignee_to_use,
            "source": "edited",
            "success": success,
        }

        if success:
            metrics.increment_counter("github_asana.assign_updates")
        logger.info(f"Updated Asana task assignee from edited event: {log_context | {'plan': plan}}")
        return plan

    if action == "closed":
        asana_task = await find_task_by_github_url(pr_url)
        if not asana_task:
            plan = {"kind": "noop", "reason": "Asana task not found"}
            metrics.increment_counter("github_asana.noops", {"reason": "task_not_found"})
            logger.warning(f"Could not find Asana task for PR: {log_context}")
            return plan

        merged = pr.get("merged", False)
        task_gid = asana_task.get("gid")
        if not task_gid:
            plan = {"kind": "noop", "reason": "task GID not found"}
            return plan
        success = await update_task_completed(task_gid, True)

        plan = {
            "kind": "complete",
            "externalId": external_id,
            "merged": merged,
            "success": success,
        }

        if success:
            metrics.increment_counter("github_asana.tasks_completed")
        logger.info(f"Marked Asana task as completed: {log_context | {'plan': plan}}")
        return plan

    if action == "reopened":
        asana_task = await find_task_by_github_url(pr_url)
        if not asana_task:
            plan = {"kind": "noop", "reason": "Asana task not found"}
            metrics.increment_counter("github_asana.noops", {"reason": "task_not_found"})
            logger.warning(f"Could not find Asana task for PR: {log_context}")
            return plan

        task_gid = asana_task.get("gid")
        if not task_gid:
            plan = {"kind": "noop", "reason": "task GID not found"}
            return plan
        success = await update_task_completed(task_gid, False)

        plan = {
            "kind": "reopen",
            "externalId": external_id,
            "success": success,
        }

        if success:
            metrics.increment_counter("github_asana.tasks_reopened")
        logger.info(f"Reopened Asana task: {log_context | {'plan': plan}}")
        return plan

    if action == "synchronize":
        plan = {
            "kind": "noop",
            "reason": "new commits pushed; no asana change",
        }
        metrics.increment_counter("github_asana.noops", {"reason": "synchronize"})
        logger.info(f"No action taken: {log_context | {'plan': plan}}")
        return plan

    plan = {
        "kind": "noop",
        "reason": f"action {action} not implemented yet",
    }

    metrics.increment_counter("github_asana.noops", {"reason": "unimplemented_action"})
    logger.info(f"No action taken: {log_context | {'plan': plan}}")
    return plan
