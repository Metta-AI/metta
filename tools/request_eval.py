#!/usr/bin/env -S uv run
"""Request evaluation script for direct checkpoint URIs."""

import argparse
import asyncio
import uuid

import wandb
from bidict import bidict
from pydantic import BaseModel, model_validator
from pydantic.fields import Field

from metta.app_backend.clients.eval_task_client import EvalTaskClient
from metta.app_backend.clients.stats_client import StatsClient
from metta.app_backend.metta_repo import TaskStatus
from metta.app_backend.routes.eval_task_routes import TaskCreateRequest, TaskFilterParams, TaskResponse
from metta.common.util.collections import group_by
from metta.common.util.constants import (
    DEV_OBSERVATORY_FRONTEND_URL,
    DEV_STATS_SERVER_URI,
    METTA_WANDB_ENTITY,
    METTA_WANDB_PROJECT,
    PROD_OBSERVATORY_FRONTEND_URL,
    PROD_STATS_SERVER_URI,
)
from metta.rl.checkpoint_manager import CheckpointManager
from metta.setup.utils import debug, info, success, warning
from metta.sim.utils import get_or_create_policy_ids


class EvalRequest(BaseModel):
    """Evaluation request configuration for direct checkpoint URIs."""

    evals: list[str]
    policies: list[str]  # Direct checkpoint URIs
    stats_server_uri: str = PROD_STATS_SERVER_URI

    git_hash: str | None = None

    wandb_project: str = Field(default="")
    wandb_entity: str = Field(default="")

    allow_duplicates: bool = Field(default=False)
    dry_run: bool = Field(default=False)

    @model_validator(mode="after")
    def validate(self) -> "EvalRequest":
        if not self.wandb_entity:
            if wandb.api.default_entity:
                self.wandb_entity = wandb.api.default_entity

        if not self.wandb_project:
            if self.wandb_entity == METTA_WANDB_ENTITY:
                self.wandb_project = METTA_WANDB_PROJECT

        assert self.wandb_project, "wandb_project must be set"
        assert self.wandb_entity, "wandb_entity must be set"
        return self


def validate_and_normalize_policy_uri(policy_uri: str) -> tuple[str, dict] | None:
    """Validate that a policy URI is accessible and return normalized URI with metadata."""
    try:
        # Normalize URI using CheckpointManager
        normalized_uri = CheckpointManager.normalize_uri(policy_uri)

        # Get metadata to verify the checkpoint exists
        metadata = CheckpointManager.get_policy_metadata(normalized_uri)

        # Quick validation that we can load the policy
        agent = CheckpointManager.load_from_uri(normalized_uri, device="cpu")
        del agent

        return normalized_uri, metadata
    except Exception as e:
        warning(f"Skipping invalid or inaccessible policy {policy_uri}: {e}")
        return None


async def _create_remote_eval_tasks(
    request: EvalRequest,
) -> None:
    info(f"Validating authentication with stats server {request.stats_server_uri}...")
    stats_client = StatsClient.create(request.stats_server_uri)
    if stats_client is None:
        warning("No stats client found")
        return

    info(f"Validating {len(request.policies)} policy URIs...")

    # Validate and normalize all policy URIs
    valid_policies = []
    for policy_uri in request.policies:
        result = validate_and_normalize_policy_uri(policy_uri)
        if result:
            normalized_uri, metadata = result
            valid_policies.append(
                {
                    "uri": normalized_uri,
                    "run_name": metadata.get("run_name", "unknown"),
                    "original_uri": policy_uri,
                }
            )

    if not valid_policies:
        warning("No valid policies found")
        return

    # Register policies with stats server
    policy_ids: bidict[str, uuid.UUID] = get_or_create_policy_ids(
        stats_client, [(p["uri"], None) for p in valid_policies]
    )

    if not policy_ids:
        warning("Failed to register policies with stats server")
        return

    # Check for existing tasks if not allowing duplicates
    existing_tasks: dict[tuple[uuid.UUID, str], list[TaskResponse]] = {}
    eval_task_client = EvalTaskClient(backend_url=request.stats_server_uri)

    if not request.allow_duplicates:
        info("Checking for duplicate tasks...")
        task_filters = TaskFilterParams(
            limit=1000,
            statuses=list(set(TaskStatus.__args__) - set(["canceled", "error"])),
            policy_ids=list(policy_ids.values()),
            git_hash=request.git_hash,
            sim_suites=request.evals,
        )
        all_tasks = await eval_task_client.get_all_tasks(filters=task_filters)
        existing_tasks = group_by(all_tasks.tasks, lambda t: (t.policy_id, t.sim_suite))

        if existing_tasks:
            info("Skipping because they would be duplicates:")
            for (policy_id, sim_suite), existing in existing_tasks.items():
                policy_name = policy_ids.inv[policy_id]
                debug(f"{policy_name} {sim_suite}:", indent=2)
                for task in existing:
                    status_str = {"unprocessed": "running"}.get(task.status, task.status)
                    debug(f"{task.id} ({status_str})", indent=4)

    # Create task requests
    task_requests = []
    for policy in valid_policies:
        run_name = policy["run_name"]
        policy_id = policy_ids.get(run_name)
        if policy_id is None:
            warning(f"Policy '{run_name}' not found in policy_ids mapping, skipping")
            continue

        for eval_name in request.evals:
            # Check if this combination already exists
            if not request.allow_duplicates and (policy_id, eval_name) in existing_tasks:
                continue

            task_requests.append(
                TaskCreateRequest(
                    policy_id=policy_id,
                    git_hash=request.git_hash,
                    sim_suite=eval_name,
                )
            )

    if not task_requests:
        warning("No new tasks to create (all would be duplicates)")
        return

    info(f"Creating {len(task_requests)} evaluation tasks for {len(valid_policies)} policies...")
    if request.dry_run:
        info("Dry run, not creating tasks")
        return

    results: list[TaskResponse] = await asyncio.gather(*[eval_task_client.create_task(task) for task in task_requests])

    for policy_id, policy_results in group_by(results, lambda result: result.policy_id).items():
        policy_name = policy_ids.inv[policy_id]
        success(f"{policy_name}:", indent=2)
        for result in policy_results:
            success(f"{result.sim_suite}: {result.id}", indent=4)

    frontend_base_url = {
        PROD_STATS_SERVER_URI: PROD_OBSERVATORY_FRONTEND_URL,
        DEV_STATS_SERVER_URI: DEV_OBSERVATORY_FRONTEND_URL,
    }.get(request.stats_server_uri)

    if frontend_base_url:
        info(f"Visit {frontend_base_url}/eval-tasks to view tasks")


def main():
    parser = argparse.ArgumentParser(description="Request evaluation tasks for direct checkpoint URIs")

    parser.add_argument(
        "--eval",
        action="append",
        dest="evals",
        required=True,
        help="Evaluation suite name. Can be specified multiple times for multiple evaluations.",
    )

    parser.add_argument(
        "--policy",
        action="append",
        dest="policies",
        help="""Direct policy checkpoint URI. Can be specified multiple times for multiple policies.
        Supported formats:
        - file://path/to/checkpoint.pt
        - wandb://entity/project/artifact:version
        - s3://bucket/path/to/checkpoint.pt""",
        required=True,
    )

    parser.add_argument(
        "--stats-server-uri",
        type=str,
        default=PROD_STATS_SERVER_URI,
        help="URI for the stats server",
    )

    parser.add_argument(
        "--wandb-project",
        type=str,
        help="Wandb project name",
    )

    parser.add_argument(
        "--wandb-entity",
        type=str,
        help="Wandb entity name",
    )

    parser.add_argument(
        "--git-hash",
        type=str,
        help="Git hash to associate with evaluation tasks",
    )

    parser.add_argument(
        "--allow-duplicates",
        action="store_true",
        help="Allow duplicate tasks to be created",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode - don't actually create tasks",
    )

    args = parser.parse_args()

    # Parse arguments into Pydantic model
    eval_request = EvalRequest(
        evals=args.evals,
        policies=args.policies,
        stats_server_uri=args.stats_server_uri,
        git_hash=args.git_hash,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        allow_duplicates=args.allow_duplicates,
        dry_run=args.dry_run,
    )

    # Run the async task creation
    asyncio.run(_create_remote_eval_tasks(eval_request))


if __name__ == "__main__":
    main()
