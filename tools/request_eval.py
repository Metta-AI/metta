#!/usr/bin/env -S uv run
"""Request evaluation script.

Simple, minimal version that works directly with CheckpointManager.
"""

import argparse
import asyncio
import concurrent.futures
import uuid
from pathlib import Path

import wandb
from bidict import bidict
from pydantic import BaseModel, model_validator
from pydantic.fields import Field

from metta.app_backend.clients.eval_task_client import EvalTaskClient
from metta.app_backend.clients.stats_client import StatsClient
from metta.app_backend.metta_repo import TaskStatus
from metta.app_backend.routes.eval_task_routes import TaskCreateRequest, TaskFilterParams, TaskResponse
from metta.common.util.collections import group_by, remove_none_values
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
    """Evaluation request configuration."""

    evals: list[str]
    policies: list[str]
    stats_server_uri: str = PROD_STATS_SERVER_URI

    git_hash: str | None = None

    policy_select_type: str = "latest"
    policy_select_metric: str = "score"
    policy_select_num: int = 1

    wandb_project: str = Field(default="")
    wandb_entity: str = Field(default="")

    disallow_missing_policies: bool = Field(default=False)
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


def _get_policies_for_path(
    policy_path: str,
    selector_type: str,
    select_num: int,
    select_metric: str,
    disallow_missing_policies: bool = False,
) -> tuple[str, list[tuple[str, str]] | None]:
    """Get policies from a file path - minimal approach."""
    try:
        # Remove file:// prefix if present
        if policy_path.startswith("file://"):
            policy_path = policy_path[7:]

        path = Path(policy_path)

        if path.is_file():
            # Direct file - extract run name from path structure
            if path.parent.name == "checkpoints":
                run_name = path.parent.parent.name
            else:
                run_name = path.stem  # Use filename without extension
            return policy_path, [(str(path), run_name)]

        if not path.is_dir():
            if disallow_missing_policies:
                raise FileNotFoundError(f"Path does not exist: {path}")
            else:
                warning(f"Path does not exist: {path}")
                return policy_path, None

        # Directory with checkpoints
        if path.name == "checkpoints":
            run_name = path.parent.name
            run_dir = str(path.parent.parent)
        else:
            run_name = path.name
            run_dir = str(path.parent)

        checkpoint_manager = CheckpointManager(run_name=run_name, run_dir=run_dir)

        # Map selector types to strategies
        strategy_map = {"latest": "latest", "top": "best_score", "best_score": "best_score", "all": "all"}
        strategy = strategy_map.get(selector_type, "latest")

        checkpoint_paths = checkpoint_manager.select_checkpoints(
            strategy=strategy, count=select_num, metric=select_metric
        )

        if not checkpoint_paths:
            if disallow_missing_policies:
                raise FileNotFoundError(f"No checkpoints found in: {path}")
            else:
                warning(f"No checkpoints found in: {path}")
                return policy_path, None

        # Return list of (path, run_name) tuples
        results = [(str(checkpoint_path), run_name) for checkpoint_path in checkpoint_paths]
        return policy_path, results

    except Exception as e:
        if not disallow_missing_policies:
            warning(f"Error processing {policy_path}: {e}")
            return policy_path, None
        else:
            raise


async def _create_remote_eval_tasks(request: EvalRequest) -> None:
    info(f"Validating authentication with stats server {request.stats_server_uri}...")
    stats_client = StatsClient.create(request.stats_server_uri)
    if stats_client is None:
        warning("No stats client found")
        return

    info(f"Retrieving {request.policy_select_type} policies for {len(request.policies)} paths...")

    # Process all policy paths
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_path = {
            executor.submit(
                _get_policies_for_path,
                policy_path=policy_path,
                selector_type=request.policy_select_type,
                select_num=request.policy_select_num,
                select_metric=request.policy_select_metric,
                disallow_missing_policies=request.disallow_missing_policies,
            ): policy_path
            for policy_path in request.policies
        }

        all_policies = {}  # run_name -> (path, run_name)
        for future in concurrent.futures.as_completed(future_to_path):
            policy_path, results = future.result()
            if results is not None:
                for path, run_name in results:
                    all_policies[run_name] = (path, run_name)

    if not all_policies:
        warning("No policies found")
        return

    # Create policy IDs in stats database
    policy_ids: bidict[str, uuid.UUID] = get_or_create_policy_ids(
        stats_client, [(run_name, f"file://{path}", None) for path, run_name in all_policies.values()]
    )

    if not policy_ids:
        warning("Failed to create policy IDs")
        return

    # Check for existing tasks if not allowing duplicates
    existing_tasks: dict[tuple[uuid.UUID, str], list[TaskResponse]] = {}
    eval_task_client = EvalTaskClient(backend_url=request.stats_server_uri)
    if not request.allow_duplicates:
        info("Checking for duplicate tasks...")
        task_filters = TaskFilterParams(
            limit=1000,
            statuses=list(set(TaskStatus.__args__) - {"canceled", "error"}),
            policy_ids=list(policy_ids.values()),
            git_hash=request.git_hash,
            sim_suites=request.evals,
        )
        all_tasks = await eval_task_client.get_all_tasks(filters=task_filters)
        existing_tasks = group_by(all_tasks.tasks, lambda t: (t.policy_id, t.sim_suite))

        if existing_tasks:
            info("Skipping duplicate tasks:")
            for (policy_id, sim_suite), existing in existing_tasks.items():
                policy_name = policy_ids.inv[policy_id]
                debug(f"{policy_name} {sim_suite}:", indent=2)
                for task in existing:
                    status_str = {"unprocessed": "running"}.get(task.status, task.status)
                    debug(f"{task.id} ({status_str})", indent=4)

    # Create task requests
    task_requests = [
        TaskCreateRequest(
            policy_id=policy_id,
            git_hash=request.git_hash,
            sim_suite=eval_name,
        )
        for policy_id in policy_ids.values()
        for eval_name in request.evals
        if request.allow_duplicates or not existing_tasks.get((policy_id, eval_name))
    ]

    if not task_requests:
        warning("No new tasks to create (all would be duplicates)")
        return

    info(f"Creating {len(task_requests)} evaluation tasks for {len(policy_ids)} policies...")
    if request.dry_run:
        info("Dry run, not creating tasks")
        return

    # Create tasks
    results: list[TaskResponse] = await asyncio.gather(*[eval_task_client.create_task(task) for task in task_requests])

    # Display results
    for policy_id, policy_results in group_by(results, lambda result: result.policy_id).items():
        policy_name = policy_ids.inv[policy_id]
        success(f"{policy_name}:", indent=2)
        for result in policy_results:
            success(f"{result.sim_suite}: {result.id}", indent=4)

    # Show frontend URL
    frontend_base_url = {
        PROD_STATS_SERVER_URI: PROD_OBSERVATORY_FRONTEND_URL,
        DEV_STATS_SERVER_URI: DEV_OBSERVATORY_FRONTEND_URL,
    }.get(request.stats_server_uri)
    if frontend_base_url:
        info(f"Visit {frontend_base_url}/eval-tasks to view tasks")


async def main() -> None:
    """Main function to handle evaluation requests."""
    parser = argparse.ArgumentParser(
        description="Request evaluation with specified configurations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

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
        help="""Policy path. Can be specified multiple times for multiple policies.
        Supports:
        - Direct file paths: /path/to/checkpoint.pt
        - Checkpoint directories: /path/to/run/checkpoints
        - file:// URIs: file:///path/to/checkpoint""",
        required=True,
    )

    parser.add_argument(
        "--policy-select-type",
        type=str,
        default="latest",
        choices=["latest", "top", "best_score", "all"],
        help="Policy selection type.",
    )

    parser.add_argument(
        "--policy-select-num",
        type=int,
        default=1,
        help="Number of policies to select. Used with 'top' or 'best_score'.",
    )

    parser.add_argument(
        "--policy-select-metric",
        type=str,
        default="score",
        help="Policy selection metric. Used with 'top' or 'best_score'.",
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
        default="",
        help="W&B project to use for the evaluation",
    )

    parser.add_argument(
        "--wandb-entity",
        type=str,
        default="",
        help="W&B entity to use for the evaluation",
    )

    parser.add_argument(
        "--git-hash",
        type=str,
        default=None,
        help="Git hash to use for the evaluation",
    )

    parser.add_argument(
        "--disallow-missing-policies",
        action="store_true",
        help="Error if a policy cannot be found",
    )

    parser.add_argument(
        "--allow-duplicates",
        action="store_true",
        help="Allow scheduling duplicate policy,eval pairs that are already scheduled or running",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not create tasks, just print what would be created",
    )

    args = parser.parse_args()

    # Parse arguments into Pydantic model
    eval_request = EvalRequest.model_validate(
        remove_none_values(
            dict(
                evals=args.evals,
                policies=args.policies,
                stats_server_uri=args.stats_server_uri,
                git_hash=args.git_hash,
                policy_select_type=args.policy_select_type,
                policy_select_num=args.policy_select_num,
                policy_select_metric=args.policy_select_metric,
                wandb_project=args.wandb_project,
                wandb_entity=args.wandb_entity,
                disallow_missing_policies=args.disallow_missing_policies,
                allow_duplicates=args.allow_duplicates,
                dry_run=args.dry_run,
            )
        )
    )
    await _create_remote_eval_tasks(eval_request)


if __name__ == "__main__":
    asyncio.run(main())
