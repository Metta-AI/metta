#!/usr/bin/env -S uv run
"""Request evaluation script."""

import argparse
import asyncio
import concurrent.futures
import logging
import uuid

import wandb
from bidict import bidict
from omegaconf import DictConfig
from pydantic import BaseModel, model_validator
from pydantic.fields import Field

from metta.agent.policy_record import PolicyRecord
from metta.agent.policy_store import PolicyMissingError, PolicySelectorType, PolicyStore
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
from metta.common.util.stats_client_cfg import get_stats_client_direct
from metta.setup.utils import debug, info, success, warning
from metta.sim.utils import get_or_create_policy_ids


class EvalRequest(BaseModel):
    """Evaluation request configuration."""

    evals: list[str]
    policies: list[str]
    stats_server_uri: str = PROD_STATS_SERVER_URI

    git_hash: str | None = None

    policy_select_type: PolicySelectorType = "latest"
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

    def get_wandb_cfg(self) -> DictConfig:
        return DictConfig(
            {
                "wandb": {"enabled": True, "project": self.wandb_project, "entity": self.wandb_entity},
                # requesting eval tasks does not really depend on device
                "device": "cpu",
            }
        )


def _get_policy_records_for_uri(
    policy_store: PolicyStore,
    policy_uri: str,
    selector_type: PolicySelectorType,
    select_num: int,
    select_metric: str,
    disallow_missing_policies: bool = False,
) -> tuple[str, list[PolicyRecord] | None]:
    try:
        records = policy_store.policy_records(
            uri_or_config=policy_uri,
            selector_type=selector_type,
            n=select_num,
            metric=select_metric,
        )
        return policy_uri, records
    except PolicyMissingError as e:
        if not disallow_missing_policies:
            warning(f"Skipping missing policy: {e}")
            return policy_uri, None
        else:
            raise


async def _create_remote_eval_tasks(
    request: EvalRequest,
) -> None:
    logger = logging.getLogger("tools.request_eval")
    info(f"Validating authentication with stats server {request.stats_server_uri}...")
    stats_client = get_stats_client_direct(request.stats_server_uri, logger)
    if stats_client is None:
        logger.error("No stats client found")
        return
    stats_client.validate_authenticated()

    policy_store = PolicyStore(cfg=request.get_wandb_cfg(), wandb_run=None)

    info(f"Retrieving {request.policy_select_type} policy records for {len(request.policies)} policies...")
    # Parallelize policy records retrieval
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_uri = {
            executor.submit(
                _get_policy_records_for_uri,
                policy_store,
                policy_uri,
                request.policy_select_type,
                request.policy_select_num,
                request.policy_select_metric,
                request.disallow_missing_policies,
            ): policy_uri
            for policy_uri in request.policies
        }

        policy_records_by_uri: dict[str, list[PolicyRecord]] = {}
        for future in concurrent.futures.as_completed(future_to_uri):
            policy_uri, records = future.result()
            if records is not None:
                policy_records_by_uri[policy_uri] = records

    all_policy_records = {pr.run_name: pr for prs in policy_records_by_uri.values() for pr in prs}
    policy_ids: bidict[str, uuid.UUID] = get_or_create_policy_ids(
        stats_client, [(pr.run_name, pr.uri, None) for pr in all_policy_records.values() if pr.uri is not None]
    )
    if not policy_ids:
        warning("No policies found")
        return

    # Check for existing tasks if not allowing duplicates
    existing_tasks: dict[tuple[uuid.UUID, str], list[TaskResponse]] = {}
    if not request.allow_duplicates:
        info("Checking for duplicate tasks...")
        task_filters = TaskFilterParams(
            limit=1000,
            statuses=list(set(TaskStatus.__args__) - set(["canceled", "error"])),
            policy_ids=list(policy_ids.values()),
            git_hash=request.git_hash,
            sim_suites=request.evals,
        )
        all_tasks = await stats_client.get_all_tasks(filters=task_filters)
        existing_tasks = group_by(all_tasks.tasks, lambda t: (t.policy_id, t.sim_suite))
        if existing_tasks:
            info("Skipping because they would be duplicates:")
            for (policy_id, sim_suite), existing in existing_tasks.items():
                policy_name = policy_ids.inv[policy_id]
                debug(f"{policy_name} {sim_suite}:", indent=2)
                for task in existing:
                    status_str = {"unprocessed": "running"}.get(task.status, task.status)
                    debug(f"{task.id} ({status_str})", indent=4)

    task_requests = [
        stats_client.create_task(
            TaskCreateRequest(
                policy_id=policy_id,
                git_hash=request.git_hash,
                sim_suite=eval_name,
            )
        )
        for policy_id in policy_ids.values()
        for eval_name in request.evals
        if not len(existing_tasks[(policy_id, eval_name)])
    ]

    if not task_requests:
        warning("No new tasks to create (all would be duplicates)")
        return

    info(f"Creating {len(task_requests)} evaluation tasks for {len(policy_ids)} policies...")
    if request.dry_run:
        info("Dry run, not creating tasks")
        return

    results: list[TaskResponse] = await asyncio.gather(*task_requests)
    for policy_id, policy_results in group_by(results, lambda result: result.policy_id).items():
        policy_name = policy_ids.inv[policy_id]
        success(f"{policy_name}:", indent=2)
        for result in policy_results:
            success(f"{result.sim_suite}: {result.id}", indent=4)

    frontend_base_url = {
        PROD_STATS_SERVER_URI: PROD_OBSERVATORY_FRONTEND_URL,
        DEV_STATS_SERVER_URI: DEV_OBSERVATORY_FRONTEND_URL,
    }.get(str(stats_client.http_client.base_url))
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
        help="""Policy string. Can be specified multiple times for multiple policies.
        Supported formats:
        - wandb://run/<run_name>[:<version>]
        - wandb://sweep/<sweep_name>[:<version>]
        - wandb://<entity>/<project>/<artifact_type>/<name>[:<version>]""",
        required=True,
    )

    parser.add_argument(
        "--policy-select-type",
        type=str,
        default="latest",
        choices=PolicySelectorType.__args__,
        help="Policy selection type.",
    )

    parser.add_argument(
        "--policy-select-num",
        type=int,
        default=1,
        help="Number of policies to select. Used only if policy-select-type is 'top'.",
    )

    parser.add_argument(
        "--policy-select-metric",
        type=str,
        default="score",
        help="Policy selection metric. Used only if policy-select-type is 'top'.",
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
