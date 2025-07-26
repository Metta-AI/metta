#!/usr/bin/env -S uv run
"""Request evaluation script."""

import argparse
import asyncio
import logging

import wandb
from omegaconf import DictConfig
from pydantic import BaseModel, model_validator
from pydantic.fields import Field

from metta.agent.policy_record import PolicyRecord
from metta.agent.policy_store import PolicyStore
from metta.app_backend.routes.eval_task_routes import TaskCreateRequest, TaskResponse
from metta.common.util.collections import group_by, remove_none_values
from metta.common.util.stats_client_cfg import get_stats_client_direct
from metta.setup.utils import info, success, warning
from metta.sim.utils import get_or_create_policy_ids


class EvalRequest(BaseModel):
    """Evaluation request configuration."""

    evals: list[str]
    policies: list[str]
    stats_server_uri: str = "https://api.observatory.softmax-research.net"

    git_hash: str | None = None

    policy_select_type: str = "all"
    policy_select_metric: str = "all_score"
    policy_select_num: int = 1

    wandb_project: str = Field(default="")
    wandb_entity: str = Field(default="")

    @model_validator(mode="after")
    def validate(self) -> "EvalRequest":
        if not self.wandb_entity:
            if wandb.api.default_entity:
                self.wandb_entity = wandb.api.default_entity

        if not self.wandb_project:
            if self.wandb_entity == "metta-research":
                self.wandb_project = "metta"

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


async def _create_remote_eval_tasks(
    request: EvalRequest,
) -> None:
    logger = logging.getLogger("tools.request_eval")
    stats_client = get_stats_client_direct(request.stats_server_uri, logger)
    if stats_client is None:
        logger.error("No stats client found")
        return
    stats_client.validate_authenticated()

    policy_store = PolicyStore(cfg=request.get_wandb_cfg(), wandb_run=None)
    policy_records_by_uri: dict[str, list[PolicyRecord]] = {
        policy_uri: policy_store.policy_records(
            uri_or_config=policy_uri,
            selector_type=request.policy_select_type,
            n=request.policy_select_num,
            metric=request.policy_select_metric,
        )
        for policy_uri in request.policies
    }
    all_policy_records = {pr.run_name: pr for prs in policy_records_by_uri.values() for pr in prs}
    policy_ids = get_or_create_policy_ids(
        stats_client, [(pr.run_name, pr.uri) for pr in all_policy_records.values() if pr.uri is not None]
    )
    if not policy_ids:
        warning("No policies found")
        return

    info(f"Creating {len(policy_ids)} evaluation tasks...")
    tasks = [
        stats_client.create_task(
            TaskCreateRequest(
                policy_id=policy_id,
                git_hash=request.git_hash,
                sim_suite=eval_name,
            )
        )
        for policy_id in policy_ids.values()
        for eval_name in request.evals
    ]

    results: list[TaskResponse] = await asyncio.gather(*tasks)
    for policy_id, policy_results in group_by(results, lambda result: result.policy_id).items():
        policy_name = policy_ids.inv[policy_id]
        success(f"{policy_name}:", indent=2)
        for result in policy_results:
            success(f"{result.sim_suite}: {result.id}", indent=4)

    # TODO: mappings like this should determined somewhere else
    frontend_base_url = {
        "https://api.observatory.softmax-research.net": "https://observatory.softmax-research.net",
        "http://localhost:8000": "http://localhost:5173",
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
        help="Policy string. Can be specified multiple times for multiple policies.",
    )

    parser.add_argument(
        "--policy-select-type",
        type=str,
        default="all",
        help="Policy selection type. Can be 'all' or 'top'.",
    )

    parser.add_argument(
        "--policy-select-num",
        type=int,
        default=1,
        help="Number of policies to select. If policy-select-type is 'all', this is ignored.",
    )

    parser.add_argument(
        "--policy-select-metric",
        type=str,
        default="all_score",
        help="Policy selection metric. Can be 'all_score' or 'score'. If policy-select-type is 'all', this is ignored.",
    )

    parser.add_argument(
        "--stats-server-uri",
        type=str,
        default="https://api.observatory.softmax-research.net",
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
            )
        )
    )
    await _create_remote_eval_tasks(eval_request)


if __name__ == "__main__":
    asyncio.run(main())
