from datetime import datetime
from itertools import islice
from typing import Iterable

import wandb
from pydantic import BaseModel
from wandb.apis.public.runs import Run

from metta.app_backend.stats_client import StatsClient


class WandbRunInfo(BaseModel):
    name: str
    id: str
    state: str
    created_at: datetime
    git_hash: str | None
    artifacts: list[dict[str, str]]


def find_training_jobs(
    wandb_tags: list[str] | None = None,
    author: str | None = None,
    state: str | None = None,
    created_after: str | None = None,
    created_before: str | None = None,
    entity: str = "metta-research",
    project: str = "metta",
    order_by: str = "-created_at",
    run_names: list[str] | None = None,
    limit: int = 50,
) -> Iterable[Run]:
    filters = {}
    if state:
        filters["state"] = state
    if author:
        filters["username"] = author
    if created_after:
        filters["created_at"] = {"$gte": created_after}

    if created_before:
        if "created_at" in filters:
            filters["created_at"]["$lte"] = created_before
        else:
            filters["created_at"] = {"$lte": created_before}
    if wandb_tags:
        filters["tags"] = {"$in": wandb_tags}
    if run_names:
        filters["name"] = {"$in": run_names}
    return islice(wandb.Api().runs(f"{entity}/{project}", filters=filters, order=order_by), limit)


def get_artifacts_for_runs(runs: Iterable[Run]) -> list[WandbRunInfo]:
    run_infos = []
    for run in runs:
        git_hash = (
            run.config.get("git_hash")
            or run.config.get("git_commit")
            or run.config.get("git", {}).get("commit")
            or run.summary.get("git_hash")
            or run.summary.get("git_commit")
            or (run.summary.get("_wandb") or {}).get("git", {}).get("commit")
            or (run.metadata or {}).get("git", {}).get("commit")
            or run.commit
        )

        artifacts_data = [
            {
                "name": artifact.name,
                "type": artifact.type,
                "version": artifact.version,
                "size": artifact.size,
                "created_at": artifact.created_at,
                "url": f"wandb://{run.entity}/{run.project}/{artifact.name}:{artifact.version}",
            }
            for artifact in run.logged_artifacts()
        ]

        run_infos.append(
            WandbRunInfo(
                name=run.name,
                id=run.id,
                state=run.state,
                created_at=datetime.fromisoformat(run.created_at),
                git_hash=git_hash,
                artifacts=artifacts_data,
            )
        )
    run_infos.sort(key=lambda x: x.created_at, reverse=True)

    return run_infos


def register_with_stats_server(runs: list[WandbRunInfo], stats_client: StatsClient) -> None:
    # Collect all model artifacts
    policies_to_create = [
        {
            "name": run.name,
            "url": artifact["url"],
            "description": run.name + f" ({run.git_hash[:8]})" if run.git_hash else "",
            "git_hash": run.git_hash,
        }
        for run in runs
        for artifact in run.artifacts
        if artifact["type"] == "model"
    ]

    if not policies_to_create:
        print("No model artifacts found to post as policies.")
        return

    existing_policies = stats_client.get_policy_ids([p["name"] for p in policies_to_create])
    existing_names = set(existing_policies.policy_ids.keys())
    unique_policies_to_create = [p for p in policies_to_create if p["name"] not in existing_names]

    posted_count = 0
    for policy_info in unique_policies_to_create:
        try:
            response = stats_client.create_policy(
                name=policy_info["name"], description=policy_info["description"], url=policy_info["url"]
            )
            print(f"  Posted policy: {policy_info['name']} (ID: {response.id})")
            posted_count += 1
        except Exception as e:
            print(f"  Error posting policy {policy_info['name']}: {e}")

    print(f"\nPosted {posted_count} new policies to stats database.")
