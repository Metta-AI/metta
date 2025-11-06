#!/usr/bin/env -S uv run
"""Request evaluation script for direct checkpoint URIs."""

import argparse
import asyncio
import uuid

import bidict
import pydantic
import pydantic.fields

import metta.app_backend.clients.eval_task_client
import metta.app_backend.clients.stats_client
import metta.app_backend.metta_repo
import metta.app_backend.routes.eval_task_routes
import metta.common.util.collections
import metta.common.util.constants
import metta.rl.checkpoint_manager
import metta.setup.utils
import metta.sim.utils


class EvalRequest(pydantic.BaseModel):
    """Evaluation request configuration for direct checkpoint URIs."""

    evals: list[str]
    policies: list[str]  # Direct checkpoint URIs
    stats_server_uri: str = metta.common.util.constants.PROD_STATS_SERVER_URI

    git_hash: str | None = None

    allow_duplicates: bool = pydantic.fields.Field(default=False)
    dry_run: bool = pydantic.fields.Field(default=False)

    @pydantic.model_validator(mode="after")
    def validate(self) -> "EvalRequest":
        return self


def validate_and_normalize_policy_uri(policy_uri: str) -> str | None:
    """Validate that a policy URI is accessible and return normalized URI with metadata."""
    try:
        normalized_uri = metta.rl.checkpoint_manager.CheckpointManager.normalize_uri(policy_uri)
        artifact = metta.rl.checkpoint_manager.CheckpointManager.load_artifact_from_uri(normalized_uri)
        if artifact.state_dict is None:
            raise ValueError("Policy artifact did not contain model weights")
        del artifact
        return normalized_uri
    except Exception as e:
        metta.setup.utils.warning(f"Skipping invalid or inaccessible policy {policy_uri}: {e}")
        return None


async def _create_remote_eval_tasks(
    request: EvalRequest,
) -> None:
    metta.setup.utils.info(f"Validating authentication with stats server {request.stats_server_uri}...")
    stats_client = metta.app_backend.clients.stats_client.StatsClient.create(request.stats_server_uri)
    if stats_client is None:
        metta.setup.utils.warning("No stats client found")
        return

    metta.setup.utils.info(f"Validating {len(request.policies)} policy URIs...")

    # Validate and normalize all policy URIs
    policy_uris: list[str] = []
    for policy_uri in request.policies:
        result = validate_and_normalize_policy_uri(policy_uri)
        if result:
            policy_uris.append(result)

    if not policy_uris:
        metta.setup.utils.warning("No valid policies found")
        return

    # Register policies with stats server
    policy_ids: bidict.bidict[str, uuid.UUID] = metta.sim.utils.get_or_create_policy_ids(
        stats_client, [(uri, None) for uri in policy_uris]
    )

    if not policy_ids:
        metta.setup.utils.warning("Failed to register policies with stats server")
        return

    # Check for existing tasks if not allowing duplicates
    existing_tasks: dict[tuple[uuid.UUID, str], list[metta.app_backend.routes.eval_task_routes.TaskResponse]] = {}
    eval_task_client = metta.app_backend.clients.eval_task_client.EvalTaskClient(backend_url=request.stats_server_uri)

    if not request.allow_duplicates:
        metta.setup.utils.info("Checking for duplicate tasks...")
        task_filters = metta.app_backend.routes.eval_task_routes.TaskFilterParams(
            limit=1000,
            statuses=list(set(metta.app_backend.metta_repo.TaskStatus.__args__) - set(["canceled", "error"])),
            policy_ids=list(policy_ids.values()),
            git_hash=request.git_hash,
            sim_suites=request.evals,
        )
        all_tasks = await eval_task_client.get_all_tasks(filters=task_filters)
        existing_tasks = metta.common.util.collections.group_by(all_tasks.tasks, lambda t: (t.policy_id, t.sim_suite))

        if existing_tasks:
            metta.setup.utils.info("Skipping because they would be duplicates:")
            for (policy_id, sim_suite), existing in existing_tasks.items():
                policy_name = policy_ids.inv[policy_id]
                metta.setup.utils.debug(f"{policy_name} {sim_suite}:", indent=2)
                for task in existing:
                    status_str = {"unprocessed": "running"}.get(task.status, task.status)
                    metta.setup.utils.debug(f"{task.id} ({status_str})", indent=4)

    # Create task requests
    task_requests = []
    for policy_uri in policy_uris:
        policy_id = policy_ids.get(policy_uri)
        if policy_id is None:
            metta.setup.utils.warning(f"Policy '{policy_uri}' not found in policy_ids mapping, skipping")
            continue

        for eval_name in request.evals:
            # Check if this combination already exists
            if not request.allow_duplicates and (policy_id, eval_name) in existing_tasks:
                continue

            task_requests.append(
                metta.app_backend.routes.eval_task_routes.TaskCreateRequest(
                    policy_id=policy_id,
                    git_hash=request.git_hash,
                    sim_suite=eval_name,
                )
            )

    if not task_requests:
        metta.setup.utils.warning("No new tasks to create (all would be duplicates)")
        return

    metta.setup.utils.info(f"Creating {len(task_requests)} evaluation tasks for {len(policy_uris)} policies...")
    if request.dry_run:
        metta.setup.utils.info("Dry run, not creating tasks")
        return

    results: list[metta.app_backend.routes.eval_task_routes.TaskResponse] = await asyncio.gather(
        *[eval_task_client.create_task(task) for task in task_requests]
    )

    for policy_id, policy_results in metta.common.util.collections.group_by(
        results, lambda result: result.policy_id
    ).items():
        policy_name = policy_ids.inv[policy_id]
        metta.setup.utils.success(f"{policy_name}:", indent=2)
        for result in policy_results:
            metta.setup.utils.success(f"{result.sim_suite}: {result.id}", indent=4)

    frontend_base_url = {
        metta.common.util.constants.PROD_STATS_SERVER_URI: metta.common.util.constants.PROD_OBSERVATORY_FRONTEND_URL,
        metta.common.util.constants.DEV_STATS_SERVER_URI: metta.common.util.constants.DEV_OBSERVATORY_FRONTEND_URL,
    }.get(request.stats_server_uri)

    if frontend_base_url:
        metta.setup.utils.info(f"Visit {frontend_base_url}/eval-tasks to view tasks")


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
        - file://path/to/run/checkpoints/run_name:v10.mpt
        - s3://bucket/path/run/checkpoints/run_name:v10.mpt
        - ./path/to/run/checkpoints (directory; latest checkpoint auto-detected)""",
        required=True,
    )

    parser.add_argument(
        "--stats-server-uri",
        type=str,
        default=metta.common.util.constants.PROD_STATS_SERVER_URI,
        help="URI for the stats server",
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
        allow_duplicates=args.allow_duplicates,
        dry_run=args.dry_run,
    )

    # Run the async task creation
    asyncio.run(_create_remote_eval_tasks(eval_request))


if __name__ == "__main__":
    main()
