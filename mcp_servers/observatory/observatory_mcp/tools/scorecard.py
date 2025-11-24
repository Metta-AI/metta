"""Observatory MCP Server Tools."""

import asyncio
import logging

from metta.app_backend.clients.stats_client import StatsClient
from observatory_mcp.utils import format_success_response, handle_backend_error, serialize_response_data

logger = logging.getLogger(__name__)


async def get_training_runs(client: StatsClient) -> str:
    logger.info("Getting all training runs and policies")
    response = await asyncio.to_thread(client.get_policies)
    data = serialize_response_data(response)
    logger.info("get_training_runs completed")
    return format_success_response(data)


async def get_policies(client: StatsClient) -> str:
    logger.info("Getting all policies and training runs")
    response = await asyncio.to_thread(client.get_policies)
    data = serialize_response_data(response)
    logger.info("get_policies completed")
    return format_success_response(data)


async def search_policies(
    client: StatsClient,
    search: str | None = None,
    policy_type: str | None = None,
    tags: list[str] | None = None,
    user_id: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> str:
    logger.info(
        f"Searching policies: search={search}, type={policy_type}, "
        f"tags={tags}, user_id={user_id}, limit={limit}, offset={offset}"
    )
    response = await asyncio.to_thread(
        client.search_policies,
        search=search,
        policy_type=policy_type,
        tags=tags,
        user_id=user_id,
        limit=limit,
        offset=offset,
    )
    data = serialize_response_data(response)
    logger.info("search_policies completed")
    return format_success_response(data)


async def get_eval_names(
    client: StatsClient,
    training_run_ids: list[str],
    run_free_policy_ids: list[str],
) -> str:
    logger.info(
        f"Getting eval names for {len(training_run_ids)} training runs "
        f"and {len(run_free_policy_ids)} run-free policies"
    )
    eval_response = await asyncio.to_thread(
        client.get_eval_names,
        training_run_ids=training_run_ids,
        run_free_policy_ids=run_free_policy_ids,
    )
    eval_names = eval_response.eval_names
    logger.info(f"get_eval_names completed (found {len(eval_names)} evals)")
    return format_success_response(eval_names)


async def get_available_metrics(
    client: StatsClient,
    training_run_ids: list[str],
    run_free_policy_ids: list[str],
    eval_names: list[str],
) -> str:
    logger.info(
        f"Getting available metrics for {len(eval_names)} evals, "
        f"{len(training_run_ids)} training runs, {len(run_free_policy_ids)} policies"
    )
    metrics_response = await asyncio.to_thread(
        client.get_available_metrics,
        training_run_ids=training_run_ids,
        run_free_policy_ids=run_free_policy_ids,
        eval_names=eval_names,
    )
    metrics = metrics_response.metrics
    logger.info(f"get_available_metrics completed (found {len(metrics)} metrics)")
    return format_success_response(metrics)


async def generate_scorecard(
    client: StatsClient,
    training_run_ids: list[str],
    run_free_policy_ids: list[str],
    eval_names: list[str],
    metric: str,
    policy_selector: str = "best",
) -> str:
    if policy_selector not in ["best", "latest"]:
        raise ValueError(f"policy_selector must be 'best' or 'latest', got '{policy_selector}'")

    logger.info(
        f"Generating scorecard: metric={metric}, selector={policy_selector}, "
        f"{len(eval_names)} evals, {len(training_run_ids)} runs, {len(run_free_policy_ids)} policies"
    )

    response = await asyncio.to_thread(
        client.generate_scorecard,
        training_run_ids=training_run_ids,
        run_free_policy_ids=run_free_policy_ids,
        eval_names=eval_names,
        metric=metric,
        policy_selector=policy_selector,
    )
    data = serialize_response_data(response)
    logger.info("generate_scorecard completed")
    return format_success_response(data)


async def run_sql_query(client: StatsClient, sql: str) -> str:
    if not sql or not sql.strip():
        raise ValueError("SQL query cannot be empty")

    logger.info(f"Executing SQL query: {sql[:100]}...")
    response = await asyncio.to_thread(client.sql_query, sql)
    data = serialize_response_data(response)
    logger.info("run_sql_query completed")
    return format_success_response(data)


async def generate_ai_query(client: StatsClient, description: str) -> str:
    if not description or not description.strip():
        raise ValueError("Description cannot be empty")

    logger.info(f"Generating AI query from description: {description[:100]}...")
    response = await asyncio.to_thread(client.generate_ai_query, description)
    data = serialize_response_data(response)
    logger.info("generate_ai_query completed")
    return format_success_response(data)
