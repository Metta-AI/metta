"""Observatory MCP Server Tools."""

import logging

from metta.app_backend.clients.scorecard_client import ScorecardClient
from observatory_mcp.utils import format_success_response, handle_backend_error, serialize_response_data

logger = logging.getLogger(__name__)


async def get_training_runs(client: ScorecardClient) -> str:
    """Get all training runs from the backend."""
    logger.info("Getting all training runs and policies")
    response = await client.get_policies()
    data = serialize_response_data(response)
    logger.info("get_training_runs completed")
    return format_success_response(data)


async def get_policies(client: ScorecardClient) -> str:
    """Get all policies and training runs from the backend."""
    logger.info("Getting all policies and training runs")
    response = await client.get_policies()
    data = serialize_response_data(response)
    logger.info("get_policies completed")
    return format_success_response(data)


async def search_policies(
    client: ScorecardClient,
    search: str | None = None,
    policy_type: str | None = None,
    tags: list[str] | None = None,
    user_id: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> str:
    """Search policies with filtering and pagination."""
    logger.info(
        f"Searching policies: search={search}, type={policy_type}, "
        f"tags={tags}, user_id={user_id}, limit={limit}, offset={offset}"
    )
    response = await client.search_policies(
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
    client: ScorecardClient,
    training_run_ids: list[str],
    run_free_policy_ids: list[str],
) -> str:
    """Get available evaluation names for selected training runs and policies."""
    logger.info(
        f"Getting eval names for {len(training_run_ids)} training runs "
        f"and {len(run_free_policy_ids)} run-free policies"
    )
    eval_names = await client.get_eval_names(
        training_run_ids=training_run_ids,
        run_free_policy_ids=run_free_policy_ids,
    )
    logger.info(f"get_eval_names completed (found {len(eval_names)} evals)")
    return format_success_response(eval_names)


async def get_available_metrics(
    client: ScorecardClient,
    training_run_ids: list[str],
    run_free_policy_ids: list[str],
    eval_names: list[str],
) -> str:
    """Get available metrics for selected policies and evaluations."""
    logger.info(
        f"Getting available metrics for {len(eval_names)} evals, "
        f"{len(training_run_ids)} training runs, {len(run_free_policy_ids)} policies"
    )
    metrics = await client.get_available_metrics(
        training_run_ids=training_run_ids,
        run_free_policy_ids=run_free_policy_ids,
        eval_names=eval_names,
    )
    logger.info(f"get_available_metrics completed (found {len(metrics)} metrics)")
    return format_success_response(metrics)


async def generate_scorecard(
    client: ScorecardClient,
    training_run_ids: list[str],
    run_free_policy_ids: list[str],
    eval_names: list[str],
    metric: str,
    policy_selector: str = "best",
) -> str:
    """Generate scorecard data for selected policies and evaluations."""
    if policy_selector not in ["best", "latest"]:
        raise ValueError(f"policy_selector must be 'best' or 'latest', got '{policy_selector}'")

    logger.info(
        f"Generating scorecard: metric={metric}, selector={policy_selector}, "
        f"{len(eval_names)} evals, {len(training_run_ids)} runs, {len(run_free_policy_ids)} policies"
    )

    response = await client.generate_scorecard(
        training_run_ids=training_run_ids,
        run_free_policy_ids=run_free_policy_ids,
        eval_names=eval_names,
        metric=metric,
        policy_selector=policy_selector,  # type: ignore
    )
    data = serialize_response_data(response)
    logger.info("generate_scorecard completed")
    return format_success_response(data)


async def run_sql_query(client: ScorecardClient, sql: str) -> str:
    """Execute SQL query against the backend database."""
    if not sql or not sql.strip():
        raise ValueError("SQL query cannot be empty")

    logger.info(f"Executing SQL query: {sql[:100]}...")
    response = await client.sql_query(sql)
    data = serialize_response_data(response)
    logger.info("run_sql_query completed")
    return format_success_response(data)


async def generate_ai_query(client: ScorecardClient, description: str) -> str:
    """Generate SQL query from natural language description."""
    if not description or not description.strip():
        raise ValueError("Description cannot be empty")

    logger.info(f"Generating AI query from description: {description[:100]}...")
    response = await client.generate_ai_query(description)
    data = serialize_response_data(response)
    logger.info("generate_ai_query completed")
    return format_success_response(data)
