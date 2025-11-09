"""
Observatory MCP Server Tools

Tool handler functions for the Observatory MCP server.
"""

import logging

from metta.app_backend.clients.scorecard_client import ScorecardClient

from observatory_mcp.utils import format_success_response, handle_backend_error, serialize_response_data

logger = logging.getLogger(__name__)


async def get_training_runs(client: ScorecardClient) -> str:
    """Get all training runs from the backend.

    Args:
        client: ScorecardClient instance for backend API calls

    Returns:
        JSON string with training runs data
    """
    try:
        logger.info("Getting all training runs and policies")
        response = await client.get_policies()
        data = serialize_response_data(response)
        logger.info("get_training_runs completed")
        return format_success_response(data)
    except Exception as e:
        logger.warning(f"get_training_runs failed: {e}")
        return handle_backend_error(e, "get_training_runs")


async def get_policies(client: ScorecardClient) -> str:
    """Get all policies and training runs from the backend.

    Args:
        client: ScorecardClient instance

    Returns:
        JSON string with policies data
    """
    try:
        logger.info("Getting all policies and training runs")
        response = await client.get_policies()
        data = serialize_response_data(response)
        logger.info("get_policies completed")
        return format_success_response(data)
    except Exception as e:
        logger.warning(f"get_policies failed: {e}")
        return handle_backend_error(e, "get_policies")


async def search_policies(
    client: ScorecardClient,
    search: str | None = None,
    policy_type: str | None = None,
    tags: list[str] | None = None,
    user_id: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> str:
    """Search policies with filtering and pagination.

    Args:
        client: ScorecardClient instance
        search: Search term for policy names (case-insensitive partial match)
        policy_type: Filter by type ('training_run' or 'policy')
        tags: Filter by tags (policies must have at least one matching tag)
        user_id: Filter by user ID
        limit: Maximum results (1-1000, default: 100)
        offset: Number of results to skip (default: 0)

    Returns:
        JSON string with matching policies
    """
    try:
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
    except Exception as e:
        logger.warning(f"search_policies failed: {e}")
        return handle_backend_error(e, "search_policies")


async def get_eval_names(
    client: ScorecardClient,
    training_run_ids: list[str],
    run_free_policy_ids: list[str],
) -> str:
    """Get available evaluation names for selected training runs and policies.

    Args:
        client: ScorecardClient instance
        training_run_ids: List of training run IDs
        run_free_policy_ids: List of run-free policy IDs

    Returns:
        JSON string with list of eval names in format 'eval_category/env_name'
    """
    try:
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
    except Exception as e:
        logger.warning(f"get_eval_names failed: {e}")
        return handle_backend_error(e, "get_eval_names")


async def get_available_metrics(
    client: ScorecardClient,
    training_run_ids: list[str],
    run_free_policy_ids: list[str],
    eval_names: list[str],
) -> str:
    """Get available metrics for selected policies and evaluations.

    Args:
        client: ScorecardClient instance
        training_run_ids: List of training run IDs
        run_free_policy_ids: List of run-free policy IDs
        eval_names: List of evaluation names (format: 'eval_category/env_name')

    Returns:
        JSON string with list of available metrics
    """
    try:
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
    except Exception as e:
        logger.warning(f"get_available_metrics failed: {e}")
        return handle_backend_error(e, "get_available_metrics")


async def generate_scorecard(
    client: ScorecardClient,
    training_run_ids: list[str],
    run_free_policy_ids: list[str],
    eval_names: list[str],
    metric: str,
    policy_selector: str = "best",
) -> str:
    """Generate scorecard data for selected policies and evaluations.

    Args:
        client: ScorecardClient instance
        training_run_ids: List of training run IDs
        run_free_policy_ids: List of run-free policy IDs
        eval_names: List of evaluation names (format: 'eval_category/env_name')
        metric: Metric to use for scorecard (e.g., 'reward', 'score', 'episode_length')
        policy_selector: Policy selection strategy ('best' or 'latest', default: 'best')

    Returns:
        JSON string with scorecard data (policies, evals, cells)
    """
    try:
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
    except Exception as e:
        logger.warning(f"generate_scorecard failed: {e}")
        return handle_backend_error(e, "generate_scorecard")


async def run_sql_query(client: ScorecardClient, sql: str) -> str:
    """Execute SQL query against the backend database.

    Args:
        client: ScorecardClient instance
        sql: SQL query string to execute

    Returns:
        JSON string with query results (columns, rows, row_count)
    """
    try:
        if not sql or not sql.strip():
            raise ValueError("SQL query cannot be empty")

        logger.info(f"Executing SQL query: {sql[:100]}...")
        response = await client.sql_query(sql)
        data = serialize_response_data(response)
        logger.info("run_sql_query completed")
        return format_success_response(data)
    except Exception as e:
        logger.warning(f"run_sql_query failed: {e}")
        return handle_backend_error(e, "run_sql_query")


async def generate_ai_query(client: ScorecardClient, description: str) -> str:
    """Generate SQL query from natural language description.

    Args:
        client: ScorecardClient instance
        description: Natural language description of desired query

    Returns:
        JSON string with generated SQL query
    """
    try:
        if not description or not description.strip():
            raise ValueError("Description cannot be empty")

        logger.info(f"Generating AI query from description: {description[:100]}...")
        response = await client.generate_ai_query(description)
        data = serialize_response_data(response)
        logger.info("generate_ai_query completed")
        return format_success_response(data)
    except Exception as e:
        logger.warning(f"generate_ai_query failed: {e}")
        return handle_backend_error(e, "generate_ai_query")
