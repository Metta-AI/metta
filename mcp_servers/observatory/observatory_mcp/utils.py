"""Observatory MCP Server Utilities."""

import asyncio
import logging
from typing import Any

from metta.app_backend.clients.stats_client import StatsClient
from observatory_mcp.models import ErrorResponse, SuccessResponse

logger = logging.getLogger(__name__)


def format_success_response(data: Any) -> str:
    """Format success response as JSON string."""
    return SuccessResponse(data=data).model_dump_json(indent=2, exclude_none=True)


def format_error_response(error: Exception, tool_name: str, context: str | None = None) -> str:
    """Format error as JSON string for MCP response."""
    return ErrorResponse(
        tool=tool_name, message=str(error), error_type=type(error).__name__, context=context
    ).model_dump_json(indent=2, exclude_none=True)


def handle_backend_error(error: Exception, tool_name: str) -> str:
    """Handle errors from backend API calls."""
    logger.error(f"Backend error in {tool_name}: {error}", exc_info=True)

    error_type = type(error).__name__
    error_message = str(error)

    if "Connection" in error_type or "ConnectTimeout" in error_type:
        return format_error_response(
            Exception("Backend connection failed. Is the backend running at the configured URL?"),
            tool_name,
            context="Check METTA_MCP_BACKEND_URL environment variable",
        )

    if "HTTPStatusError" in error_type or "HTTPError" in error_type:
        if "401" in error_message or "403" in error_message:
            return format_error_response(
                Exception("Authentication failed. Check your machine token."),
                tool_name,
                context="Set METTA_MCP_MACHINE_TOKEN environment variable",
            )
        if "404" in error_message:
            return format_error_response(
                Exception("Backend endpoint not found. The backend may be outdated."), tool_name
            )
        if "500" in error_message:
            return format_error_response(Exception("Backend server error. Check backend logs."), tool_name)

    return format_error_response(error, tool_name)


def handle_validation_error(error: Exception, tool_name: str, field: str | None = None) -> str:
    """Handle validation errors (missing required fields, invalid types, etc.)."""
    error_msg = str(error)
    if field:
        error_msg = f"Validation error for field '{field}': {error_msg}"

    return format_error_response(
        ValueError(error_msg), tool_name, context="Check tool arguments and ensure all required fields are provided"
    )


def serialize_response_data(data: Any) -> Any:
    """Serialize response data to make it JSON-compatible."""
    if hasattr(data, "model_dump"):
        return data.model_dump(mode="json")
    if isinstance(data, dict):
        return {key: serialize_response_data(value) for key, value in data.items()}
    if isinstance(data, list):
        return [serialize_response_data(item) for item in data]
    return data


# Scorecard tool functions (moved from deleted tools/scorecard.py)
async def get_training_runs(client: StatsClient) -> str:
    """Get all training runs and policies."""
    logger.info("Getting all training runs and policies")
    try:
        response = await asyncio.to_thread(client.get_policies)
        data = serialize_response_data(response)
        logger.info("get_training_runs completed")
        return format_success_response(data)
    except Exception as e:
        return handle_backend_error(e, "get_training_runs")


async def get_policies(client: StatsClient) -> str:
    """Get all policies and training runs."""
    logger.info("Getting all policies and training runs")
    try:
        response = await asyncio.to_thread(client.get_policies)
        data = serialize_response_data(response)
        logger.info("get_policies completed")
        return format_success_response(data)
    except Exception as e:
        return handle_backend_error(e, "get_policies")


async def search_policies(
    client: StatsClient,
    search: str | None = None,
    policy_type: str | None = None,
    tags: list[str] | None = None,
    user_id: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> str:
    """Search policies with filters."""
    logger.info(
        f"Searching policies: search={search}, type={policy_type}, "
        f"tags={tags}, user_id={user_id}, limit={limit}, offset={offset}"
    )
    try:
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
    except Exception as e:
        return handle_backend_error(e, "search_policies")


async def get_eval_names(
    client: StatsClient,
    training_run_ids: list[str],
    run_free_policy_ids: list[str],
) -> str:
    """Get evaluation names for given training runs and policies."""
    logger.info(
        f"Getting eval names for {len(training_run_ids)} training runs "
        f"and {len(run_free_policy_ids)} run-free policies"
    )
    try:
        eval_response = await asyncio.to_thread(
            client.get_eval_names,
            training_run_ids=training_run_ids,
            run_free_policy_ids=run_free_policy_ids,
        )
        eval_names = eval_response.eval_names
        logger.info(f"get_eval_names completed (found {len(eval_names)} evals)")
        return format_success_response(eval_names)
    except Exception as e:
        return handle_backend_error(e, "get_eval_names")


async def get_available_metrics(
    client: StatsClient,
    training_run_ids: list[str],
    run_free_policy_ids: list[str],
    eval_names: list[str],
) -> str:
    """Get available metrics for given evals, training runs, and policies."""
    logger.info(
        f"Getting available metrics for {len(eval_names)} evals, "
        f"{len(training_run_ids)} training runs, {len(run_free_policy_ids)} policies"
    )
    try:
        metrics_response = await asyncio.to_thread(
            client.get_available_metrics,
            training_run_ids=training_run_ids,
            run_free_policy_ids=run_free_policy_ids,
            eval_names=eval_names,
        )
        metrics = metrics_response.metrics
        logger.info(f"get_available_metrics completed (found {len(metrics)} metrics)")
        return format_success_response(metrics)
    except Exception as e:
        return handle_backend_error(e, "get_available_metrics")


async def generate_scorecard(
    client: StatsClient,
    training_run_ids: list[str],
    run_free_policy_ids: list[str],
    eval_names: list[str],
    metric: str,
    policy_selector: str = "best",
) -> str:
    """Generate a scorecard (heatmap) for given policies and evaluations."""
    if policy_selector not in ["best", "latest"]:
        return format_error_response(
            ValueError(f"policy_selector must be 'best' or 'latest', got '{policy_selector}'"),
            "generate_scorecard",
        )

    logger.info(
        f"Generating scorecard: metric={metric}, selector={policy_selector}, "
        f"{len(eval_names)} evals, {len(training_run_ids)} runs, {len(run_free_policy_ids)} policies"
    )
    try:
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
    except Exception as e:
        return handle_backend_error(e, "generate_scorecard")


async def run_sql_query(client: StatsClient, sql: str) -> str:
    """Execute a SQL query against the backend database."""
    if not sql or not sql.strip():
        return format_error_response(
            ValueError("SQL query cannot be empty"), "run_sql_query"
        )

    logger.info(f"Executing SQL query: {sql[:100]}...")
    try:
        response = await asyncio.to_thread(client.sql_query, sql)
        data = serialize_response_data(response)
        logger.info("run_sql_query completed")
        return format_success_response(data)
    except Exception as e:
        return handle_backend_error(e, "run_sql_query")


async def generate_ai_query(client: StatsClient, description: str) -> str:
    """Generate a SQL query from a natural language description."""
    if not description or not description.strip():
        return format_error_response(
            ValueError("Description cannot be empty"), "generate_ai_query"
        )

    logger.info(f"Generating AI query from description: {description[:100]}...")
    try:
        response = await asyncio.to_thread(client.generate_ai_query, description)
        data = serialize_response_data(response)
        logger.info("generate_ai_query completed")
        return format_success_response(data)
    except Exception as e:
        return handle_backend_error(e, "generate_ai_query")
