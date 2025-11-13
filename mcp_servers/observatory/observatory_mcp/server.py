"""
Observatory MCP Server

Model Context Protocol server that enables LLMs to interact with the Metta Observatory backend.
Provides access to training runs, policies, evaluations, scorecards, and SQL queries.
"""

import asyncio
import logging
import sys
from typing import Any, Dict, List

import boto3
import mcp.types as types
from botocore.exceptions import ClientError, NoCredentialsError
from mcp.server import Server
from mcp.server.stdio import stdio_server

try:
    import wandb
    from wandb import Api
except ImportError:
    wandb = None
    Api = None

from metta.adaptive.stores.wandb import WandbStore
from metta.app_backend.clients.scorecard_client import ScorecardClient
from metta.utils.s3 import S3Store
from pydantic import BaseModel, ValidationError

from .config import ObservatoryMCPConfig
from .descriptions import MCP_TOOL_DESCRIPTIONS
from metta.app_backend.routes.scorecard_routes import (
    EvalsRequest,
    MetricsRequest,
    PoliciesSearchRequest,
)
from metta.app_backend.routes.sql_routes import AIQueryRequest
from .models import (
    AnalyzeS3CheckpointProgressionInput,
    AnalyzeS3CheckpointUsageInput,
    AnalyzeSkypilotJobFailuresInput,
    AnalyzeSkypilotJobPerformanceInput,
    AnalyzeWandbBehavioralPatternsInput,
    AnalyzeWandbLearningCurvesInput,
    AnalyzeWandbTrainingProgressionInput,
    CheckS3ObjectExistsInput,
    CompareS3CheckpointsAcrossRunsInput,
    CompareSkypilotJobConfigsInput,
    CompareWandbRunsInput,
    CorrelateWandbMetricsInput,
    DiscoverWandbRunMetricsInput,
    ErrorResponse,
    FindBestS3CheckpointInput,
    GenerateScorecardInput,
    GenerateWandbTrainingInsightsInput,
    GetS3CheckpointMetadataInput,
    GetS3CheckpointStatisticsInput,
    GetS3CheckpointUrlInput,
    GetSkypilotJobCostEstimatesInput,
    GetSkypilotJobLogsInput,
    GetSkypilotJobStatusInput,
    GetSkypilotResourceUtilizationInput,
    GetWandbRunArtifactsInput,
    GetWandbRunInput,
    GetWandbRunLogsInput,
    GetWandbRunMetricsInput,
    IdentifyWandbCriticalMomentsInput,
    LinkS3CheckpointToSkypilotJobInput,
    LinkS3CheckpointToWandbRunInput,
    LinkSkypilotJobToS3CheckpointsInput,
    LinkSkypilotJobToWandbRunsInput,
    LinkWandbRunToS3CheckpointsInput,
    LinkWandbRunToSkypilotJobInput,
    ListS3CheckpointsInput,
    ListS3ReplaysInput,
    ListSkypilotJobsInput,
    ListWandbRunsInput,
    PredictWandbTrainingOutcomeInput,
    RunSqlQueryInput,
)
from .tools import s3, scorecard, skypilot, wandb

logger = logging.getLogger(__name__)


class ObservatoryMCPServer:
    """MCP Server that exposes Observatory backend functionality."""

    def __init__(self, server_name: str = "observatory-mcp", version: str = "0.1.0"):
        """Initialize the Observatory MCP Server.

        Args:
            server_name: Name of the MCP server (used in protocol)
            version: Version of the server
        """
        self.app = Server(server_name)
        self.version = version
        self.config = ObservatoryMCPConfig.from_env()

        config_errors = self.config.validate()
        if config_errors:
            logger.warning("Configuration validation errors:")
            for error in config_errors:
                logger.warning(f"  - {error}")

        self.scorecard_client = ScorecardClient(
            backend_url=self.config.backend_url,
            machine_token=self.config.machine_token,
        )

        # Initialize WandB store
        self.wandb_store: WandbStore | None = None
        if self.config.is_wandb_configured():
            try:
                if Api is None:
                    raise ImportError("wandb package not installed")

                # Try to use existing wandb authentication first
                try:
                    test_api = Api()
                    _ = test_api.viewer
                    logger.info("WandB API initialized using cached credentials")
                except Exception:
                    logger.info("Cached credentials not working, trying explicit login...")
                    if self.config.wandb_api_key:
                        wandb.login(key=self.config.wandb_api_key)
                        logger.info("WandB API initialized with provided API key")
                    else:
                        logger.info("WandB API initialized (authentication may be limited)")

                # Create WandB store instance
                self.wandb_store = WandbStore(
                    entity=self.config.wandb_entity,
                    project=self.config.wandb_project,
                )

                logger.info(
                    f"WandB store initialized (entity={self.config.wandb_entity}, project={self.config.wandb_project})"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize WandB store: {e}. WandB tools will be unavailable.")
                self.wandb_store = None
        else:
            logger.debug("WandB not configured. WandB tools will be unavailable.")

        # Initialize S3 client and store
        self.s3_client = None
        self.s3_bucket = self.config.s3_bucket
        self.s3_store: S3Store | None = None
        # Always try to initialize S3 client - it can use profile or default credentials
        try:
            if self.config.aws_profile:
                session = boto3.Session(profile_name=self.config.aws_profile)
                self.s3_client = session.client("s3")
                logger.info(f"S3 client initialized (profile={self.config.aws_profile}, bucket={self.s3_bucket})")
            else:
                self.s3_client = boto3.client("s3")
                logger.info(f"S3 client initialized (bucket={self.s3_bucket})")

            # Create S3 store instance
            self.s3_store = S3Store(self.s3_client, self.s3_bucket)
        except (NoCredentialsError, ClientError) as e:
            logger.warning(f"Failed to initialize S3 client: {e}. S3 tools will be unavailable.")
            self.s3_client = None
            self.s3_store = None

        # Skypilot doesn't need a client - we'll use subprocess directly
        # Store the URL for reference if needed
        if self.config.skypilot_url:
            logger.info(f"Skypilot URL configured: {self.config.skypilot_url}")

        self._setup_tools()
        self._setup_resources()

        logger.info(
            f"Observatory MCP Server initialized "
            f"(backend={self.config.backend_url}, "
            f"authenticated={self.config.is_authenticated()}, "
            f"wandb={'enabled' if self.wandb_store else 'disabled'}, "
            f"s3={'enabled' if self.s3_client else 'disabled'}, "
            f"skypilot={'enabled' if self.config.skypilot_url else 'disabled'})"
        )

    def _pydantic_to_mcp_schema(self, model: type[BaseModel]) -> Dict[str, Any]:
        """Convert Pydantic model to MCP inputSchema format.

        Args:
            model: Pydantic model class

        Returns:
            Dictionary representing the MCP inputSchema
        """
        schema = model.model_json_schema()
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        mcp_schema = {
            "type": "object",
            "properties": {},
            "required": required,
        }

        for prop_name, prop_info in properties.items():
            mcp_prop = {"type": prop_info.get("type"), "description": prop_info.get("description", "")}

            if "enum" in prop_info:
                mcp_prop["enum"] = prop_info["enum"]
            if "default" in prop_info:
                mcp_prop["default"] = prop_info["default"]
            if "minimum" in prop_info:
                mcp_prop["minimum"] = prop_info["minimum"]
            if "maximum" in prop_info:
                mcp_prop["maximum"] = prop_info["maximum"]
            if prop_info.get("type") == "array":
                items = prop_info.get("items", {})
                if isinstance(items, dict):
                    mcp_prop["items"] = {"type": items.get("type", "string")}
                else:
                    mcp_prop["items"] = items

            mcp_schema["properties"][prop_name] = mcp_prop

        return mcp_schema

    def _create_error_response(self, message: str, tool_name: str | None = None) -> str:
        """Create error response JSON string.

        Args:
            message: Error message
            tool_name: Optional tool name

        Returns:
            JSON string with error response
        """
        return ErrorResponse(tool=tool_name, message=message).model_dump_json(indent=2, exclude_none=True)

    def _get_tool_input_models(self) -> Dict[str, type[BaseModel] | None]:
        """Get mapping of tool names to their input models.

        Returns:
            Dictionary mapping tool names to input model classes (None for tools with no inputs)
        """
        return {
            "get_training_runs": None,
            "get_policies": None,
            "search_policies": PoliciesSearchRequest,
            "get_eval_names": EvalsRequest,
            "get_available_metrics": MetricsRequest,
            "generate_scorecard": GenerateScorecardInput,
            "run_sql_query": RunSqlQueryInput,
            "generate_ai_query": AIQueryRequest,
            "list_wandb_runs": ListWandbRunsInput,
            "get_wandb_run": GetWandbRunInput,
            "get_wandb_run_metrics": GetWandbRunMetricsInput,
            "discover_wandb_run_metrics": DiscoverWandbRunMetricsInput,
            "get_wandb_run_artifacts": GetWandbRunArtifactsInput,
            "get_wandb_run_logs": GetWandbRunLogsInput,
            "analyze_wandb_training_progression": AnalyzeWandbTrainingProgressionInput,
            "compare_wandb_runs": CompareWandbRunsInput,
            "analyze_wandb_learning_curves": AnalyzeWandbLearningCurvesInput,
            "identify_wandb_critical_moments": IdentifyWandbCriticalMomentsInput,
            "correlate_wandb_metrics": CorrelateWandbMetricsInput,
            "analyze_wandb_behavioral_patterns": AnalyzeWandbBehavioralPatternsInput,
            "generate_wandb_training_insights": GenerateWandbTrainingInsightsInput,
            "predict_wandb_training_outcome": PredictWandbTrainingOutcomeInput,
            "list_s3_checkpoints": ListS3CheckpointsInput,
            "get_s3_checkpoint_metadata": GetS3CheckpointMetadataInput,
            "get_s3_checkpoint_url": GetS3CheckpointUrlInput,
            "list_s3_replays": ListS3ReplaysInput,
            "check_s3_object_exists": CheckS3ObjectExistsInput,
            "list_skypilot_jobs": ListSkypilotJobsInput,
            "get_skypilot_job_status": GetSkypilotJobStatusInput,
            "get_skypilot_job_logs": GetSkypilotJobLogsInput,
            "analyze_s3_checkpoint_progression": AnalyzeS3CheckpointProgressionInput,
            "find_best_s3_checkpoint": FindBestS3CheckpointInput,
            "analyze_s3_checkpoint_usage": AnalyzeS3CheckpointUsageInput,
            "get_s3_checkpoint_statistics": GetS3CheckpointStatisticsInput,
            "compare_s3_checkpoints_across_runs": CompareS3CheckpointsAcrossRunsInput,
            "analyze_skypilot_job_performance": AnalyzeSkypilotJobPerformanceInput,
            "get_skypilot_resource_utilization": GetSkypilotResourceUtilizationInput,
            "compare_skypilot_job_configs": CompareSkypilotJobConfigsInput,
            "analyze_skypilot_job_failures": AnalyzeSkypilotJobFailuresInput,
            "get_skypilot_job_cost_estimates": GetSkypilotJobCostEstimatesInput,
            "link_wandb_run_to_s3_checkpoints": LinkWandbRunToS3CheckpointsInput,
            "link_wandb_run_to_skypilot_job": LinkWandbRunToSkypilotJobInput,
            "link_s3_checkpoint_to_wandb_run": LinkS3CheckpointToWandbRunInput,
            "link_s3_checkpoint_to_skypilot_job": LinkS3CheckpointToSkypilotJobInput,
            "link_skypilot_job_to_wandb_runs": LinkSkypilotJobToWandbRunsInput,
            "link_skypilot_job_to_s3_checkpoints": LinkSkypilotJobToS3CheckpointsInput,
        }


    def _setup_tools(self) -> None:
        """Register all MCP tools with the server."""

        @self.app.list_tools()
        async def list_tools() -> List[types.Tool]:
            tool_models = self._get_tool_input_models()
            tools = []
            for tool_name, input_model in tool_models.items():
                description = MCP_TOOL_DESCRIPTIONS.get(tool_name, "")
                if input_model is None:
                    input_schema = {"type": "object", "properties": {}, "required": []}
                else:
                    input_schema = self._pydantic_to_mcp_schema(input_model)
                tools.append(
                    types.Tool(
                        name=tool_name,
                        description=description,
                        inputSchema=input_schema,
                    )
                )
            return tools

        @self.app.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
            """Handle tool invocation requests.

            Args:
                name: Name of the tool to call
                arguments: Dictionary of tool arguments (from client)

            Returns:
                List of TextContent objects with tool results
            """
            logger.info(f"Tool called: {name} with arguments: {arguments}")

            try:
                tool_models = self._get_tool_input_models()
                input_model = tool_models.get(name)

                if input_model is not None:
                    try:
                        validated_args = input_model.model_validate(arguments)
                        arguments = validated_args.model_dump(exclude_none=True)
                    except ValidationError as e:
                        error_msg = f"Validation error: {e.errors()[0]['msg']}"
                        return [types.TextContent(type="text", text=self._create_error_response(error_msg, name))]

                if name == "get_policies":
                    result = await scorecard.get_policies(self.scorecard_client)
                    return [types.TextContent(type="text", text=result)]

                elif name == "search_policies":
                    result = await scorecard.search_policies(
                        self.scorecard_client,
                        search=arguments.get("search"),
                        policy_type=arguments.get("policy_type"),
                        tags=arguments.get("tags"),
                        user_id=arguments.get("user_id"),
                        limit=arguments.get("limit", 100),
                        offset=arguments.get("offset", 0),
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "get_eval_names":
                    training_run_ids = arguments.get("training_run_ids", [])
                    run_free_policy_ids = arguments.get("run_free_policy_ids", [])

                    if not training_run_ids and not run_free_policy_ids:
                        return [
                            types.TextContent(
                                type="text",
                                text=(
                                    '{"status": "error", "message": '
                                    '"At least one of training_run_ids or run_free_policy_ids must be provided"}'
                                ),
                            )
                        ]

                    result = await scorecard.get_eval_names(
                        self.scorecard_client,
                        training_run_ids=training_run_ids,
                        run_free_policy_ids=run_free_policy_ids,
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "get_available_metrics":
                    training_run_ids = arguments.get("training_run_ids", [])
                    run_free_policy_ids = arguments.get("run_free_policy_ids", [])
                    eval_names = arguments.get("eval_names", [])

                    if not training_run_ids and not run_free_policy_ids:
                        return [
                            types.TextContent(
                                type="text",
                                text=(
                                    '{"status": "error", "message": '
                                    '"At least one of training_run_ids or run_free_policy_ids must be provided"}'
                                ),
                            )
                        ]

                    # Backend returns empty list if eval_names is empty, so we match that behavior
                    # rather than throwing an error
                    result = await scorecard.get_available_metrics(
                        self.scorecard_client,
                        training_run_ids=training_run_ids,
                        run_free_policy_ids=run_free_policy_ids,
                        eval_names=eval_names,
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "generate_scorecard":
                    training_run_ids = arguments.get("training_run_ids", [])
                    run_free_policy_ids = arguments.get("run_free_policy_ids", [])
                    eval_names = arguments.get("eval_names", [])
                    metric = arguments.get("metric")
                    policy_selector = arguments.get("policy_selector", "best")

                    if not training_run_ids and not run_free_policy_ids:
                        return [
                            types.TextContent(
                                type="text",
                                text=(
                                    '{"status": "error", "message": '
                                    '"At least one of training_run_ids or run_free_policy_ids must be provided"}'
                                ),
                            )
                        ]

                    if not eval_names:
                        return [
                            types.TextContent(
                                type="text",
                                text='{"status": "error", "message": "eval_names is required and cannot be empty"}',
                            )
                        ]

                    if not metric:
                        return [
                            types.TextContent(type="text", text='{"status": "error", "message": "metric is required"}')
                        ]

                    result = await scorecard.generate_scorecard(
                        self.scorecard_client,
                        training_run_ids=training_run_ids,
                        run_free_policy_ids=run_free_policy_ids,
                        eval_names=eval_names,
                        metric=metric,
                        policy_selector=policy_selector,
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "run_sql_query":
                    sql = arguments.get("sql")
                    if not sql:
                        return [
                            types.TextContent(
                                type="text", text='{"status": "error", "message": "sql parameter is required"}'
                            )
                        ]

                    result = await scorecard.run_sql_query(self.scorecard_client, sql=sql)
                    return [types.TextContent(type="text", text=result)]

                elif name == "generate_ai_query":
                    description = arguments.get("description")
                    if not description:
                        return [
                            types.TextContent(
                                type="text", text='{"status": "error", "message": "description parameter is required"}'
                            )
                        ]

                    result = await scorecard.generate_ai_query(self.scorecard_client, description=description)
                    return [types.TextContent(type="text", text=result)]

                elif name == "list_wandb_runs":
                    if not self.wandb_store:
                        return [
                            types.TextContent(
                                type="text", text='{"status": "error", "message": "WandB store not initialized"}'
                            )
                        ]
                    entity = arguments.get("entity")
                    project = arguments.get("project")
                    if not entity or not project:
                        return [
                            types.TextContent(
                                type="text", text='{"status": "error", "message": "entity and project are required"}'
                            )
                        ]
                    result = await wandb.list_wandb_runs(
                        self.wandb_store,
                        entity=entity,
                        project=project,
                        tags=arguments.get("tags"),
                        state=arguments.get("state"),
                        limit=arguments.get("limit", 50),
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "get_wandb_run":
                    if not self.wandb_store:
                        return [
                            types.TextContent(
                                type="text", text='{"status": "error", "message": "WandB store not initialized"}'
                            )
                        ]
                    entity = arguments.get("entity")
                    project = arguments.get("project")
                    if not entity or not project:
                        return [
                            types.TextContent(
                                type="text", text='{"status": "error", "message": "entity and project are required"}'
                            )
                        ]
                    result = await wandb.get_wandb_run(
                        self.wandb_store,
                        entity=entity,
                        project=project,
                        run_id=arguments.get("run_id"),
                        run_name=arguments.get("run_name"),
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "get_wandb_run_metrics":
                    if not self.wandb_store:
                        return [
                            types.TextContent(
                                type="text", text='{"status": "error", "message": "WandB store not initialized"}'
                            )
                        ]
                    entity = arguments.get("entity")
                    project = arguments.get("project")
                    run_id = arguments.get("run_id")
                    metric_keys = arguments.get("metric_keys")
                    if not entity or not project or not run_id or not metric_keys:
                        return [
                            types.TextContent(
                                type="text",
                                text=(
                                    '{"status": "error", "message": '
                                    '"entity, project, run_id, and metric_keys are required"}'
                                ),
                            )
                        ]
                    result = await wandb.get_wandb_run_metrics(
                        self.wandb_store,
                        entity=entity,
                        project=project,
                        run_id=run_id,
                        metric_keys=metric_keys,
                        samples=arguments.get("samples"),
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "discover_wandb_run_metrics":
                    if not self.wandb_store:
                        return [
                            types.TextContent(
                                type="text", text='{"status": "error", "message": "WandB store not initialized"}'
                            )
                        ]
                    entity = arguments.get("entity")
                    project = arguments.get("project")
                    run_id = arguments.get("run_id")
                    if not entity or not project or not run_id:
                        return [
                            types.TextContent(
                                type="text",
                                text=('{"status": "error", "message": "entity, project, and run_id are required"}'),
                            )
                        ]
                    result = await wandb.discover_wandb_run_metrics(
                        self.wandb_store,
                        entity=entity,
                        project=project,
                        run_id=run_id,
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "get_wandb_run_artifacts":
                    if not self.wandb_api:
                        return [
                            types.TextContent(
                                type="text", text='{"status": "error", "message": "WandB client not initialized"}'
                            )
                        ]
                    entity = arguments.get("entity")
                    project = arguments.get("project")
                    run_id = arguments.get("run_id")
                    if not entity or not project or not run_id:
                        return [
                            types.TextContent(
                                type="text",
                                text=('{"status": "error", "message": "entity, project, and run_id are required"}'),
                            )
                        ]
                    result = await wandb.get_wandb_run_artifacts(
                        self.wandb_api,
                        entity=entity,
                        project=project,
                        run_id=run_id,
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "get_wandb_run_logs":
                    if not self.wandb_api:
                        return [
                            types.TextContent(
                                type="text", text='{"status": "error", "message": "WandB client not initialized"}'
                            )
                        ]
                    entity = arguments.get("entity")
                    project = arguments.get("project")
                    run_id = arguments.get("run_id")
                    if not entity or not project or not run_id:
                        return [
                            types.TextContent(
                                type="text",
                                text=('{"status": "error", "message": "entity, project, and run_id are required"}'),
                            )
                        ]
                    result = await wandb.get_wandb_run_logs(
                        self.wandb_api,
                        entity=entity,
                        project=project,
                        run_id=run_id,
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "analyze_wandb_training_progression":
                    if not self.wandb_store:
                        return [
                            types.TextContent(
                                type="text", text='{"status": "error", "message": "WandB store not initialized"}'
                            )
                        ]
                    entity = arguments.get("entity")
                    project = arguments.get("project")
                    run_id = arguments.get("run_id")
                    metric_keys = arguments.get("metric_keys")
                    if not entity or not project or not run_id or not metric_keys:
                        return [
                            types.TextContent(
                                type="text",
                                text=(
                                    '{"status": "error", "message": '
                                    '"entity, project, run_id, and metric_keys are required"}'
                                ),
                            )
                        ]
                    result = await wandb.analyze_wandb_training_progression(
                        self.wandb_store,
                        entity=entity,
                        project=project,
                        run_id=run_id,
                        metric_keys=metric_keys,
                        context_window_steps=arguments.get("context_window_steps", 1000),
                        center_step=arguments.get("center_step"),
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "compare_wandb_runs":
                    if not self.wandb_store:
                        return [
                            types.TextContent(
                                type="text", text='{"status": "error", "message": "WandB store not initialized"}'
                            )
                        ]
                    entity = arguments.get("entity")
                    project = arguments.get("project")
                    run_ids = arguments.get("run_ids")
                    metric_keys = arguments.get("metric_keys")
                    if not entity or not project or not run_ids or not metric_keys:
                        return [
                            types.TextContent(
                                type="text",
                                text=(
                                    '{"status": "error", "message": '
                                    '"entity, project, run_ids, and metric_keys are required"}'
                                ),
                            )
                        ]
                    result = await wandb.compare_wandb_runs(
                        self.wandb_store,
                        entity=entity,
                        project=project,
                        run_ids=run_ids,
                        metric_keys=metric_keys,
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "analyze_wandb_learning_curves":
                    if not self.wandb_store:
                        return [
                            types.TextContent(
                                type="text", text='{"status": "error", "message": "WandB store not initialized"}'
                            )
                        ]
                    entity = arguments.get("entity")
                    project = arguments.get("project")
                    run_id = arguments.get("run_id")
                    metric_keys = arguments.get("metric_keys")
                    if not entity or not project or not run_id or not metric_keys:
                        return [
                            types.TextContent(
                                type="text",
                                text=(
                                    '{"status": "error", "message": '
                                    '"entity, project, run_id, and metric_keys are required"}'
                                ),
                            )
                        ]
                    result = await wandb.analyze_wandb_learning_curves(
                        self.wandb_store,
                        entity=entity,
                        project=project,
                        run_id=run_id,
                        metric_keys=metric_keys,
                        smoothing_window=arguments.get("smoothing_window", 10),
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "identify_wandb_critical_moments":
                    if not self.wandb_store:
                        return [
                            types.TextContent(
                                type="text", text='{"status": "error", "message": "WandB store not initialized"}'
                            )
                        ]
                    entity = arguments.get("entity")
                    project = arguments.get("project")
                    run_id = arguments.get("run_id")
                    metric_keys = arguments.get("metric_keys")
                    if not entity or not project or not run_id or not metric_keys:
                        return [
                            types.TextContent(
                                type="text",
                                text=(
                                    '{"status": "error", "message": '
                                    '"entity, project, run_id, and metric_keys are required"}'
                                ),
                            )
                        ]
                    result = await wandb.identify_wandb_critical_moments(
                        self.wandb_store,
                        entity=entity,
                        project=project,
                        run_id=run_id,
                        metric_keys=metric_keys,
                        threshold=arguments.get("threshold", 0.1),
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "correlate_wandb_metrics":
                    if not self.wandb_store:
                        return [
                            types.TextContent(
                                type="text", text='{"status": "error", "message": "WandB store not initialized"}'
                            )
                        ]
                    entity = arguments.get("entity")
                    project = arguments.get("project")
                    run_id = arguments.get("run_id")
                    metric_pairs = arguments.get("metric_pairs")
                    if not entity or not project or not run_id or not metric_pairs:
                        return [
                            types.TextContent(
                                type="text",
                                text=(
                                    '{"status": "error", "message": '
                                    '"entity, project, run_id, and metric_pairs are required"}'
                                ),
                            )
                        ]
                    result = await wandb.correlate_wandb_metrics(
                        self.wandb_store,
                        entity=entity,
                        project=project,
                        run_id=run_id,
                        metric_pairs=metric_pairs,
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "analyze_wandb_behavioral_patterns":
                    if not self.wandb_store:
                        return [
                            types.TextContent(
                                type="text", text='{"status": "error", "message": "WandB store not initialized"}'
                            )
                        ]
                    entity = arguments.get("entity")
                    project = arguments.get("project")
                    run_id = arguments.get("run_id")
                    if not entity or not project or not run_id:
                        return [
                            types.TextContent(
                                type="text",
                                text=('{"status": "error", "message": "entity, project, and run_id are required"}'),
                            )
                        ]
                    result = await wandb.analyze_wandb_behavioral_patterns(
                        self.wandb_store,
                        entity=entity,
                        project=project,
                        run_id=run_id,
                        behavior_categories=arguments.get("behavior_categories"),
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "generate_wandb_training_insights":
                    if not self.wandb_store:
                        return [
                            types.TextContent(
                                type="text", text='{"status": "error", "message": "WandB store not initialized"}'
                            )
                        ]
                    entity = arguments.get("entity")
                    project = arguments.get("project")
                    run_id = arguments.get("run_id")
                    if not entity or not project or not run_id:
                        return [
                            types.TextContent(
                                type="text",
                                text=('{"status": "error", "message": "entity, project, and run_id are required"}'),
                            )
                        ]
                    result = await wandb.generate_wandb_training_insights(
                        self.wandb_store,
                        entity=entity,
                        project=project,
                        run_id=run_id,
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "predict_wandb_training_outcome":
                    if not self.wandb_store:
                        return [
                            types.TextContent(
                                type="text", text='{"status": "error", "message": "WandB store not initialized"}'
                            )
                        ]
                    entity = arguments.get("entity")
                    project = arguments.get("project")
                    run_id = arguments.get("run_id")
                    target_metric = arguments.get("target_metric")
                    if not entity or not project or not run_id or not target_metric:
                        return [
                            types.TextContent(
                                type="text",
                                text=(
                                    '{"status": "error", "message": '
                                    '"entity, project, run_id, and target_metric are required"}'
                                ),
                            )
                        ]
                    result = await wandb.predict_wandb_training_outcome(
                        self.wandb_store,
                        entity=entity,
                        project=project,
                        run_id=run_id,
                        target_metric=target_metric,
                        projection_steps=arguments.get("projection_steps", 1000),
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "list_s3_checkpoints":
                    if not self.s3_store:
                        return [
                            types.TextContent(
                                type="text", text='{"status": "error", "message": "S3 store not initialized"}'
                            )
                        ]
                    result = await s3.list_s3_checkpoints(
                        self.s3_store,
                        run_name=arguments.get("run_name"),
                        prefix=arguments.get("prefix"),
                        max_keys=arguments.get("max_keys", 1000),
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "get_s3_checkpoint_metadata":
                    if not self.s3_store:
                        return [
                            types.TextContent(
                                type="text", text='{"status": "error", "message": "S3 store not initialized"}'
                            )
                        ]
                    key = arguments.get("key")
                    if not key:
                        return [
                            types.TextContent(
                                type="text", text='{"status": "error", "message": "key parameter is required"}'
                            )
                        ]
                    result = await s3.get_s3_checkpoint_metadata(self.s3_store, key=key)
                    return [types.TextContent(type="text", text=result)]

                elif name == "get_s3_checkpoint_url":
                    if not self.s3_store:
                        return [
                            types.TextContent(
                                type="text", text='{"status": "error", "message": "S3 store not initialized"}'
                            )
                        ]
                    key = arguments.get("key")
                    if not key:
                        return [
                            types.TextContent(
                                type="text", text='{"status": "error", "message": "key parameter is required"}'
                            )
                        ]
                    result = await s3.get_s3_checkpoint_url(
                        self.s3_store,
                        key=key,
                        expires_in=arguments.get("expires_in", 3600),
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "list_s3_replays":
                    if not self.s3_store:
                        return [
                            types.TextContent(
                                type="text", text='{"status": "error", "message": "S3 store not initialized"}'
                            )
                        ]
                    result = await s3.list_s3_replays(
                        self.s3_store,
                        run_name=arguments.get("run_name"),
                        prefix=arguments.get("prefix"),
                        max_keys=arguments.get("max_keys", 1000),
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "check_s3_object_exists":
                    if not self.s3_client:
                        return [
                            types.TextContent(
                                type="text", text='{"status": "error", "message": "S3 client not initialized"}'
                            )
                        ]
                    key = arguments.get("key")
                    if not key:
                        return [
                            types.TextContent(
                                type="text", text='{"status": "error", "message": "key parameter is required"}'
                            )
                        ]
                    result = await s3.check_s3_object_exists(self.s3_client, self.s3_bucket, key=key)
                    return [types.TextContent(type="text", text=result)]

                elif name == "list_skypilot_jobs":
                    result = await skypilot.list_skypilot_jobs(
                        status=arguments.get("status"),
                        limit=arguments.get("limit", 100),
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "get_skypilot_job_status":
                    job_id = arguments.get("job_id")
                    if not job_id:
                        return [
                            types.TextContent(
                                type="text", text='{"status": "error", "message": "job_id parameter is required"}'
                            )
                        ]
                    result = await skypilot.get_skypilot_job_status(job_id=job_id)
                    return [types.TextContent(type="text", text=result)]

                elif name == "get_skypilot_job_logs":
                    job_id = arguments.get("job_id")
                    if not job_id:
                        return [
                            types.TextContent(
                                type="text", text='{"status": "error", "message": "job_id parameter is required"}'
                            )
                        ]
                    result = await skypilot.get_skypilot_job_logs(
                        job_id=job_id,
                        tail_lines=arguments.get("tail_lines", 100),
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "analyze_s3_checkpoint_progression":
                    if not self.s3_store:
                        return [
                            types.TextContent(
                                type="text", text='{"status": "error", "message": "S3 store not initialized"}'
                            )
                        ]
                    run_name = arguments.get("run_name")
                    if not run_name:
                        return [
                            types.TextContent(
                                type="text", text='{"status": "error", "message": "run_name parameter is required"}'
                            )
                        ]
                    result = await s3.analyze_s3_checkpoint_progression(
                        self.s3_store,
                        run_name=run_name,
                        prefix=arguments.get("prefix"),
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "find_best_s3_checkpoint":
                    if not self.s3_store:
                        return [
                            types.TextContent(
                                type="text", text='{"status": "error", "message": "S3 store not initialized"}'
                            )
                        ]
                    run_name = arguments.get("run_name")
                    if not run_name:
                        return [
                            types.TextContent(
                                type="text", text='{"status": "error", "message": "run_name parameter is required"}'
                            )
                        ]
                    result = await s3.find_best_s3_checkpoint(
                        self.s3_store,
                        run_name=run_name,
                        criteria=arguments.get("criteria", "latest"),
                        prefix=arguments.get("prefix"),
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "analyze_s3_checkpoint_usage":
                    if not self.s3_store:
                        return [
                            types.TextContent(
                                type="text", text='{"status": "error", "message": "S3 store not initialized"}'
                            )
                        ]
                    result = await s3.analyze_s3_checkpoint_usage(
                        self.s3_store,
                        run_name=arguments.get("run_name"),
                        prefix=arguments.get("prefix"),
                        time_window_days=arguments.get("time_window_days", 30),
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "get_s3_checkpoint_statistics":
                    if not self.s3_store:
                        return [
                            types.TextContent(
                                type="text", text='{"status": "error", "message": "S3 store not initialized"}'
                            )
                        ]
                    result = await s3.get_s3_checkpoint_statistics(
                        self.s3_store,
                        run_name=arguments.get("run_name"),
                        prefix=arguments.get("prefix"),
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "compare_s3_checkpoints_across_runs":
                    if not self.s3_store:
                        return [
                            types.TextContent(
                                type="text", text='{"status": "error", "message": "S3 store not initialized"}'
                            )
                        ]
                    run_names = arguments.get("run_names")
                    if not run_names:
                        return [
                            types.TextContent(
                                type="text", text='{"status": "error", "message": "run_names parameter is required"}'
                            )
                        ]
                    result = await s3.compare_s3_checkpoints_across_runs(
                        self.s3_store,
                        run_names=run_names,
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "analyze_skypilot_job_performance":
                    result = await skypilot.analyze_skypilot_job_performance(
                        limit=arguments.get("limit", 100),
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "get_skypilot_resource_utilization":
                    result = await skypilot.get_skypilot_resource_utilization(
                        limit=arguments.get("limit", 100),
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "compare_skypilot_job_configs":
                    job_ids = arguments.get("job_ids")
                    if not job_ids:
                        return [
                            types.TextContent(
                                type="text", text='{"status": "error", "message": "job_ids parameter is required"}'
                            )
                        ]
                    result = await skypilot.compare_skypilot_job_configs(
                        job_ids=job_ids,
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "analyze_skypilot_job_failures":
                    result = await skypilot.analyze_skypilot_job_failures(
                        limit=arguments.get("limit", 100),
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "get_skypilot_job_cost_estimates":
                    result = await skypilot.get_skypilot_job_cost_estimates(
                        limit=arguments.get("limit", 100),
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "link_wandb_run_to_s3_checkpoints":
                    if not self.wandb_api or not self.s3_client:
                        return [
                            types.TextContent(
                                type="text",
                                text=('{"status": "error", "message": "WandB and S3 clients must be initialized"}'),
                            )
                        ]
                    entity = arguments.get("entity")
                    project = arguments.get("project")
                    run_id = arguments.get("run_id")
                    if not entity or not project or not run_id:
                        return [
                            types.TextContent(
                                type="text",
                                text=('{"status": "error", "message": "entity, project, and run_id are required"}'),
                            )
                        ]
                    result = await wandb.link_wandb_run_to_s3_checkpoints(
                        self.wandb_api,
                        self.s3_client,
                        self.s3_bucket,
                        entity=entity,
                        project=project,
                        run_id=run_id,
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "link_wandb_run_to_skypilot_job":
                    if not self.wandb_api:
                        return [
                            types.TextContent(
                                type="text", text='{"status": "error", "message": "WandB client not initialized"}'
                            )
                        ]
                    entity = arguments.get("entity")
                    project = arguments.get("project")
                    run_id = arguments.get("run_id")
                    if not entity or not project or not run_id:
                        return [
                            types.TextContent(
                                type="text",
                                text=('{"status": "error", "message": "entity, project, and run_id are required"}'),
                            )
                        ]
                    result = await wandb.link_wandb_run_to_skypilot_job(
                        self.wandb_api,
                        entity=entity,
                        project=project,
                        run_id=run_id,
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "link_s3_checkpoint_to_wandb_run":
                    if not self.s3_client or not self.wandb_api:
                        return [
                            types.TextContent(
                                type="text",
                                text=('{"status": "error", "message": "S3 and WandB clients must be initialized"}'),
                            )
                        ]
                    key = arguments.get("key")
                    entity = arguments.get("entity")
                    project = arguments.get("project")
                    if not key or not entity or not project:
                        return [
                            types.TextContent(
                                type="text",
                                text=('{"status": "error", "message": "key, entity, and project are required"}'),
                            )
                        ]
                    result = await s3.link_s3_checkpoint_to_wandb_run(
                        self.s3_client,
                        self.s3_bucket,
                        self.wandb_api,
                        key=key,
                        entity=entity,
                        project=project,
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "link_s3_checkpoint_to_skypilot_job":
                    if not self.s3_client:
                        return [
                            types.TextContent(
                                type="text", text='{"status": "error", "message": "S3 client not initialized"}'
                            )
                        ]
                    key = arguments.get("key")
                    if not key:
                        return [
                            types.TextContent(
                                type="text", text='{"status": "error", "message": "key parameter is required"}'
                            )
                        ]
                    result = await s3.link_s3_checkpoint_to_skypilot_job(
                        self.s3_client,
                        self.s3_bucket,
                        key=key,
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "link_skypilot_job_to_wandb_runs":
                    if not self.wandb_store:
                        return [
                            types.TextContent(
                                type="text", text='{"status": "error", "message": "WandB store not initialized"}'
                            )
                        ]
                    job_id = arguments.get("job_id")
                    entity = arguments.get("entity")
                    project = arguments.get("project")
                    if not job_id or not entity or not project:
                        return [
                            types.TextContent(
                                type="text",
                                text=('{"status": "error", "message": "job_id, entity, and project are required"}'),
                            )
                        ]
                    result = await skypilot.link_skypilot_job_to_wandb_runs(
                        self.wandb_store,
                        job_id=job_id,
                        entity=entity,
                        project=project,
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "link_skypilot_job_to_s3_checkpoints":
                    if not self.s3_store:
                        return [
                            types.TextContent(
                                type="text", text='{"status": "error", "message": "S3 store not initialized"}'
                            )
                        ]
                    job_id = arguments.get("job_id")
                    if not job_id:
                        return [
                            types.TextContent(
                                type="text", text='{"status": "error", "message": "job_id parameter is required"}'
                            )
                        ]
                    result = await skypilot.link_skypilot_job_to_s3_checkpoints(
                        self.s3_store,
                        job_id=job_id,
                    )
                    return [types.TextContent(type="text", text=result)]

                else:
                    return [
                        types.TextContent(type="text", text=f'{{"status": "error", "message": "Unknown tool: {name}"}}')
                    ]

            except Exception as e:
                logger.error(f"Error calling tool {name}: {e}", exc_info=True)
                return [
                    types.TextContent(
                        type="text", text=f'{{"status": "error", "tool": "{name}", "message": "{str(e)}"}}'
                    )
                ]

    def _setup_resources(self) -> None:
        """Register MCP resources with the server."""

        @self.app.list_resources()
        async def list_resources() -> List[types.Resource]:
            return []

        @self.app.read_resource()
        async def read_resource(uri: str) -> str:
            raise ValueError(f"Unknown resource URI: {uri}")


async def main() -> None:
    """Main entry point for the Observatory MCP Server."""
    config = ObservatoryMCPConfig.from_env()
    log_level_str = config.log_level.upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    logging.basicConfig(
        level=log_level,
        format=config.log_format,
        handlers=[logging.StreamHandler(sys.stderr)],
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Starting Observatory MCP Server... (log_level={log_level_str})")

    server: ObservatoryMCPServer | None = None
    try:
        server = ObservatoryMCPServer()
    except Exception as e:
        logger.error(f"Failed to initialize server: {e}", exc_info=True)
        sys.exit(1)

    try:
        async with stdio_server() as (read_stream, write_stream):
            logger.info("Entering MCP stdio server loop...")
            await server.app.run(read_stream, write_stream, server.app.create_initialization_options())
    except KeyboardInterrupt:
        logger.info("Server stopped by user (KeyboardInterrupt)")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if server is not None:
            try:
                await server.scorecard_client.close()
            except Exception as e:
                logger.warning(f"Error closing scorecard client: {e}")
        logger.info("Observatory MCP Server shutdown complete")


def cli_main() -> None:
    """CLI entry point for the server."""
    asyncio.run(main())


if __name__ == "__main__":
    cli_main()
