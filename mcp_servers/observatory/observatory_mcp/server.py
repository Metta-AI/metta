"""Observatory MCP Server - Model Context Protocol server for Metta Observatory backend."""

import asyncio
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import boto3
import mcp.types as types
from botocore.exceptions import ClientError, NoCredentialsError, ProfileNotFound
from mcp.server import Server
from mcp.server.stdio import stdio_server

try:
    import wandb
    from wandb import Api
except ImportError:
    wandb = None
    Api = None

from pydantic import BaseModel, ValidationError

from metta.adaptive.stores.wandb import WandbStore
from metta.app_backend.clients.stats_client import StatsClient
from metta.app_backend.routes.stats_routes import (
    EvalsRequest,
    MetricsRequest,
    PoliciesSearchRequest,
)
from metta.app_backend.routes.sql_routes import AIQueryRequest
from metta.utils.s3 import S3Store

from .descriptions import MCP_TOOL_DESCRIPTIONS
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
from .tools import s3, skypilot
from .tools import wandb as wandb_tools
from .utils import (
    generate_ai_query,
    generate_scorecard,
    get_available_metrics,
    get_eval_names,
    get_policies,
    get_training_runs,
    run_sql_query,
    search_policies,
)

logger = logging.getLogger(__name__)


@dataclass
class ObservatoryMCPConfig:
    """Configuration for the Observatory MCP Server."""

    # Server configuration
    server_name: str = "observatory-mcp"
    version: str = "0.1.0"

    # Backend API configuration
    backend_url: str = field(default_factory=lambda: os.getenv("METTA_MCP_BACKEND_URL", "http://localhost:8000"))

    # Authentication configuration
    machine_token: Optional[str] = field(default_factory=lambda: os.getenv("METTA_MCP_MACHINE_TOKEN"))

    # AWS configuration
    aws_profile: Optional[str] = field(default_factory=lambda: os.getenv("AWS_PROFILE"))
    s3_bucket: str = field(default_factory=lambda: os.getenv("METTA_S3_BUCKET", "softmax-public"))

    # W&B configuration
    wandb_api_key: Optional[str] = field(default_factory=lambda: os.getenv("WANDB_API_KEY"))
    wandb_entity: Optional[str] = field(default_factory=lambda: os.getenv("WANDB_ENTITY"))
    wandb_project: Optional[str] = field(default_factory=lambda: os.getenv("WANDB_PROJECT"))

    # Skypilot configuration
    skypilot_url: Optional[str] = field(default_factory=lambda: os.getenv("METTA_SKYPILOT_URL"))

    # Logging configuration
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    @classmethod
    def from_env(cls) -> "ObservatoryMCPConfig":
        """Create configuration from environment variables."""
        return cls(
            backend_url=os.getenv("METTA_MCP_BACKEND_URL", "http://localhost:8000"),
            machine_token=os.getenv("METTA_MCP_MACHINE_TOKEN"),
            aws_profile=os.getenv("AWS_PROFILE"),
            s3_bucket=os.getenv("METTA_S3_BUCKET", "softmax-public"),
            wandb_api_key=os.getenv("WANDB_API_KEY"),
            wandb_entity=os.getenv("WANDB_ENTITY"),
            wandb_project=os.getenv("WANDB_PROJECT"),
            skypilot_url=os.getenv("METTA_SKYPILOT_URL"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )

    def validate(self) -> List[str]:
        """Validate the configuration and return any errors."""
        errors = []
        if not self.backend_url:
            errors.append("METTA_MCP_BACKEND_URL is required but not set.")
        else:
            try:
                parsed = urlparse(self.backend_url)
                if not parsed.scheme or not parsed.netloc:
                    errors.append(
                        f"Invalid backend URL format: {self.backend_url}. "
                        "Expected format: http://host:port or https://host:port"
                    )
            except Exception as e:
                errors.append(f"Error parsing backend URL: {e}")

        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level.upper() not in valid_log_levels:
            errors.append(f"Invalid log level: {self.log_level}. Must be one of: {', '.join(valid_log_levels)}")

        if self.s3_bucket:
            if not (3 <= len(self.s3_bucket) <= 63):
                errors.append(
                    f"Invalid S3 bucket name: {self.s3_bucket}. Bucket names must be between 3 and 63 characters."
                )

        return errors

    def is_backend_configured(self) -> bool:
        """Check if backend is properly configured."""
        return bool(self.backend_url) and len(self.validate()) == 0

    def is_aws_configured(self) -> bool:
        """Check if AWS is configured (always returns True to allow boto3 to attempt initialization)."""
        return True  # Always try - boto3 can use default credentials or profile

    def is_wandb_configured(self) -> bool:
        """Check if W&B is configured."""
        return bool(self.wandb_api_key)

    def is_authenticated(self) -> bool:
        """Check if authentication token is available."""
        return bool(self.machine_token)


class ObservatoryMCPServer:
    """MCP Server that exposes Observatory backend functionality."""

    def __init__(self, server_name: str = "observatory-mcp", version: str = "0.1.0"):
        """Initialize the Observatory MCP Server."""
        self.app = Server(server_name)
        self.version = version
        self.config = ObservatoryMCPConfig.from_env()

        config_errors = self.config.validate()
        if config_errors:
            logger.warning("Configuration validation errors:")
            for error in config_errors:
                logger.warning(f"  - {error}")

        self.stats_client = StatsClient(
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

                # Initialize WandB API client for artifact/log tools
                try:
                    self.wandb_api = Api()
                    logger.info("WandB API client initialized")
                except Exception as e:
                    logger.warning(f"Failed to initialize WandB API client: {e}")
                    self.wandb_api = None
            except Exception as e:
                logger.warning(f"Failed to initialize WandB store: {e}. WandB tools will be unavailable.")
                self.wandb_store = None
                self.wandb_api = None
        else:
            logger.debug("WandB not configured. WandB tools will be unavailable.")
            self.wandb_api = None

        # Initialize S3 client and store
        self.s3_client = None
        self.s3_bucket = self.config.s3_bucket
        self.s3_store: S3Store | None = None
        # Always try to initialize S3 client - it can use profile or default credentials
        try:
            if self.config.aws_profile:
                try:
                    session = boto3.Session(profile_name=self.config.aws_profile)
                    self.s3_client = session.client("s3")
                    logger.info(f"S3 client initialized (profile={self.config.aws_profile}, bucket={self.s3_bucket})")
                except ProfileNotFound:
                    logger.warning(f"AWS profile '{self.config.aws_profile}' not found. Trying default credentials...")
                    self.s3_client = boto3.client("s3")
                    logger.info(f"S3 client initialized with default credentials (bucket={self.s3_bucket})")
            else:
                self.s3_client = boto3.client("s3")
                logger.info(f"S3 client initialized (bucket={self.s3_bucket})")

            # Create S3 store instance
            self.s3_store = S3Store(self.s3_client, self.s3_bucket)
        except (NoCredentialsError, ClientError, ProfileNotFound) as e:
            logger.warning(f"Failed to initialize S3 client: {e}. S3 tools will be unavailable.")
            self.s3_client = None
            self.s3_store = None

        # Skypilot doesn't need a client - we'll use subprocess directly
        # Store the URL for reference if needed
        if self.config.skypilot_url:
            logger.info(f"Skypilot URL configured: {self.config.skypilot_url}")

        self._tool_handlers = self._setup_tool_handlers()
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
        """Convert Pydantic model to MCP inputSchema format."""
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
        """Create error response JSON string."""
        return ErrorResponse(tool=tool_name, message=message).model_dump_json(indent=2, exclude_none=True)

    def _get_tool_input_models(self) -> Dict[str, type[BaseModel] | None]:
        """Get mapping of tool names to their input models."""
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

    def _setup_tool_handlers(self) -> Dict[str, Any]:
        """Create dispatch dictionary mapping tool names to handler methods."""
        return {
            "get_training_runs": self._handle_get_training_runs,
            "get_policies": self._handle_get_policies,
            "search_policies": self._handle_search_policies,
            "get_eval_names": self._handle_get_eval_names,
            "get_available_metrics": self._handle_get_available_metrics,
            "generate_scorecard": self._handle_generate_scorecard,
            "run_sql_query": self._handle_run_sql_query,
            "generate_ai_query": self._handle_generate_ai_query,
            "list_wandb_runs": self._handle_list_wandb_runs,
            "get_wandb_run": self._handle_get_wandb_run,
            "get_wandb_run_metrics": self._handle_get_wandb_run_metrics,
            "discover_wandb_run_metrics": self._handle_discover_wandb_run_metrics,
            "get_wandb_run_artifacts": self._handle_get_wandb_run_artifacts,
            "get_wandb_run_logs": self._handle_get_wandb_run_logs,
            "analyze_wandb_training_progression": self._handle_analyze_wandb_training_progression,
            "compare_wandb_runs": self._handle_compare_wandb_runs,
            "analyze_wandb_learning_curves": self._handle_analyze_wandb_learning_curves,
            "identify_wandb_critical_moments": self._handle_identify_wandb_critical_moments,
            "correlate_wandb_metrics": self._handle_correlate_wandb_metrics,
            "analyze_wandb_behavioral_patterns": self._handle_analyze_wandb_behavioral_patterns,
            "generate_wandb_training_insights": self._handle_generate_wandb_training_insights,
            "predict_wandb_training_outcome": self._handle_predict_wandb_training_outcome,
            "list_s3_checkpoints": self._handle_list_s3_checkpoints,
            "get_s3_checkpoint_metadata": self._handle_get_s3_checkpoint_metadata,
            "get_s3_checkpoint_url": self._handle_get_s3_checkpoint_url,
            "list_s3_replays": self._handle_list_s3_replays,
            "check_s3_object_exists": self._handle_check_s3_object_exists,
            "analyze_s3_checkpoint_progression": self._handle_analyze_s3_checkpoint_progression,
            "find_best_s3_checkpoint": self._handle_find_best_s3_checkpoint,
            "analyze_s3_checkpoint_usage": self._handle_analyze_s3_checkpoint_usage,
            "get_s3_checkpoint_statistics": self._handle_get_s3_checkpoint_statistics,
            "compare_s3_checkpoints_across_runs": self._handle_compare_s3_checkpoints_across_runs,
            "list_skypilot_jobs": self._handle_list_skypilot_jobs,
            "get_skypilot_job_status": self._handle_get_skypilot_job_status,
            "get_skypilot_job_logs": self._handle_get_skypilot_job_logs,
            "analyze_skypilot_job_performance": self._handle_analyze_skypilot_job_performance,
            "get_skypilot_resource_utilization": self._handle_get_skypilot_resource_utilization,
            "compare_skypilot_job_configs": self._handle_compare_skypilot_job_configs,
            "analyze_skypilot_job_failures": self._handle_analyze_skypilot_job_failures,
            "get_skypilot_job_cost_estimates": self._handle_get_skypilot_job_cost_estimates,
            "link_wandb_run_to_s3_checkpoints": self._handle_link_wandb_run_to_s3_checkpoints,
            "link_wandb_run_to_skypilot_job": self._handle_link_wandb_run_to_skypilot_job,
            "link_s3_checkpoint_to_wandb_run": self._handle_link_s3_checkpoint_to_wandb_run,
            "link_s3_checkpoint_to_skypilot_job": self._handle_link_s3_checkpoint_to_skypilot_job,
            "link_skypilot_job_to_wandb_runs": self._handle_link_skypilot_job_to_wandb_runs,
            "link_skypilot_job_to_s3_checkpoints": self._handle_link_skypilot_job_to_s3_checkpoints,
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
            """Handle tool invocation requests."""
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

                handler = self._tool_handlers.get(name)
                if handler:
                    result = await handler(arguments)
                    return [types.TextContent(type="text", text=result)]
                else:
                    return [
                        types.TextContent(type="text", text=self._create_error_response(f"Unknown tool: {name}", name))
                    ]

            except Exception as e:
                logger.error(f"Error calling tool {name}: {e}", exc_info=True)
                return [
                    types.TextContent(
                        type="text", text=f'{{"status": "error", "tool": "{name}", "message": "{str(e)}"}}'
                    )
                ]

    async def _handle_get_training_runs(self, arguments: Dict[str, Any]) -> str:
        """Handle get_training_runs tool."""
        return await get_training_runs(self.stats_client)

    async def _handle_get_policies(self, arguments: Dict[str, Any]) -> str:
        """Handle get_policies tool."""
        return await get_policies(self.stats_client)

    async def _handle_search_policies(self, arguments: Dict[str, Any]) -> str:
        """Handle search_policies tool."""
        return await search_policies(
            self.stats_client,
            search=arguments.get("search"),
            policy_type=arguments.get("policy_type"),
            tags=arguments.get("tags"),
            user_id=arguments.get("user_id"),
            limit=arguments.get("limit", 100),
            offset=arguments.get("offset", 0),
        )

    async def _handle_get_eval_names(self, arguments: Dict[str, Any]) -> str:
        """Handle get_eval_names tool."""
        training_run_ids = arguments.get("training_run_ids", [])
        run_free_policy_ids = arguments.get("run_free_policy_ids", [])
        if not training_run_ids and not run_free_policy_ids:
            return self._create_error_response(
                "At least one of training_run_ids or run_free_policy_ids must be provided", "get_eval_names"
            )
        return await get_eval_names(
            self.stats_client,
            training_run_ids=training_run_ids,
            run_free_policy_ids=run_free_policy_ids,
        )

    async def _handle_get_available_metrics(self, arguments: Dict[str, Any]) -> str:
        """Handle get_available_metrics tool."""
        training_run_ids = arguments.get("training_run_ids", [])
        run_free_policy_ids = arguments.get("run_free_policy_ids", [])
        eval_names = arguments.get("eval_names", [])
        if not training_run_ids and not run_free_policy_ids:
            return self._create_error_response(
                "At least one of training_run_ids or run_free_policy_ids must be provided", "get_available_metrics"
            )
        return await get_available_metrics(
            self.stats_client,
            training_run_ids=training_run_ids,
            run_free_policy_ids=run_free_policy_ids,
            eval_names=eval_names,
        )

    async def _handle_generate_scorecard(self, arguments: Dict[str, Any]) -> str:
        """Handle generate_scorecard tool."""
        training_run_ids = arguments.get("training_run_ids", [])
        run_free_policy_ids = arguments.get("run_free_policy_ids", [])
        eval_names = arguments.get("eval_names", [])
        metric = arguments.get("metric")
        policy_selector = arguments.get("policy_selector", "best")
        if not training_run_ids and not run_free_policy_ids:
            return self._create_error_response(
                "At least one of training_run_ids or run_free_policy_ids must be provided", "generate_scorecard"
            )
        if not eval_names:
            return self._create_error_response("eval_names is required and cannot be empty", "generate_scorecard")
        if not metric:
            return self._create_error_response("metric is required", "generate_scorecard")
        return await generate_scorecard(
            self.stats_client,
            training_run_ids=training_run_ids,
            run_free_policy_ids=run_free_policy_ids,
            eval_names=eval_names,
            metric=metric,
            policy_selector=policy_selector,
        )

    async def _handle_run_sql_query(self, arguments: Dict[str, Any]) -> str:
        """Handle run_sql_query tool."""
        sql = arguments.get("sql")
        if not sql:
            return self._create_error_response("sql parameter is required", "run_sql_query")
        return await run_sql_query(self.stats_client, sql=sql)

    async def _handle_generate_ai_query(self, arguments: Dict[str, Any]) -> str:
        """Handle generate_ai_query tool."""
        description = arguments.get("description")
        if not description:
            return self._create_error_response("description parameter is required", "generate_ai_query")
        return await generate_ai_query(self.stats_client, description=description)

    async def _handle_list_wandb_runs(self, arguments: Dict[str, Any]) -> str:
        """Handle list_wandb_runs tool."""
        if not self.wandb_store:
            return self._create_error_response("WandB store not initialized", "list_wandb_runs")
        entity = arguments.get("entity")
        project = arguments.get("project")
        if not entity or not project:
            return self._create_error_response("entity and project are required", "list_wandb_runs")
        return await wandb_tools.list_wandb_runs(
            self.wandb_store,
            entity=entity,
            project=project,
            tags=arguments.get("tags"),
            state=arguments.get("state"),
            limit=arguments.get("limit", 50),
        )

    async def _handle_get_wandb_run(self, arguments: Dict[str, Any]) -> str:
        """Handle get_wandb_run tool."""
        if not self.wandb_store:
            return self._create_error_response("WandB store not initialized", "get_wandb_run")
        entity = arguments.get("entity")
        project = arguments.get("project")
        if not entity or not project:
            return self._create_error_response("entity and project are required", "get_wandb_run")
        return await wandb_tools.get_wandb_run(
            self.wandb_store,
            entity=entity,
            project=project,
            run_id=arguments.get("run_id"),
            run_name=arguments.get("run_name"),
        )

    async def _handle_get_wandb_run_metrics(self, arguments: Dict[str, Any]) -> str:
        """Handle get_wandb_run_metrics tool."""
        if not self.wandb_store:
            return self._create_error_response("WandB store not initialized", "get_wandb_run_metrics")
        entity = arguments.get("entity")
        project = arguments.get("project")
        run_id = arguments.get("run_id")
        metric_keys = arguments.get("metric_keys")
        if not entity or not project or not run_id or not metric_keys:
            return self._create_error_response(
                "entity, project, run_id, and metric_keys are required", "get_wandb_run_metrics"
            )
        return await wandb_tools.get_wandb_run_metrics(
            self.wandb_store,
            entity=entity,
            project=project,
            run_id=run_id,
            metric_keys=metric_keys,
            samples=arguments.get("samples"),
        )

    async def _handle_discover_wandb_run_metrics(self, arguments: Dict[str, Any]) -> str:
        """Handle discover_wandb_run_metrics tool."""
        if not self.wandb_store:
            return self._create_error_response("WandB store not initialized", "discover_wandb_run_metrics")
        entity = arguments.get("entity")
        project = arguments.get("project")
        run_id = arguments.get("run_id")
        if not entity or not project or not run_id:
            return self._create_error_response("entity, project, and run_id are required", "discover_wandb_run_metrics")
        return await wandb_tools.discover_wandb_run_metrics(
            self.wandb_store,
            entity=entity,
            project=project,
            run_id=run_id,
        )

    async def _handle_get_wandb_run_artifacts(self, arguments: Dict[str, Any]) -> str:
        """Handle get_wandb_run_artifacts tool."""
        if not self.wandb_api:
            return self._create_error_response("WandB client not initialized", "get_wandb_run_artifacts")
        entity = arguments.get("entity")
        project = arguments.get("project")
        run_id = arguments.get("run_id")
        if not entity or not project or not run_id:
            return self._create_error_response("entity, project, and run_id are required", "get_wandb_run_artifacts")
        return await wandb_tools.get_wandb_run_artifacts(
            self.wandb_api,
            entity=entity,
            project=project,
            run_id=run_id,
        )

    async def _handle_get_wandb_run_logs(self, arguments: Dict[str, Any]) -> str:
        """Handle get_wandb_run_logs tool."""
        if not self.wandb_api:
            return self._create_error_response("WandB client not initialized", "get_wandb_run_logs")
        entity = arguments.get("entity")
        project = arguments.get("project")
        run_id = arguments.get("run_id")
        if not entity or not project or not run_id:
            return self._create_error_response("entity, project, and run_id are required", "get_wandb_run_logs")
        return await wandb_tools.get_wandb_run_logs(
            self.wandb_api,
            entity=entity,
            project=project,
            run_id=run_id,
        )

    async def _handle_analyze_wandb_training_progression(self, arguments: Dict[str, Any]) -> str:
        """Handle analyze_wandb_training_progression tool."""
        if not self.wandb_store:
            return self._create_error_response("WandB store not initialized", "analyze_wandb_training_progression")
        entity = arguments.get("entity")
        project = arguments.get("project")
        run_id = arguments.get("run_id")
        metric_keys = arguments.get("metric_keys")
        if not entity or not project or not run_id or not metric_keys:
            return self._create_error_response(
                "entity, project, run_id, and metric_keys are required", "analyze_wandb_training_progression"
            )
        return await wandb_tools.analyze_wandb_training_progression(
            self.wandb_store,
            entity=entity,
            project=project,
            run_id=run_id,
            metric_keys=metric_keys,
            context_window_steps=arguments.get("context_window_steps", 1000),
            center_step=arguments.get("center_step"),
        )

    async def _handle_compare_wandb_runs(self, arguments: Dict[str, Any]) -> str:
        """Handle compare_wandb_runs tool."""
        if not self.wandb_store:
            return self._create_error_response("WandB store not initialized", "compare_wandb_runs")
        entity = arguments.get("entity")
        project = arguments.get("project")
        run_ids = arguments.get("run_ids")
        metric_keys = arguments.get("metric_keys")
        if not entity or not project or not run_ids or not metric_keys:
            return self._create_error_response(
                "entity, project, run_ids, and metric_keys are required", "compare_wandb_runs"
            )
        return await wandb_tools.compare_wandb_runs(
            self.wandb_store,
            entity=entity,
            project=project,
            run_ids=run_ids,
            metric_keys=metric_keys,
        )

    async def _handle_analyze_wandb_learning_curves(self, arguments: Dict[str, Any]) -> str:
        """Handle analyze_wandb_learning_curves tool."""
        if not self.wandb_store:
            return self._create_error_response("WandB store not initialized", "analyze_wandb_learning_curves")
        entity = arguments.get("entity")
        project = arguments.get("project")
        run_id = arguments.get("run_id")
        metric_keys = arguments.get("metric_keys")
        if not entity or not project or not run_id or not metric_keys:
            return self._create_error_response(
                "entity, project, run_id, and metric_keys are required", "analyze_wandb_learning_curves"
            )
        return await wandb_tools.analyze_wandb_learning_curves(
            self.wandb_store,
            entity=entity,
            project=project,
            run_id=run_id,
            metric_keys=metric_keys,
            smoothing_window=arguments.get("smoothing_window", 10),
        )

    async def _handle_identify_wandb_critical_moments(self, arguments: Dict[str, Any]) -> str:
        """Handle identify_wandb_critical_moments tool."""
        if not self.wandb_store:
            return self._create_error_response("WandB store not initialized", "identify_wandb_critical_moments")
        entity = arguments.get("entity")
        project = arguments.get("project")
        run_id = arguments.get("run_id")
        metric_keys = arguments.get("metric_keys")
        if not entity or not project or not run_id or not metric_keys:
            return self._create_error_response(
                "entity, project, run_id, and metric_keys are required", "identify_wandb_critical_moments"
            )
        return await wandb_tools.identify_wandb_critical_moments(
            self.wandb_store,
            entity=entity,
            project=project,
            run_id=run_id,
            metric_keys=metric_keys,
            threshold=arguments.get("threshold", 0.1),
        )

    async def _handle_correlate_wandb_metrics(self, arguments: Dict[str, Any]) -> str:
        """Handle correlate_wandb_metrics tool."""
        if not self.wandb_store:
            return self._create_error_response("WandB store not initialized", "correlate_wandb_metrics")
        entity = arguments.get("entity")
        project = arguments.get("project")
        run_id = arguments.get("run_id")
        metric_pairs = arguments.get("metric_pairs")
        if not entity or not project or not run_id or not metric_pairs:
            return self._create_error_response(
                "entity, project, run_id, and metric_pairs are required", "correlate_wandb_metrics"
            )
        return await wandb_tools.correlate_wandb_metrics(
            self.wandb_store,
            entity=entity,
            project=project,
            run_id=run_id,
            metric_pairs=metric_pairs,
        )

    async def _handle_analyze_wandb_behavioral_patterns(self, arguments: Dict[str, Any]) -> str:
        """Handle analyze_wandb_behavioral_patterns tool."""
        if not self.wandb_store:
            return self._create_error_response("WandB store not initialized", "analyze_wandb_behavioral_patterns")
        entity = arguments.get("entity")
        project = arguments.get("project")
        run_id = arguments.get("run_id")
        if not entity or not project or not run_id:
            return self._create_error_response(
                "entity, project, and run_id are required", "analyze_wandb_behavioral_patterns"
            )
        return await wandb_tools.analyze_wandb_behavioral_patterns(
            self.wandb_store,
            entity=entity,
            project=project,
            run_id=run_id,
            behavior_categories=arguments.get("behavior_categories"),
        )

    async def _handle_generate_wandb_training_insights(self, arguments: Dict[str, Any]) -> str:
        """Handle generate_wandb_training_insights tool."""
        if not self.wandb_store:
            return self._create_error_response("WandB store not initialized", "generate_wandb_training_insights")
        entity = arguments.get("entity")
        project = arguments.get("project")
        run_id = arguments.get("run_id")
        if not entity or not project or not run_id:
            return self._create_error_response(
                "entity, project, and run_id are required", "generate_wandb_training_insights"
            )
        return await wandb_tools.generate_wandb_training_insights(
            self.wandb_store,
            entity=entity,
            project=project,
            run_id=run_id,
        )

    async def _handle_predict_wandb_training_outcome(self, arguments: Dict[str, Any]) -> str:
        """Handle predict_wandb_training_outcome tool."""
        if not self.wandb_store:
            return self._create_error_response("WandB store not initialized", "predict_wandb_training_outcome")
        entity = arguments.get("entity")
        project = arguments.get("project")
        run_id = arguments.get("run_id")
        target_metric = arguments.get("target_metric")
        if not entity or not project or not run_id or not target_metric:
            return self._create_error_response(
                "entity, project, run_id, and target_metric are required", "predict_wandb_training_outcome"
            )
        return await wandb_tools.predict_wandb_training_outcome(
            self.wandb_store,
            entity=entity,
            project=project,
            run_id=run_id,
            target_metric=target_metric,
            projection_steps=arguments.get("projection_steps", 1000),
        )

    async def _handle_list_s3_checkpoints(self, arguments: Dict[str, Any]) -> str:
        """Handle list_s3_checkpoints tool."""
        if not self.s3_store:
            return self._create_error_response("S3 store not initialized", "list_s3_checkpoints")
        return await s3.list_s3_checkpoints(
            self.s3_store,
            run_name=arguments.get("run_name"),
            prefix=arguments.get("prefix"),
            max_keys=arguments.get("max_keys", 1000),
        )

    async def _handle_get_s3_checkpoint_metadata(self, arguments: Dict[str, Any]) -> str:
        """Handle get_s3_checkpoint_metadata tool."""
        if not self.s3_store:
            return self._create_error_response("S3 store not initialized", "get_s3_checkpoint_metadata")
        key = arguments.get("key")
        if not key:
            return self._create_error_response("key parameter is required", "get_s3_checkpoint_metadata")
        return await s3.get_s3_checkpoint_metadata(self.s3_store, key=key)

    async def _handle_get_s3_checkpoint_url(self, arguments: Dict[str, Any]) -> str:
        """Handle get_s3_checkpoint_url tool."""
        if not self.s3_store:
            return self._create_error_response("S3 store not initialized", "get_s3_checkpoint_url")
        key = arguments.get("key")
        if not key:
            return self._create_error_response("key parameter is required", "get_s3_checkpoint_url")
        return await s3.get_s3_checkpoint_url(
            self.s3_store,
            key=key,
            expires_in=arguments.get("expires_in", 3600),
        )

    async def _handle_list_s3_replays(self, arguments: Dict[str, Any]) -> str:
        """Handle list_s3_replays tool."""
        if not self.s3_store:
            return self._create_error_response("S3 store not initialized", "list_s3_replays")
        return await s3.list_s3_replays(
            self.s3_store,
            run_name=arguments.get("run_name"),
            prefix=arguments.get("prefix"),
            max_keys=arguments.get("max_keys", 1000),
        )

    async def _handle_check_s3_object_exists(self, arguments: Dict[str, Any]) -> str:
        """Handle check_s3_object_exists tool."""
        if not self.s3_store:
            return self._create_error_response("S3 store not initialized", "check_s3_object_exists")
        key = arguments.get("key")
        if not key:
            return self._create_error_response("key parameter is required", "check_s3_object_exists")
        return await s3.check_s3_object_exists(self.s3_store, key=key)

    async def _handle_list_skypilot_jobs(self, arguments: Dict[str, Any]) -> str:
        """Handle list_skypilot_jobs tool."""
        return await skypilot.list_skypilot_jobs(
            status=arguments.get("status"),
            limit=arguments.get("limit", 100),
        )

    async def _handle_get_skypilot_job_status(self, arguments: Dict[str, Any]) -> str:
        """Handle get_skypilot_job_status tool."""
        job_id = arguments.get("job_id")
        if not job_id:
            return self._create_error_response("job_id parameter is required", "get_skypilot_job_status")
        return await skypilot.get_skypilot_job_status(job_id=job_id)

    async def _handle_get_skypilot_job_logs(self, arguments: Dict[str, Any]) -> str:
        """Handle get_skypilot_job_logs tool."""
        job_id = arguments.get("job_id")
        if not job_id:
            return self._create_error_response("job_id parameter is required", "get_skypilot_job_logs")
        return await skypilot.get_skypilot_job_logs(
            job_id=job_id,
            tail_lines=arguments.get("tail_lines", 100),
        )

    async def _handle_analyze_s3_checkpoint_progression(self, arguments: Dict[str, Any]) -> str:
        """Handle analyze_s3_checkpoint_progression tool."""
        if not self.s3_store:
            return self._create_error_response("S3 store not initialized", "analyze_s3_checkpoint_progression")
        run_name = arguments.get("run_name")
        if not run_name:
            return self._create_error_response("run_name parameter is required", "analyze_s3_checkpoint_progression")
        return await s3.analyze_s3_checkpoint_progression(
            self.s3_store,
            run_name=run_name,
            prefix=arguments.get("prefix"),
        )

    async def _handle_find_best_s3_checkpoint(self, arguments: Dict[str, Any]) -> str:
        """Handle find_best_s3_checkpoint tool."""
        if not self.s3_store:
            return self._create_error_response("S3 store not initialized", "find_best_s3_checkpoint")
        run_name = arguments.get("run_name")
        if not run_name:
            return self._create_error_response("run_name parameter is required", "find_best_s3_checkpoint")
        return await s3.find_best_s3_checkpoint(
            self.s3_store,
            run_name=run_name,
            criteria=arguments.get("criteria", "latest"),
            prefix=arguments.get("prefix"),
        )

    async def _handle_analyze_s3_checkpoint_usage(self, arguments: Dict[str, Any]) -> str:
        """Handle analyze_s3_checkpoint_usage tool."""
        if not self.s3_store:
            return self._create_error_response("S3 store not initialized", "analyze_s3_checkpoint_usage")
        return await s3.analyze_s3_checkpoint_usage(
            self.s3_store,
            run_name=arguments.get("run_name"),
            prefix=arguments.get("prefix"),
            time_window_days=arguments.get("time_window_days", 30),
        )

    async def _handle_get_s3_checkpoint_statistics(self, arguments: Dict[str, Any]) -> str:
        """Handle get_s3_checkpoint_statistics tool."""
        if not self.s3_store:
            return self._create_error_response("S3 store not initialized", "get_s3_checkpoint_statistics")
        return await s3.get_s3_checkpoint_statistics(
            self.s3_store,
            run_name=arguments.get("run_name"),
            prefix=arguments.get("prefix"),
        )

    async def _handle_compare_s3_checkpoints_across_runs(self, arguments: Dict[str, Any]) -> str:
        """Handle compare_s3_checkpoints_across_runs tool."""
        if not self.s3_store:
            return self._create_error_response("S3 store not initialized", "compare_s3_checkpoints_across_runs")
        run_names = arguments.get("run_names")
        if not run_names:
            return self._create_error_response("run_names parameter is required", "compare_s3_checkpoints_across_runs")
        return await s3.compare_s3_checkpoints_across_runs(
            self.s3_store,
            run_names=run_names,
        )

    async def _handle_analyze_skypilot_job_performance(self, arguments: Dict[str, Any]) -> str:
        """Handle analyze_skypilot_job_performance tool."""
        return await skypilot.analyze_skypilot_job_performance(
            limit=arguments.get("limit", 100),
        )

    async def _handle_get_skypilot_resource_utilization(self, arguments: Dict[str, Any]) -> str:
        """Handle get_skypilot_resource_utilization tool."""
        return await skypilot.get_skypilot_resource_utilization(
            limit=arguments.get("limit", 100),
        )

    async def _handle_compare_skypilot_job_configs(self, arguments: Dict[str, Any]) -> str:
        """Handle compare_skypilot_job_configs tool."""
        job_ids = arguments.get("job_ids")
        if not job_ids:
            return self._create_error_response("job_ids parameter is required", "compare_skypilot_job_configs")
        return await skypilot.compare_skypilot_job_configs(
            job_ids=job_ids,
        )

    async def _handle_analyze_skypilot_job_failures(self, arguments: Dict[str, Any]) -> str:
        """Handle analyze_skypilot_job_failures tool."""
        return await skypilot.analyze_skypilot_job_failures(
            limit=arguments.get("limit", 100),
        )

    async def _handle_get_skypilot_job_cost_estimates(self, arguments: Dict[str, Any]) -> str:
        """Handle get_skypilot_job_cost_estimates tool."""
        return await skypilot.get_skypilot_job_cost_estimates(
            limit=arguments.get("limit", 100),
        )

    async def _handle_link_wandb_run_to_s3_checkpoints(self, arguments: Dict[str, Any]) -> str:
        """Handle link_wandb_run_to_s3_checkpoints tool."""
        if not self.wandb_api or not self.s3_client:
            return self._create_error_response(
                "WandB and S3 clients must be initialized", "link_wandb_run_to_s3_checkpoints"
            )
        entity = arguments.get("entity")
        project = arguments.get("project")
        run_id = arguments.get("run_id")
        if not entity or not project or not run_id:
            return self._create_error_response(
                "entity, project, and run_id are required", "link_wandb_run_to_s3_checkpoints"
            )
        return await wandb_tools.link_wandb_run_to_s3_checkpoints(
            self.wandb_api,
            self.s3_client,
            self.s3_bucket,
            entity=entity,
            project=project,
            run_id=run_id,
        )

    async def _handle_link_wandb_run_to_skypilot_job(self, arguments: Dict[str, Any]) -> str:
        """Handle link_wandb_run_to_skypilot_job tool."""
        if not self.wandb_api:
            return self._create_error_response("WandB client not initialized", "link_wandb_run_to_skypilot_job")
        entity = arguments.get("entity")
        project = arguments.get("project")
        run_id = arguments.get("run_id")
        if not entity or not project or not run_id:
            return self._create_error_response(
                "entity, project, and run_id are required", "link_wandb_run_to_skypilot_job"
            )
        return await wandb_tools.link_wandb_run_to_skypilot_job(
            self.wandb_api,
            entity=entity,
            project=project,
            run_id=run_id,
        )

    async def _handle_link_s3_checkpoint_to_wandb_run(self, arguments: Dict[str, Any]) -> str:
        """Handle link_s3_checkpoint_to_wandb_run tool."""
        if not self.s3_client or not self.wandb_api:
            return self._create_error_response(
                "S3 and WandB clients must be initialized", "link_s3_checkpoint_to_wandb_run"
            )
        key = arguments.get("key")
        entity = arguments.get("entity")
        project = arguments.get("project")
        if not key or not entity or not project:
            return self._create_error_response(
                "key, entity, and project are required", "link_s3_checkpoint_to_wandb_run"
            )
        return await s3.link_s3_checkpoint_to_wandb_run(
            self.s3_client,
            self.s3_bucket,
            self.wandb_api,
            key=key,
            entity=entity,
            project=project,
        )

    async def _handle_link_s3_checkpoint_to_skypilot_job(self, arguments: Dict[str, Any]) -> str:
        """Handle link_s3_checkpoint_to_skypilot_job tool."""
        if not self.s3_client:
            return self._create_error_response("S3 client not initialized", "link_s3_checkpoint_to_skypilot_job")
        key = arguments.get("key")
        if not key:
            return self._create_error_response("key parameter is required", "link_s3_checkpoint_to_skypilot_job")
        return await s3.link_s3_checkpoint_to_skypilot_job(
            self.s3_client,
            self.s3_bucket,
            key=key,
        )

    async def _handle_link_skypilot_job_to_wandb_runs(self, arguments: Dict[str, Any]) -> str:
        """Handle link_skypilot_job_to_wandb_runs tool."""
        if not self.wandb_store:
            return self._create_error_response("WandB store not initialized", "link_skypilot_job_to_wandb_runs")
        job_id = arguments.get("job_id")
        entity = arguments.get("entity")
        project = arguments.get("project")
        if not job_id or not entity or not project:
            return self._create_error_response(
                "job_id, entity, and project are required", "link_skypilot_job_to_wandb_runs"
            )
        return await skypilot.link_skypilot_job_to_wandb_runs(
            self.wandb_store,
            job_id=job_id,
            entity=entity,
            project=project,
        )

    async def _handle_link_skypilot_job_to_s3_checkpoints(self, arguments: Dict[str, Any]) -> str:
        """Handle link_skypilot_job_to_s3_checkpoints tool."""
        if not self.s3_store:
            return self._create_error_response("S3 store not initialized", "link_skypilot_job_to_s3_checkpoints")
        job_id = arguments.get("job_id")
        if not job_id:
            return self._create_error_response("job_id parameter is required", "link_skypilot_job_to_s3_checkpoints")
        return await skypilot.link_skypilot_job_to_s3_checkpoints(
            self.s3_store,
            job_id=job_id,
        )

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
                server.stats_client.close()
            except Exception as e:
                logger.warning(f"Error closing stats client: {e}")
        logger.info("Observatory MCP Server shutdown complete")


def cli_main() -> None:
    """CLI entry point for the server."""
    asyncio.run(main())


if __name__ == "__main__":
    cli_main()
