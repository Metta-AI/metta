"""
Observatory MCP Server

Model Context Protocol server that enables LLMs to interact with the Metta Observatory backend.
Provides access to training runs, policies, evaluations, scorecards, and SQL queries.
"""

import asyncio
import logging
import sys
from typing import Any, Dict, List

import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from metta.app_backend.clients.scorecard_client import ScorecardClient

from .clients import S3Client, SkypilotClient, WandBClient
from .config import ObservatoryMCPConfig
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

        self.wandb_client: WandBClient | None = None
        if self.config.is_wandb_configured():
            try:
                self.wandb_client = WandBClient(api_key=self.config.wandb_api_key)
                logger.info(
                    f"WandB client initialized (entity={self.config.wandb_entity}, project={self.config.wandb_project})"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize WandB client: {e}. WandB tools will be unavailable.")
                self.wandb_client = None
        else:
            logger.debug("WandB not configured. WandB tools will be unavailable.")

        self.s3_client: S3Client | None = None
        # Always try to initialize S3 client - it can use profile or default credentials
        try:
            self.s3_client = S3Client(profile=self.config.aws_profile, bucket=self.config.s3_bucket)
            if self.s3_client._client is None:
                self.s3_client = None
                logger.warning("S3 client initialization failed. S3 tools will be unavailable.")
            else:
                profile_info = (
                    f"profile={self.config.aws_profile}" if self.config.aws_profile else "default credentials"
                )
                logger.info(f"S3 client initialized ({profile_info}, bucket={self.config.s3_bucket})")
        except Exception as e:
            logger.warning(f"Failed to initialize S3 client: {e}. S3 tools will be unavailable.")
            self.s3_client = None

        self.skypilot_client = SkypilotClient(url=self.config.skypilot_url)
        if self.config.skypilot_url:
            logger.info(f"Skypilot URL configured: {self.config.skypilot_url}")

        self._setup_tools()
        self._setup_resources()

        logger.info(
            f"Observatory MCP Server initialized "
            f"(backend={self.config.backend_url}, "
            f"authenticated={self.config.is_authenticated()}, "
            f"wandb={'enabled' if self.wandb_client else 'disabled'}, "
            f"s3={'enabled' if self.s3_client else 'disabled'}, "
            f"skypilot={'enabled' if self.config.skypilot_url else 'disabled'})"
        )

    def _setup_tools(self) -> None:
        """Register all MCP tools with the server."""

        @self.app.list_tools()
        async def list_tools() -> List[types.Tool]:
            return [
                types.Tool(
                    name="get_training_runs",
                    description=(
                        "Get all training runs from the backend. "
                        "Returns training runs along with their metadata "
                        "(name, created_at, tags, etc.)."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                ),
                types.Tool(
                    name="get_policies",
                    description=(
                        "Get all policies and training runs from the backend. "
                        "Returns both training runs and standalone (run-free) policies."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                ),
                types.Tool(
                    name="search_policies",
                    description=(
                        "Search policies with filtering and pagination. "
                        "Supports filtering by name, type, tags, and user ID."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "search": {
                                "type": "string",
                                "description": "Search term for policy names (case-insensitive partial match)",
                            },
                            "policy_type": {
                                "type": "string",
                                "enum": ["training_run", "policy"],
                                "description": "Filter by policy type: 'training_run' or 'policy'",
                            },
                            "tags": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Filter by tags (policies must have at least one matching tag)",
                            },
                            "user_id": {
                                "type": "string",
                                "description": "Filter by user ID",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results (1-1000)",
                                "default": 100,
                                "minimum": 1,
                                "maximum": 1000,
                            },
                            "offset": {
                                "type": "integer",
                                "description": "Number of results to skip",
                                "default": 0,
                                "minimum": 0,
                            },
                        },
                        "required": [],
                    },
                ),
                types.Tool(
                    name="get_eval_names",
                    description=(
                        "Get available evaluation names for selected training runs and policies. "
                        "Returns list of eval names in format 'eval_category/env_name'."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "training_run_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of training run IDs",
                            },
                            "run_free_policy_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of run-free policy IDs",
                            },
                        },
                        "required": ["training_run_ids", "run_free_policy_ids"],
                    },
                ),
                types.Tool(
                    name="get_available_metrics",
                    description=(
                        "Get available metrics for selected policies and evaluations. "
                        "Returns list of metric names that can be used for scorecard generation."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "training_run_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of training run IDs",
                            },
                            "run_free_policy_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of run-free policy IDs",
                            },
                            "eval_names": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of evaluation names (format: 'eval_category/env_name')",
                            },
                        },
                        "required": ["training_run_ids", "run_free_policy_ids", "eval_names"],
                    },
                ),
                types.Tool(
                    name="generate_scorecard",
                    description=(
                        "Generate scorecard (heatmap) data showing policy performance "
                        "across evaluations for a specific metric. "
                        "Creates a 2D grid of policy vs evaluation performance."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "training_run_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of training run IDs",
                            },
                            "run_free_policy_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of run-free policy IDs",
                            },
                            "eval_names": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of evaluation names (format: 'eval_category/env_name')",
                            },
                            "metric": {
                                "type": "string",
                                "description": (
                                    "Metric to use for scorecard (e.g., 'reward', 'score', 'episode_length')"
                                ),
                            },
                            "policy_selector": {
                                "type": "string",
                                "enum": ["best", "latest"],
                                "description": (
                                    "Policy selection strategy for training runs: "
                                    "'best' (best performing) or 'latest' (most recent)"
                                ),
                                "default": "best",
                            },
                        },
                        "required": ["training_run_ids", "run_free_policy_ids", "eval_names", "metric"],
                    },
                ),
                types.Tool(
                    name="run_sql_query",
                    description=(
                        "Execute SQL query against the backend database. "
                        "The query is validated and executed by the backend API. "
                        "Returns query results with columns and rows."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "sql": {
                                "type": "string",
                                "description": "SQL query string to execute",
                            },
                        },
                        "required": ["sql"],
                    },
                ),
                types.Tool(
                    name="generate_ai_query",
                    description=(
                        "Generate SQL query from natural language description using AI. "
                        "Converts a natural language description into a SQL query "
                        "that can be executed against the backend database."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "description": {
                                "type": "string",
                                "description": (
                                    "Natural language description of desired query "
                                    "(e.g., 'Get all training runs created in the last week')"
                                ),
                            },
                        },
                        "required": ["description"],
                    },
                ),
                types.Tool(
                    name="list_wandb_runs",
                    description="List WandB runs for entity/project with optional filters (tags, state).",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "entity": {"type": "string", "description": "WandB entity (user/team)"},
                            "project": {"type": "string", "description": "WandB project name"},
                            "tags": {"type": "array", "items": {"type": "string"}, "description": "Filter by tags"},
                            "state": {
                                "type": "string",
                                "enum": ["running", "finished", "crashed", "killed"],
                                "description": "Filter by state",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of runs to return",
                                "default": 50,
                            },
                        },
                        "required": ["entity", "project"],
                    },
                ),
                types.Tool(
                    name="get_wandb_run",
                    description="Get detailed information about a specific WandB run.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "entity": {"type": "string", "description": "WandB entity (user/team)"},
                            "project": {"type": "string", "description": "WandB project name"},
                            "run_id": {"type": "string", "description": "WandB run ID (preferred if available)"},
                            "run_name": {
                                "type": "string",
                                "description": "WandB run name (used if run_id not provided)",
                            },
                        },
                        "required": ["entity", "project"],
                    },
                ),
                types.Tool(
                    name="get_wandb_run_metrics",
                    description="Get metric time series data for a WandB run.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "entity": {"type": "string", "description": "WandB entity (user/team)"},
                            "project": {"type": "string", "description": "WandB project name"},
                            "run_id": {"type": "string", "description": "WandB run ID"},
                            "metric_keys": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of metric names to fetch",
                            },
                            "samples": {"type": "integer", "description": "Optional limit on number of samples"},
                        },
                        "required": ["entity", "project", "run_id", "metric_keys"],
                    },
                ),
                types.Tool(
                    name="get_wandb_run_artifacts",
                    description="Get list of artifacts for a WandB run.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "entity": {"type": "string", "description": "WandB entity (user/team)"},
                            "project": {"type": "string", "description": "WandB project name"},
                            "run_id": {"type": "string", "description": "WandB run ID"},
                        },
                        "required": ["entity", "project", "run_id"],
                    },
                ),
                types.Tool(
                    name="get_wandb_run_logs",
                    description="Get logs for a WandB run.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "entity": {"type": "string", "description": "WandB entity (user/team)"},
                            "project": {"type": "string", "description": "WandB project name"},
                            "run_id": {"type": "string", "description": "WandB run ID"},
                        },
                        "required": ["entity", "project", "run_id"],
                    },
                ),
                types.Tool(
                    name="analyze_wandb_training_progression",
                    description=(
                        "Analyze training progression for a WandB run with "
                        "learning velocity, stability, and critical moments."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "entity": {"type": "string", "description": "WandB entity (user/team)"},
                            "project": {"type": "string", "description": "WandB project name"},
                            "run_id": {"type": "string", "description": "WandB run ID"},
                            "metric_keys": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of metric keys to analyze",
                            },
                            "context_window_steps": {
                                "type": "integer",
                                "description": "Number of steps to analyze around center",
                                "default": 1000,
                            },
                            "center_step": {
                                "type": "integer",
                                "description": "Optional center step (defaults to middle of data)",
                            },
                        },
                        "required": ["entity", "project", "run_id", "metric_keys"],
                    },
                ),
                types.Tool(
                    name="compare_wandb_runs",
                    description="Compare multiple WandB runs across specified metrics.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "entity": {"type": "string", "description": "WandB entity (user/team)"},
                            "project": {"type": "string", "description": "WandB project name"},
                            "run_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of WandB run IDs to compare",
                            },
                            "metric_keys": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of metric keys to compare",
                            },
                        },
                        "required": ["entity", "project", "run_ids", "metric_keys"],
                    },
                ),
                types.Tool(
                    name="analyze_wandb_learning_curves",
                    description="Analyze learning curves for trends, convergence, and plateaus.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "entity": {"type": "string", "description": "WandB entity (user/team)"},
                            "project": {"type": "string", "description": "WandB project name"},
                            "run_id": {"type": "string", "description": "WandB run ID"},
                            "metric_keys": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of metric keys to analyze",
                            },
                            "smoothing_window": {
                                "type": "integer",
                                "description": "Window size for smoothing",
                                "default": 10,
                            },
                        },
                        "required": ["entity", "project", "run_id", "metric_keys"],
                    },
                ),
                types.Tool(
                    name="identify_wandb_critical_moments",
                    description="Identify critical moments in training (breakthroughs, drops, plateaus).",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "entity": {"type": "string", "description": "WandB entity (user/team)"},
                            "project": {"type": "string", "description": "WandB project name"},
                            "run_id": {"type": "string", "description": "WandB run ID"},
                            "metric_keys": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of metric keys to analyze",
                            },
                            "threshold": {
                                "type": "number",
                                "description": "Threshold for detecting significant changes",
                                "default": 0.1,
                            },
                        },
                        "required": ["entity", "project", "run_id", "metric_keys"],
                    },
                ),
                types.Tool(
                    name="correlate_wandb_metrics",
                    description="Calculate correlations between metric pairs with statistical significance.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "entity": {"type": "string", "description": "WandB entity (user/team)"},
                            "project": {"type": "string", "description": "WandB project name"},
                            "run_id": {"type": "string", "description": "WandB run ID"},
                            "metric_pairs": {
                                "type": "array",
                                "items": {"type": "array", "items": {"type": "string"}},
                                "description": "List of [metric1, metric2] pairs to correlate",
                            },
                        },
                        "required": ["entity", "project", "run_id", "metric_pairs"],
                    },
                ),
                types.Tool(
                    name="analyze_wandb_behavioral_patterns",
                    description=(
                        "Analyze behavioral patterns including action mastery, "
                        "resource efficiency, and strategy consistency."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "entity": {"type": "string", "description": "WandB entity (user/team)"},
                            "project": {"type": "string", "description": "WandB project name"},
                            "run_id": {"type": "string", "description": "WandB run ID"},
                            "behavior_categories": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Optional list of behavior categories to analyze",
                            },
                        },
                        "required": ["entity", "project", "run_id"],
                    },
                ),
                types.Tool(
                    name="generate_wandb_training_insights",
                    description=(
                        "Generate AI-powered training insights including achievements, "
                        "concerning patterns, and recommendations."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "entity": {"type": "string", "description": "WandB entity (user/team)"},
                            "project": {"type": "string", "description": "WandB project name"},
                            "run_id": {"type": "string", "description": "WandB run ID"},
                        },
                        "required": ["entity", "project", "run_id"],
                    },
                ),
                types.Tool(
                    name="predict_wandb_training_outcome",
                    description="Predict training outcome including projected values and convergence estimates.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "entity": {"type": "string", "description": "WandB entity (user/team)"},
                            "project": {"type": "string", "description": "WandB project name"},
                            "run_id": {"type": "string", "description": "WandB run ID"},
                            "target_metric": {
                                "type": "string",
                                "description": "Metric to predict (e.g., 'overview/reward')",
                            },
                            "projection_steps": {
                                "type": "integer",
                                "description": "Number of steps to project forward",
                                "default": 1000,
                            },
                        },
                        "required": ["entity", "project", "run_id", "target_metric"],
                    },
                ),
                types.Tool(
                    name="list_s3_checkpoints",
                    description="List checkpoints in S3 bucket/prefix.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "run_name": {"type": "string", "description": "Optional training run name to filter by"},
                            "prefix": {
                                "type": "string",
                                "description": "Optional S3 prefix (overrides run_name if both provided)",
                            },
                            "max_keys": {
                                "type": "integer",
                                "description": "Maximum number of objects to return",
                                "default": 1000,
                            },
                        },
                        "required": [],
                    },
                ),
                types.Tool(
                    name="get_s3_checkpoint_metadata",
                    description="Get metadata for a specific S3 checkpoint.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "key": {"type": "string", "description": "S3 object key (full path)"},
                        },
                        "required": ["key"],
                    },
                ),
                types.Tool(
                    name="get_s3_checkpoint_url",
                    description="Generate presigned URL for downloading a checkpoint.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "key": {"type": "string", "description": "S3 object key (full path)"},
                            "expires_in": {
                                "type": "integer",
                                "description": "URL expiration time in seconds",
                                "default": 3600,
                            },
                        },
                        "required": ["key"],
                    },
                ),
                types.Tool(
                    name="list_s3_replays",
                    description="List replay files in S3 bucket/prefix.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "run_name": {"type": "string", "description": "Optional training run name to filter by"},
                            "prefix": {
                                "type": "string",
                                "description": "Optional S3 prefix (overrides run_name if both provided)",
                            },
                            "max_keys": {
                                "type": "integer",
                                "description": "Maximum number of objects to return",
                                "default": 1000,
                            },
                        },
                        "required": [],
                    },
                ),
                types.Tool(
                    name="check_s3_object_exists",
                    description="Check if an S3 object exists and return metadata if it does.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "key": {"type": "string", "description": "S3 object key (full path)"},
                        },
                        "required": ["key"],
                    },
                ),
                types.Tool(
                    name="list_skypilot_jobs",
                    description="List Skypilot jobs with optional status filter.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "status": {
                                "type": "string",
                                "enum": ["PENDING", "RUNNING", "SUCCEEDED", "FAILED", "CANCELLED"],
                                "description": "Optional status filter",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of jobs to return",
                                "default": 100,
                            },
                        },
                        "required": [],
                    },
                ),
                types.Tool(
                    name="get_skypilot_job_status",
                    description="Get detailed status for a specific Skypilot job.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "job_id": {"type": "string", "description": "Skypilot job ID"},
                        },
                        "required": ["job_id"],
                    },
                ),
                types.Tool(
                    name="get_skypilot_job_logs",
                    description="Get logs for a Skypilot job.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "job_id": {"type": "string", "description": "Skypilot job ID"},
                            "tail_lines": {
                                "type": "integer",
                                "description": "Number of lines to return",
                                "default": 100,
                            },
                        },
                        "required": ["job_id"],
                    },
                ),
                types.Tool(
                    name="analyze_s3_checkpoint_progression",
                    description="Analyze checkpoint progression over time for a training run.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "run_name": {"type": "string", "description": "Training run name"},
                            "prefix": {
                                "type": "string",
                                "description": "Optional S3 prefix (overrides run_name if provided)",
                            },
                        },
                        "required": ["run_name"],
                    },
                ),
                types.Tool(
                    name="find_best_s3_checkpoint",
                    description="Find best checkpoint by criteria (latest, largest, smallest, earliest).",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "run_name": {"type": "string", "description": "Training run name"},
                            "criteria": {
                                "type": "string",
                                "enum": ["latest", "largest", "smallest", "earliest"],
                                "description": "Criteria to use",
                                "default": "latest",
                            },
                            "prefix": {
                                "type": "string",
                                "description": "Optional S3 prefix (overrides run_name if provided)",
                            },
                        },
                        "required": ["run_name"],
                    },
                ),
                types.Tool(
                    name="analyze_s3_checkpoint_usage",
                    description="Analyze checkpoint usage patterns over time.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "run_name": {"type": "string", "description": "Optional training run name to filter by"},
                            "prefix": {
                                "type": "string",
                                "description": "Optional S3 prefix (overrides run_name if both provided)",
                            },
                            "time_window_days": {
                                "type": "integer",
                                "description": "Time window in days to analyze",
                                "default": 30,
                            },
                        },
                        "required": [],
                    },
                ),
                types.Tool(
                    name="get_s3_checkpoint_statistics",
                    description="Get statistics about checkpoints (count, size, epoch ranges).",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "run_name": {"type": "string", "description": "Optional training run name to filter by"},
                            "prefix": {
                                "type": "string",
                                "description": "Optional S3 prefix (overrides run_name if both provided)",
                            },
                        },
                        "required": [],
                    },
                ),
                types.Tool(
                    name="compare_s3_checkpoints_across_runs",
                    description="Compare checkpoints across multiple training runs.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "run_names": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of training run names to compare",
                            },
                        },
                        "required": ["run_names"],
                    },
                ),
                types.Tool(
                    name="analyze_skypilot_job_performance",
                    description="Analyze job performance trends including success rates and health scores.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of jobs to analyze",
                                "default": 100,
                            },
                        },
                        "required": [],
                    },
                ),
                types.Tool(
                    name="get_skypilot_resource_utilization",
                    description="Get resource utilization statistics for running and pending jobs.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of jobs to analyze",
                                "default": 100,
                            },
                        },
                        "required": [],
                    },
                ),
                types.Tool(
                    name="compare_skypilot_job_configs",
                    description="Compare job configurations across multiple jobs.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "job_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of job IDs to compare",
                            },
                        },
                        "required": ["job_ids"],
                    },
                ),
                types.Tool(
                    name="analyze_skypilot_job_failures",
                    description="Analyze job failure patterns including failure rates and failed job IDs.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of jobs to analyze",
                                "default": 100,
                            },
                        },
                        "required": [],
                    },
                ),
                types.Tool(
                    name="get_skypilot_job_cost_estimates",
                    description="Get job cost estimates for running and total jobs.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of jobs to analyze",
                                "default": 100,
                            },
                        },
                        "required": [],
                    },
                ),
                types.Tool(
                    name="link_wandb_run_to_s3_checkpoints",
                    description="Link a WandB run to its S3 checkpoints by matching run names.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "entity": {"type": "string", "description": "WandB entity (user/team)"},
                            "project": {"type": "string", "description": "WandB project name"},
                            "run_id": {"type": "string", "description": "WandB run ID"},
                        },
                        "required": ["entity", "project", "run_id"],
                    },
                ),
                types.Tool(
                    name="link_wandb_run_to_skypilot_job",
                    description="Link a WandB run to its Skypilot job by matching run names.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "entity": {"type": "string", "description": "WandB entity (user/team)"},
                            "project": {"type": "string", "description": "WandB project name"},
                            "run_id": {"type": "string", "description": "WandB run ID"},
                        },
                        "required": ["entity", "project", "run_id"],
                    },
                ),
                types.Tool(
                    name="link_s3_checkpoint_to_wandb_run",
                    description="Link an S3 checkpoint to its WandB run by extracting run name from checkpoint path.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "key": {"type": "string", "description": "S3 checkpoint key"},
                            "entity": {"type": "string", "description": "WandB entity (user/team)"},
                            "project": {"type": "string", "description": "WandB project name"},
                        },
                        "required": ["key", "entity", "project"],
                    },
                ),
                types.Tool(
                    name="link_s3_checkpoint_to_skypilot_job",
                    description=(
                        "Link an S3 checkpoint to its Skypilot job by extracting run name from checkpoint path."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "key": {"type": "string", "description": "S3 checkpoint key"},
                        },
                        "required": ["key"],
                    },
                ),
                types.Tool(
                    name="link_skypilot_job_to_wandb_runs",
                    description="Link a Skypilot job to its WandB runs by matching job names.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "job_id": {"type": "string", "description": "Skypilot job ID"},
                            "entity": {"type": "string", "description": "WandB entity (user/team)"},
                            "project": {"type": "string", "description": "WandB project name"},
                        },
                        "required": ["job_id", "entity", "project"],
                    },
                ),
                types.Tool(
                    name="link_skypilot_job_to_s3_checkpoints",
                    description="Link a Skypilot job to its S3 checkpoints by matching job names.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "job_id": {"type": "string", "description": "Skypilot job ID"},
                        },
                        "required": ["job_id"],
                    },
                ),
            ]

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
                if name == "get_training_runs":
                    result = await scorecard.get_training_runs(self.scorecard_client)
                    return [types.TextContent(type="text", text=result)]

                elif name == "get_policies":
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

                    if not eval_names:
                        return [
                            types.TextContent(
                                type="text",
                                text='{"status": "error", "message": "eval_names is required and cannot be empty"}',
                            )
                        ]

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
                    if not self.wandb_client:
                        return [
                            types.TextContent(
                                type="text", text='{"status": "error", "message": "WandB client not initialized"}'
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
                        self.wandb_client,
                        entity=entity,
                        project=project,
                        tags=arguments.get("tags"),
                        state=arguments.get("state"),
                        limit=arguments.get("limit", 50),
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "get_wandb_run":
                    if not self.wandb_client:
                        return [
                            types.TextContent(
                                type="text", text='{"status": "error", "message": "WandB client not initialized"}'
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
                        self.wandb_client,
                        entity=entity,
                        project=project,
                        run_id=arguments.get("run_id"),
                        run_name=arguments.get("run_name"),
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "get_wandb_run_metrics":
                    if not self.wandb_client:
                        return [
                            types.TextContent(
                                type="text", text='{"status": "error", "message": "WandB client not initialized"}'
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
                        self.wandb_client,
                        entity=entity,
                        project=project,
                        run_id=run_id,
                        metric_keys=metric_keys,
                        samples=arguments.get("samples"),
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "get_wandb_run_artifacts":
                    if not self.wandb_client:
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
                        self.wandb_client,
                        entity=entity,
                        project=project,
                        run_id=run_id,
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "get_wandb_run_logs":
                    if not self.wandb_client:
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
                        self.wandb_client,
                        entity=entity,
                        project=project,
                        run_id=run_id,
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "analyze_wandb_training_progression":
                    if not self.wandb_client:
                        return [
                            types.TextContent(
                                type="text", text='{"status": "error", "message": "WandB client not initialized"}'
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
                        self.wandb_client,
                        entity=entity,
                        project=project,
                        run_id=run_id,
                        metric_keys=metric_keys,
                        context_window_steps=arguments.get("context_window_steps", 1000),
                        center_step=arguments.get("center_step"),
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "compare_wandb_runs":
                    if not self.wandb_client:
                        return [
                            types.TextContent(
                                type="text", text='{"status": "error", "message": "WandB client not initialized"}'
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
                        self.wandb_client,
                        entity=entity,
                        project=project,
                        run_ids=run_ids,
                        metric_keys=metric_keys,
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "analyze_wandb_learning_curves":
                    if not self.wandb_client:
                        return [
                            types.TextContent(
                                type="text", text='{"status": "error", "message": "WandB client not initialized"}'
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
                        self.wandb_client,
                        entity=entity,
                        project=project,
                        run_id=run_id,
                        metric_keys=metric_keys,
                        smoothing_window=arguments.get("smoothing_window", 10),
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "identify_wandb_critical_moments":
                    if not self.wandb_client:
                        return [
                            types.TextContent(
                                type="text", text='{"status": "error", "message": "WandB client not initialized"}'
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
                        self.wandb_client,
                        entity=entity,
                        project=project,
                        run_id=run_id,
                        metric_keys=metric_keys,
                        threshold=arguments.get("threshold", 0.1),
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "correlate_wandb_metrics":
                    if not self.wandb_client:
                        return [
                            types.TextContent(
                                type="text", text='{"status": "error", "message": "WandB client not initialized"}'
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
                        self.wandb_client,
                        entity=entity,
                        project=project,
                        run_id=run_id,
                        metric_pairs=metric_pairs,
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "analyze_wandb_behavioral_patterns":
                    if not self.wandb_client:
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
                    result = await wandb.analyze_wandb_behavioral_patterns(
                        self.wandb_client,
                        entity=entity,
                        project=project,
                        run_id=run_id,
                        behavior_categories=arguments.get("behavior_categories"),
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "generate_wandb_training_insights":
                    if not self.wandb_client:
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
                    result = await wandb.generate_wandb_training_insights(
                        self.wandb_client,
                        entity=entity,
                        project=project,
                        run_id=run_id,
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "predict_wandb_training_outcome":
                    if not self.wandb_client:
                        return [
                            types.TextContent(
                                type="text", text='{"status": "error", "message": "WandB client not initialized"}'
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
                        self.wandb_client,
                        entity=entity,
                        project=project,
                        run_id=run_id,
                        target_metric=target_metric,
                        projection_steps=arguments.get("projection_steps", 1000),
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "list_s3_checkpoints":
                    if not self.s3_client:
                        return [
                            types.TextContent(
                                type="text", text='{"status": "error", "message": "S3 client not initialized"}'
                            )
                        ]
                    result = await s3.list_s3_checkpoints(
                        self.s3_client,
                        run_name=arguments.get("run_name"),
                        prefix=arguments.get("prefix"),
                        max_keys=arguments.get("max_keys", 1000),
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "get_s3_checkpoint_metadata":
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
                    result = await s3.get_s3_checkpoint_metadata(self.s3_client, key=key)
                    return [types.TextContent(type="text", text=result)]

                elif name == "get_s3_checkpoint_url":
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
                    result = await s3.get_s3_checkpoint_url(
                        self.s3_client,
                        key=key,
                        expires_in=arguments.get("expires_in", 3600),
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "list_s3_replays":
                    if not self.s3_client:
                        return [
                            types.TextContent(
                                type="text", text='{"status": "error", "message": "S3 client not initialized"}'
                            )
                        ]
                    result = await s3.list_s3_replays(
                        self.s3_client,
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
                    result = await s3.check_s3_object_exists(self.s3_client, key=key)
                    return [types.TextContent(type="text", text=result)]

                elif name == "list_skypilot_jobs":
                    result = await skypilot.list_skypilot_jobs(
                        self.skypilot_client,
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
                    result = await skypilot.get_skypilot_job_status(self.skypilot_client, job_id=job_id)
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
                        self.skypilot_client,
                        job_id=job_id,
                        tail_lines=arguments.get("tail_lines", 100),
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "analyze_s3_checkpoint_progression":
                    if not self.s3_client:
                        return [
                            types.TextContent(
                                type="text", text='{"status": "error", "message": "S3 client not initialized"}'
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
                        self.s3_client,
                        run_name=run_name,
                        prefix=arguments.get("prefix"),
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "find_best_s3_checkpoint":
                    if not self.s3_client:
                        return [
                            types.TextContent(
                                type="text", text='{"status": "error", "message": "S3 client not initialized"}'
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
                        self.s3_client,
                        run_name=run_name,
                        criteria=arguments.get("criteria", "latest"),
                        prefix=arguments.get("prefix"),
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "analyze_s3_checkpoint_usage":
                    if not self.s3_client:
                        return [
                            types.TextContent(
                                type="text", text='{"status": "error", "message": "S3 client not initialized"}'
                            )
                        ]
                    result = await s3.analyze_s3_checkpoint_usage(
                        self.s3_client,
                        run_name=arguments.get("run_name"),
                        prefix=arguments.get("prefix"),
                        time_window_days=arguments.get("time_window_days", 30),
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "get_s3_checkpoint_statistics":
                    if not self.s3_client:
                        return [
                            types.TextContent(
                                type="text", text='{"status": "error", "message": "S3 client not initialized"}'
                            )
                        ]
                    result = await s3.get_s3_checkpoint_statistics(
                        self.s3_client,
                        run_name=arguments.get("run_name"),
                        prefix=arguments.get("prefix"),
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "compare_s3_checkpoints_across_runs":
                    if not self.s3_client:
                        return [
                            types.TextContent(
                                type="text", text='{"status": "error", "message": "S3 client not initialized"}'
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
                        self.s3_client,
                        run_names=run_names,
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "analyze_skypilot_job_performance":
                    result = await skypilot.analyze_skypilot_job_performance(
                        self.skypilot_client,
                        limit=arguments.get("limit", 100),
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "get_skypilot_resource_utilization":
                    result = await skypilot.get_skypilot_resource_utilization(
                        self.skypilot_client,
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
                        self.skypilot_client,
                        job_ids=job_ids,
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "analyze_skypilot_job_failures":
                    result = await skypilot.analyze_skypilot_job_failures(
                        self.skypilot_client,
                        limit=arguments.get("limit", 100),
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "get_skypilot_job_cost_estimates":
                    result = await skypilot.get_skypilot_job_cost_estimates(
                        self.skypilot_client,
                        limit=arguments.get("limit", 100),
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "link_wandb_run_to_s3_checkpoints":
                    if not self.wandb_client or not self.s3_client:
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
                        self.wandb_client,
                        self.s3_client,
                        entity=entity,
                        project=project,
                        run_id=run_id,
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "link_wandb_run_to_skypilot_job":
                    if not self.wandb_client:
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
                        self.wandb_client,
                        self.skypilot_client,
                        entity=entity,
                        project=project,
                        run_id=run_id,
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "link_s3_checkpoint_to_wandb_run":
                    if not self.s3_client or not self.wandb_client:
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
                        self.wandb_client,
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
                        self.skypilot_client,
                        key=key,
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "link_skypilot_job_to_wandb_runs":
                    if not self.wandb_client:
                        return [
                            types.TextContent(
                                type="text", text='{"status": "error", "message": "WandB client not initialized"}'
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
                        self.skypilot_client,
                        self.wandb_client,
                        job_id=job_id,
                        entity=entity,
                        project=project,
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "link_skypilot_job_to_s3_checkpoints":
                    if not self.s3_client:
                        return [
                            types.TextContent(
                                type="text", text='{"status": "error", "message": "S3 client not initialized"}'
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
                        self.skypilot_client,
                        self.s3_client,
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
