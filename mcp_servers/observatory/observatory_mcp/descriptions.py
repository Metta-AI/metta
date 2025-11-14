"""Tool descriptions for Observatory MCP Server."""

MCP_TOOL_DESCRIPTIONS = {
    "get_training_runs": (
        "Get all training runs from the backend. "
        "Returns training runs along with their metadata "
        "(name, created_at, tags, etc.)."
    ),
    "get_policies": (
        "Get all policies and training runs from the backend. "
        "Returns both training runs and standalone (run-free) policies."
    ),
    "search_policies": (
        "Search policies with filtering and pagination. "
        "Supports filtering by name, type, tags, and user ID."
    ),
    "get_eval_names": (
        "Get available evaluation names for selected training runs and policies. "
        "Returns list of eval names in format 'eval_category/env_name'."
    ),
    "get_available_metrics": (
        "Get available metrics for selected policies and evaluations. "
        "Returns list of metric names that can be used for scorecard generation."
    ),
    "generate_scorecard": (
        "Generate scorecard (heatmap) data showing policy performance "
        "across evaluations for a specific metric. "
        "Creates a 2D grid of policy vs evaluation performance."
    ),
    "run_sql_query": (
        "Execute SQL query against the backend database. "
        "The query is validated and executed by the backend API. "
        "Returns query results with columns and rows."
    ),
    "generate_ai_query": (
        "Generate SQL query from natural language description using AI. "
        "Converts a natural language description into a SQL query "
        "that can be executed against the backend database."
    ),
    "list_wandb_runs": "List WandB runs for entity/project with optional filters (tags, state).",
    "get_wandb_run": "Get detailed information about a specific WandB run.",
    "get_wandb_run_metrics": "Get metric time series data for a WandB run.",
    "discover_wandb_run_metrics": (
        "Discover available metrics for a WandB run by sampling its history and summary. "
        "Useful when you don't know what metrics are logged in a run."
    ),
    "get_wandb_run_artifacts": "Get list of artifacts for a WandB run.",
    "get_wandb_run_logs": "Get logs for a WandB run.",
    "analyze_wandb_training_progression": (
        "Analyze training progression for a WandB run with "
        "learning velocity, stability, and critical moments."
    ),
    "compare_wandb_runs": "Compare multiple WandB runs across specified metrics.",
    "analyze_wandb_learning_curves": "Analyze learning curves for trends, convergence, and plateaus.",
    "identify_wandb_critical_moments": "Identify critical moments in training (breakthroughs, drops, plateaus).",
    "correlate_wandb_metrics": "Calculate correlations between metric pairs with statistical significance.",
    "analyze_wandb_behavioral_patterns": (
        "Analyze behavioral patterns including action mastery, "
        "resource efficiency, and strategy consistency."
    ),
    "generate_wandb_training_insights": (
        "Generate AI-powered training insights including achievements, "
        "concerning patterns, and recommendations."
    ),
    "predict_wandb_training_outcome": (
        "Predict training outcome including projected values and convergence estimates."
    ),
    "list_s3_checkpoints": "List checkpoints in S3 bucket/prefix.",
    "get_s3_checkpoint_metadata": "Get metadata for a specific S3 checkpoint.",
    "get_s3_checkpoint_url": "Generate presigned URL for downloading a checkpoint.",
    "list_s3_replays": "List replay files in S3 bucket/prefix.",
    "check_s3_object_exists": "Check if an S3 object exists and return metadata if it does.",
    "list_skypilot_jobs": "List Skypilot jobs with optional status filter.",
    "get_skypilot_job_status": "Get detailed status for a specific Skypilot job.",
    "get_skypilot_job_logs": "Get logs for a Skypilot job.",
    "analyze_s3_checkpoint_progression": (
        "Analyze checkpoint progression over time for a training run."
    ),
    "find_best_s3_checkpoint": (
        "Find best checkpoint by criteria (latest, largest, smallest, earliest)."
    ),
    "analyze_s3_checkpoint_usage": "Analyze checkpoint usage patterns over time.",
    "get_s3_checkpoint_statistics": (
        "Get statistics about checkpoints (count, size, epoch ranges)."
    ),
    "compare_s3_checkpoints_across_runs": "Compare checkpoints across multiple training runs.",
    "analyze_skypilot_job_performance": (
        "Analyze job performance trends including success rates and health scores."
    ),
    "get_skypilot_resource_utilization": (
        "Get resource utilization statistics for running and pending jobs."
    ),
    "compare_skypilot_job_configs": "Compare job configurations across multiple jobs.",
    "analyze_skypilot_job_failures": (
        "Analyze job failure patterns including failure rates and failed job IDs."
    ),
    "get_skypilot_job_cost_estimates": "Get job cost estimates for running and total jobs.",
    "link_wandb_run_to_s3_checkpoints": (
        "Link a WandB run to its S3 checkpoints by matching run names."
    ),
    "link_wandb_run_to_skypilot_job": (
        "Link a WandB run to its Skypilot job by matching run names."
    ),
    "link_s3_checkpoint_to_wandb_run": (
        "Link an S3 checkpoint to its WandB run by extracting run name from checkpoint path."
    ),
    "link_s3_checkpoint_to_skypilot_job": (
        "Link an S3 checkpoint to its Skypilot job by extracting run name from checkpoint path."
    ),
    "link_skypilot_job_to_wandb_runs": (
        "Link a Skypilot job to its WandB runs by matching job names."
    ),
    "link_skypilot_job_to_s3_checkpoints": (
        "Link a Skypilot job to its S3 checkpoints by matching job names."
    ),
}

