# Observatory MCP Server

Model Context Protocol server for accessing Metta Observatory backend functionality.

## Installation

```bash
uv pip install -e mcp_servers/observatory
```

## Configuration

Set environment variables (optional, defaults provided):

```bash
export METTA_MCP_BACKEND_URL="https://api.observatory.softmax-research.net"
export METTA_MCP_MACHINE_TOKEN="your_token"  # Optional, auto-loaded from ~/.metta/config.yaml
export AWS_PROFILE="your-profile"
export METTA_S3_BUCKET="softmax-public"
export WANDB_API_KEY="your_key"
export WANDB_ENTITY="your-entity"
export WANDB_PROJECT="your-project"
export METTA_SKYPILOT_URL="url"
export LOG_LEVEL="INFO"
```

To authenticate, run:

```bash
uv run devops/observatory_login.py <backend_url> <backend_url>
```

## Available Tools

### Backend & Policy Management

- `get_training_runs` - List all training runs with metadata
- `get_policies` - Get all policies and training runs
- `search_policies` - Search policies with filters (name, type, tags, user ID)
- `get_eval_names` - Get available evaluation names for training runs/policies
- `get_available_metrics` - Get available metrics for policies and evaluations
- `generate_scorecard` - Generate scorecard (heatmap) data for policy performance
- `run_sql_query` - Execute SQL queries against the backend database
- `generate_ai_query` - Generate SQL queries from natural language descriptions

### WandB Integration

- `list_wandb_runs` - List WandB runs with optional filters (tags, state)
- `get_wandb_run` - Get detailed information about a WandB run
- `get_wandb_run_metrics` - Get metric time series data for a WandB run
- `get_wandb_run_artifacts` - Get list of artifacts for a WandB run
- `get_wandb_run_logs` - Get logs for a WandB run
- `analyze_wandb_training_progression` - Analyze training progression with learning velocity and critical moments
- `compare_wandb_runs` - Compare multiple WandB runs across metrics
- `analyze_wandb_learning_curves` - Analyze learning curves for trends and convergence
- `identify_wandb_critical_moments` - Identify critical moments (breakthroughs, drops, plateaus)
- `correlate_wandb_metrics` - Calculate correlations between metric pairs
- `analyze_wandb_behavioral_patterns` - Analyze behavioral patterns (action mastery, resource efficiency)
- `generate_wandb_training_insights` - Generate AI-powered training insights and recommendations
- `predict_wandb_training_outcome` - Predict training outcome with projected values

### S3 Storage

- `list_s3_checkpoints` - List checkpoints in S3 bucket/prefix
- `get_s3_checkpoint_metadata` - Get metadata for a specific S3 checkpoint
- `get_s3_checkpoint_url` - Generate presigned URL for downloading a checkpoint
- `list_s3_replays` - List replay files in S3 bucket/prefix
- `check_s3_object_exists` - Check if an S3 object exists and return metadata
- `analyze_s3_checkpoint_progression` - Analyze checkpoint progression over time
- `find_best_s3_checkpoint` - Find best checkpoint by criteria (latest, largest, smallest, earliest)
- `analyze_s3_checkpoint_usage` - Analyze checkpoint usage patterns over time
- `get_s3_checkpoint_statistics` - Get statistics about checkpoints (count, size, epoch ranges)
- `compare_s3_checkpoints_across_runs` - Compare checkpoints across multiple training runs

### Skypilot Job Management

- `list_skypilot_jobs` - List Skypilot jobs with optional status filter
- `get_skypilot_job_status` - Get detailed status for a specific Skypilot job
- `get_skypilot_job_logs` - Get logs for a Skypilot job
- `analyze_skypilot_job_performance` - Analyze job performance trends (success rates, health scores)
- `get_skypilot_resource_utilization` - Get resource utilization statistics
- `compare_skypilot_job_configs` - Compare job configurations across multiple jobs
- `analyze_skypilot_job_failures` - Analyze job failure patterns
- `get_skypilot_job_cost_estimates` - Get job cost estimates

### Cross-Platform Linking

- `link_wandb_run_to_s3_checkpoints` - Link a WandB run to its S3 checkpoints
- `link_wandb_run_to_skypilot_job` - Link a WandB run to its Skypilot job
- `link_s3_checkpoint_to_wandb_run` - Link an S3 checkpoint to its WandB run
- `link_s3_checkpoint_to_skypilot_job` - Link an S3 checkpoint to its Skypilot job
- `link_skypilot_job_to_wandb_runs` - Link a Skypilot job to its WandB runs
- `link_skypilot_job_to_s3_checkpoints` - Link a Skypilot job to its S3 checkpoints

See `CODEX_SETUP.md`, `CLAUDE_DESKTOP_SETUP.md`, or `CURSOR_SETUP.md` for client-specific setup.
