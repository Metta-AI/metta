import json
import logging
from typing import Any, List, Optional

import wandb

from metta.adaptive.models import Observation, RunInfo
from metta.common.util.numpy_helpers import clean_numpy_types
from metta.common.util.retry import retry_on_exception

logger = logging.getLogger(__name__)


class WandbStore:
    """WandB implementation of sweep store."""

    # WandB run states
    # TODO We shuold probably just put this into a string enum
    STATUS_RUNNING = "running"
    STATUS_FINISHED = "finished"
    STATUS_CRASHED = "crashed"
    STATUS_FAILED = "failed"
    STATUS_CANCELLED = "cancelled"

    def __init__(self, entity: str, project: str):
        self.entity = entity
        self.project = project
        # Don't store api instance - create fresh one each time to avoid caching

    @retry_on_exception(max_retries=3, initial_delay=1.0, max_delay=30.0)
    def init_run(self, run_id: str, group: str | None = None, tags: list[str] | None = None) -> None:
        """Initialize a new run in WandB."""
        logger.info(f"[WandbStore] Initializing run {run_id} for group {group}")

        try:
            # Create the run with specific metadata
            # Note: This creates a placeholder run that will be taken over when training starts
            run = wandb.init(
                entity=self.entity,
                project=self.project,
                id=run_id,  # Use run_id as the WandB run ID
                name=run_id,  # Also use run_id as the display name
                group=group,  # Group by sweep_id for organization
                tags=tags or [],  # Tag as sweep run if part of a sweep
                reinit=True,  # Allow reinitializing if run exists
                resume="allow",  # Allow resuming existing runs
            )

            # Mark as initialized but not started
            run.summary["initialized"] = True

            # Finish immediately - the actual training process will resume this run
            wandb.finish()

            logger.info(f"[WandbStore] Successfully initialized run {run_id}")

        except Exception as e:
            logger.error(f"[WandbStore] Failed to initialize run {run_id}: {e}")
            # Re-raise to prevent dispatch - critical for resource management
            raise RuntimeError(f"Failed to initialize WandB run {run_id}: {e}") from e

    @retry_on_exception(max_retries=3, initial_delay=1.0, max_delay=30.0)
    def fetch_runs(self, filters: dict, limit: Optional[int] = None) -> List[RunInfo]:
        """Fetch runs matching filter criteria.

        Args:
            filters: Dictionary of filter criteria
            limit: Maximum number of runs to fetch (None for no limit)
        """
        # Create fresh API instance to avoid caching
        api = wandb.Api()

        # Convert filters to WandB format
        wandb_filters = {}
        if "sweep_id" in filters:
            wandb_filters["group"] = filters["sweep_id"]
        elif "group" in filters:
            wandb_filters["group"] = filters["group"]

        # Handle name filter (regex pattern)
        if "name" in filters and "regex" in filters["name"]:
            wandb_filters["name"] = {"$regex": filters["name"]["regex"]}

        logger.debug(f"[WandbStore] Fetching runs with filters: {wandb_filters}")

        try:
            # Fetch runs ordered by creation time (newest first)
            runs = api.runs(f"{self.entity}/{self.project}", filters=wandb_filters, order="-created_at")

            run_infos = []
            count = 0
            for run in runs:
                if limit is not None and count >= limit:
                    break
                try:
                    info = self._convert_run_to_info(run)
                    run_infos.append(info)
                    count += 1
                    logger.debug(f"[WandbStore] Converted run {run.id}: state={run.state}, status={info.status}")
                except Exception as e:
                    logger.warning(f"[WandbStore] Failed to convert run {run.id}: {e}")
                    continue

            logger.debug(f"[WandbStore] Found {len(run_infos)} runs")
            return run_infos
        except Exception as e:
            logger.error(f"[WandbStore] Error fetching runs: {e}")
            raise

    @retry_on_exception(max_retries=3, initial_delay=1.0, max_delay=30.0)
    def update_run_summary(self, run_id: str, summary_update: dict) -> bool:
        """Update run summary in WandB."""
        try:
            # Create fresh API instance to avoid caching
            api = wandb.Api()
            run = api.run(f"{self.entity}/{self.project}/{run_id}")

            # Deep clean the update to ensure it's JSON serializable
            clean_update = deep_clean(summary_update)

            # Debug log what we're updating
            logger.debug(f"[WandbStore] Updating run {run_id} summary with: {clean_update}")

            # Update the summary
            for key, value in clean_update.items():
                run.summary[key] = value
                logger.debug(f"[WandbStore] Set {key}={value} for run {run_id}")

            run.summary.update()  # Force sync with WandB

            # Verify the update took effect
            refreshed_run = api.run(f"{self.entity}/{self.project}/{run_id}")
            logger.debug(
                f"[WandbStore] After update, run {run_id} "
                f"has_started_eval={refreshed_run.summary.get('has_started_eval')}"
            )

            return True
        except Exception as e:
            logger.error(f"[WandbStore] Error updating run summary for run {run_id}: {e}")
            return False

    def _convert_run_to_info(self, run: Any) -> RunInfo:
        """Convert WandB run to RunInfo."""
        summary = deep_clean(run.summary)
        # assert isinstance(summary, dict)

        # Debug log the summary to see what's available
        logger.debug(
            f"[WandbStore] Run {run.id} summary keys: "
            f"{list(summary.keys()) if isinstance(summary, dict) else 'not a dict'}"
        )
        logger.debug(
            f"[WandbStore] Run {run.id} has_started_eval in summary: "
            f"{'has_started_eval' in summary if isinstance(summary, dict) else 'N/A'}"
        )
        if isinstance(summary, dict) and "has_started_eval" in summary:
            logger.debug(f"[WandbStore] Run {run.id} has_started_eval value: {summary.get('has_started_eval')}")

        # Determine training/eval status from run state and summary
        has_started_training = False
        has_completed_training = False
        has_started_eval = False
        has_been_evaluated = False
        has_failed = False
        # Check run state and runtime to determine actual status
        runtime = float(run.summary.get("_runtime", 0))

        if run.state == self.STATUS_CRASHED or run.state == self.STATUS_FAILED or run.state == self.STATUS_CANCELLED:
            has_failed = True
        if run.state == self.STATUS_RUNNING:
            has_started_training = True
        elif run.state in [self.STATUS_FINISHED, self.STATUS_CRASHED, self.STATUS_FAILED]:
            if runtime > 0:
                # Actually ran training
                has_started_training = True
                # Completion will be determined by comparing current_steps to total_timesteps below
            else:
                # Just initialized, never actually ran - stays PENDING
                has_started_training = False

            # Check evaluation status
            if "has_started_eval" in summary and summary["has_started_eval"] is True:  # type: ignore
                has_started_eval = True
                logger.debug(f"[WandbStore] Run {run.id} has_started_eval flag found and set to True")
            else:
                eval_value = summary.get("has_started_eval") if "has_started_eval" in summary else "missing"
                logger.debug(
                    f"[WandbStore] Run {run.id} has_started_eval flag not found or not True. Value: {eval_value}"
                )

            # Check for evaluator metrics (ONLY keys starting with "evaluator/")
            # This avoids confusion with in-training eval metrics
            has_evaluator_metrics = any(k.startswith("evaluator/") for k in summary.keys())  # type: ignore

            if has_evaluator_metrics:
                has_started_eval = True
                has_been_evaluated = True

        # Extract cost and runtime first
        # TEMPORARY PATCH: Calculate cost as $4.6 per hour of runtime
        # TODO: Remove this patch when cost tracking is fixed upstream
        # Cost is stored under monitor/cost/accrued_total in WandB
        # cost = float(summary.get("monitor/cost/accrued_total", 0.0))  # type: ignore
        # Runtime is stored under _runtime in WandB
        runtime = float(summary.get("_runtime", 0.0))  # type: ignore
        if runtime == 0.0 and hasattr(run, "duration"):
            runtime = float(run.duration) if run.duration else 0.0

        # Calculate cost based on runtime at $4.6 per hour
        cost_per_hour = 4.6
        runtime_hours = runtime / 3600.0 if runtime > 0 else 0
        cost = cost_per_hour * runtime_hours

        # Extract observation if present - use calculated cost instead of stored value
        observation = None
        if "observation" in summary:
            obs_data = summary["observation"]  # type: ignore
            if isinstance(obs_data, dict) and "score" in obs_data:
                # TEMPORARY PATCH: Use calculated cost instead of stored observation cost
                observation = Observation(
                    score=float(obs_data["score"]),  # type: ignore
                    cost=cost,  # Use calculated cost instead of obs_data["cost"]
                    suggestion=obs_data.get("suggestion", {}),  # type: ignore
                )

        # Extract training progress metrics
        total_timesteps = None
        current_steps = None

        # Get total_timesteps from config
        if hasattr(run, "config"):
            config = dict(run.config)
            # Check trainer.total_timesteps
            if "trainer" in config and isinstance(config["trainer"], dict):
                total_timesteps = config["trainer"].get("total_timesteps")
                if total_timesteps is not None:
                    total_timesteps = int(total_timesteps)

        # Get current_steps from summary
        if "metric/agent_step" in summary:
            current_steps = int(summary["metric/agent_step"])  # type: ignore

        # Determine training completion based on progress
        # Training is complete only if we've reached the target timesteps
        if total_timesteps is not None and current_steps is not None:
            has_completed_training = current_steps >= total_timesteps
        else:
            # If either is None, we haven't completed training
            has_completed_training = False

        # Create RunInfo with all fields set in constructor
        info = RunInfo(
            run_id=run.id,
            group=run.group if hasattr(run, "group") else None,
            tags=run.tags if hasattr(run, "tags") else None,
            created_at=run.created_at if hasattr(run, "created_at") else None,
            started_at=None,  # WandB doesn't have separate started_at
            completed_at=None,  # Could be derived from state change
            last_heartbeat_at=run.heartbeat_at if hasattr(run, "heartbeat_at") else None,
            summary=summary,  # type: ignore
            has_started_training=has_started_training,
            has_completed_training=has_completed_training,
            has_started_eval=has_started_eval,
            has_been_evaluated=has_been_evaluated,
            has_failed=has_failed,
            cost=cost,
            runtime=runtime,
            total_timesteps=total_timesteps,
            current_steps=current_steps,
            observation=observation,
        )

        return info


def deep_clean(obj):
    """Convert object to JSON-serializable types."""
    if isinstance(obj, dict):
        # Already a regular dict, just recursively clean values
        return {k: deep_clean(v) for k, v in obj.items()}
    elif hasattr(obj, "items"):
        # Handle dict-like objects (including WandB SummarySubDict)
        # Convert to regular dict first, then recursively clean
        return {k: deep_clean(v) for k, v in dict(obj).items()}
    elif isinstance(obj, (list, tuple)):
        return [deep_clean(v) for v in obj]
    else:
        # For any other type, use clean_numpy_types first
        cleaned = clean_numpy_types(obj)
        # Then verify it's serializable
        try:
            json.dumps(cleaned)
        except (TypeError, ValueError):
            # If still not serializable, convert to string
            return str(cleaned)
        return cleaned
