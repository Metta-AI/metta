import json
import logging
from typing import Any, List
from metta.common.util.numpy_helpers import clean_numpy_types
from metta.sweep.sweep_orchestrator import RunInfo, JobStatus, Observation
import wandb

logger = logging.getLogger(__name__)


class WandbStore:
    """WandB implementation of the Store protocol"""

    # WandB run states
    STATUS_RUNNING = "running"
    STATUS_FINISHED = "finished"
    STATUS_CRASHED = "crashed"
    STATUS_FAILED = "failed"

    def __init__(self, entity: str, project: str):
        self.entity = entity
        self.project = project
        self.api = wandb.Api()

    def init_run(self, run_id: str, sweep_id: str | None = None) -> None:
        """Initialize a new run in WandB with proper metadata for sweep tracking"""
        logger.info(f"[WandbStore] Initializing run {run_id} for sweep {sweep_id}")

        try:
            # Create the run with specific metadata
            # Note: This creates a placeholder run that will be taken over when training starts
            run = wandb.init(
                entity=self.entity,
                project=self.project,
                id=run_id,  # Use run_id as the WandB run ID
                name=run_id,  # Also use run_id as the display name
                group=sweep_id,  # Group by sweep_id for organization
                tags=["sweep"] if sweep_id else [],  # Tag as sweep run if part of a sweep
                reinit=True,  # Allow reinitializing if run exists
                resume="allow",  # Allow resuming existing runs
            )

            # Mark as initialized but not started
            run.summary["initialized"] = True
            # run.summary["has_started_training"] = False

            # Finish immediately - the actual training process will resume this run
            wandb.finish()

            logger.info(f"[WandbStore] Successfully initialized run {run_id}")

        except Exception as e:
            logger.error(f"[WandbStore] Failed to initialize run {run_id}: {e}")
            # Re-raise to prevent dispatch - critical for resource management
            raise RuntimeError(f"Failed to initialize WandB run {run_id}: {e}") from e

    def fetch_runs(self, filters: dict) -> List[RunInfo]:
        """Fetch runs matching filter criteria"""
        # Convert sweep_id filter to group filter for WandB
        wandb_filters = {}
        if "sweep_id" in filters:
            wandb_filters["group"] = filters["sweep_id"]
        elif "group" in filters:
            wandb_filters["group"] = filters["group"]

        logger.debug(f"[WandbStore] Fetching runs with filters: {wandb_filters}")

        try:
            runs = self.api.runs(f"{self.entity}/{self.project}", filters=wandb_filters)
            run_infos = []
            for run in runs:
                try:
                    info = self._convert_run_to_info(run)
                    run_infos.append(info)
                    logger.debug(f"[WandbStore] Converted run {run.id}: state={run.state}, status={info.status}")
                except Exception as e:
                    logger.warning(f"[WandbStore] Failed to convert run {run.id}: {e}")
                    continue

            logger.debug(f"[WandbStore] Found {len(run_infos)} runs")
            return run_infos
        except Exception as e:
            logger.warning(f"[WandbStore] Error fetching runs: {e}. Returning empty list.")
            return []

    def update_run_summary(self, run_id: str, summary_update: dict) -> bool:
        """Update the summary of a WandB run"""
        try:
            run = self.api.run(f"{self.entity}/{self.project}/{run_id}")

            # Deep clean the update to ensure it's JSON serializable
            clean_update = deep_clean(summary_update)

            # Update the summary
            for key, value in clean_update.items():
                run.summary[key] = value

            run.summary.update()  # Force sync with WandB
            return True
        except Exception as e:
            logger.error(f"[WandbStore] Error updating run summary for run {run_id}: {e}")
            return False

    # Adapter
    def _convert_run_to_info(self, run: Any) -> RunInfo:
        """Convert WandB run to RunInfo"""
        summary = deep_clean(run.summary)
        assert isinstance(summary, dict)

        # Determine training/eval status from run state and summary
        has_started_training = False
        has_completed_training = False
        has_started_eval = False
        has_been_evaluated = False

        # Check run state and runtime to determine actual status
        runtime = float(run.summary.get("_runtime", 0))

        logger.info(f"Run {run.id}: state={run.state}, runtime={runtime}")

        if run.state == self.STATUS_RUNNING:
            has_started_training = True
        elif run.state in [self.STATUS_FINISHED, self.STATUS_CRASHED, self.STATUS_FAILED]:
            if runtime > 0:
                # Actually ran training
                has_started_training = True
                has_completed_training = True
            else:
                # Just initialized, never actually ran - stays PENDING
                logger.debug(f"Run {run.id} finished but has no runtime, treating as PENDING")
                has_started_training = False
                has_completed_training = False

            # Check evaluation status
            if "has_started_eval" in summary and summary["has_started_eval"] is True:
                has_started_eval = True

            # Check for evaluator metrics (ONLY keys starting with "evaluator/")
            # This avoids confusion with in-training eval metrics
            has_evaluator_metrics = any(k.startswith("evaluator/") for k in summary.keys())

            if has_evaluator_metrics:
                has_started_eval = True
                has_been_evaluated = True

        # Extract observation if present
        observation = None
        if "observation" in summary:
            obs_data = summary["observation"]
            if isinstance(obs_data, dict) and "score" in obs_data and "cost" in obs_data:
                observation = Observation(
                    score=float(obs_data["score"]),  # type: ignore
                    cost=float(obs_data["cost"]),  # type: ignore
                    suggestion=obs_data.get("suggestion", {}),  # type: ignore
                )

        # Extract cost and runtime
        # Cost is stored under monitor/cost/accrued_total in WandB
        cost = float(summary.get("monitor/cost/accrued_total", 0.0))  # type: ignore
        # Runtime is stored under _runtime in WandB
        runtime = float(summary.get("_runtime", 0.0))  # type: ignore
        if runtime == 0.0 and hasattr(run, "duration"):
            runtime = float(run.duration) if run.duration else 0.0

        # Create RunInfo with all fields set in constructor
        info = RunInfo(
            run_id=run.id,
            group=run.group if hasattr(run, "group") else None,
            tags=run.tags if hasattr(run, "tags") else None,
            created_at=run.created_at if hasattr(run, "created_at") else None,
            started_at=None,  # WandB doesn't have separate started_at
            completed_at=None,  # Could be derived from state change
            last_heartbeat_at=run.heartbeat_at if hasattr(run, "heartbeat_at") else None,
            summary=summary,
            has_started_training=has_started_training,
            has_completed_training=has_completed_training,
            has_started_eval=has_started_eval,
            has_been_evaluated=has_been_evaluated,
            cost=cost,
            runtime=runtime,
            observation=observation,
        )

        return info


# Data Utility
def deep_clean(obj):
    """Recursively convert any object to JSON-serializable Python types."""
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Fetching test run from WandB...")
    test_run = wandb.Api().run("metta-research/metta/arena_shaped_test")
    print(f"Run state: {test_run.state}")
    print(f"Run summary keys (first 20): {list(test_run.summary.keys())[:20]}")

    # Check different places for evaluation markers
    print("\n=== Checking for evaluation markers ===")

    # 1. Check summary
    print("In summary:")
    eval_keys = [k for k in test_run.summary.keys() if "eval" in k.lower() or "arena" in k.lower()]
    for key in eval_keys:
        print(f"  {key}: {test_run.summary[key]}")

    # 2. Check config
    print("\nIn config:")
    if hasattr(test_run, "config"):
        eval_config_keys = [k for k in test_run.config.keys() if "eval" in k.lower()]
        for key in eval_config_keys:
            print(f"  {key}: {test_run.config[key]}")

    # 3. Check history (last few metrics)
    print("\nIn history (last 5 steps):")
    if hasattr(test_run, "history"):
        history = test_run.history()
        if len(history) > 0:
            last_entries = history.tail(5)
            for col in last_entries.columns:
                if "eval" in col.lower() or "arena" in col.lower():
                    print(f"  {col} found in history")

    # 4. Check system metrics
    print("\nSystem metrics:")
    if hasattr(test_run, "system_metrics"):
        print(f"  System metrics keys: {test_run.system_metrics}")

    store = WandbStore("metta-research", "metta")
    info = store._convert_run_to_info(test_run)

    print("\n=== RunInfo ===")
    print(f"run_id: {info.run_id}")
    print(f"group: {info.group}")
    print(f"status: {info.status}")

    # Add visual indicators for boolean states
    def check_mark(condition: bool) -> str:
        return "✅" if condition else "❌"

    print(f"{check_mark(info.has_started_training)} has_started_training: {info.has_started_training}")
    print(f"{check_mark(info.has_completed_training)} has_completed_training: {info.has_completed_training}")
    print(f"{check_mark(info.has_started_eval)} has_started_eval: {info.has_started_eval}")
    print(f"{check_mark(info.has_been_evaluated)} has_been_evaluated: {info.has_been_evaluated}")
    print(f"{check_mark(info.observation is not None)} observation: {info.observation}")
    print(f"{check_mark(info.cost > 0)} cost: {info.cost}")
    print(f"{check_mark(info.runtime > 0)} runtime: {info.runtime}")

    # Also check specific eval markers
    print("\n=== Direct checks ===")
    has_started_eval_flag = test_run.summary.get("has_started_eval")
    evaluator_in_summary = "evaluator" in test_run.summary

    print(f"{check_mark(has_started_eval_flag)} 'has_started_eval' in summary: {has_started_eval_flag}")
    print(f"{check_mark(evaluator_in_summary)} 'evaluator' in summary: {evaluator_in_summary}")

    # Summary of overall status
    print("\n=== Overall Status ===")
    is_fully_complete = (
        info.has_started_training and info.has_completed_training and info.has_started_eval and info.has_been_evaluated
    )
    print(f"{check_mark(is_fully_complete)} Run fully complete (trained and evaluated)")
