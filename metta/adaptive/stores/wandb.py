import json
import logging
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any, List, Optional

import wandb
from dateutil import parser

from metta.adaptive.models import RunInfo
from metta.common.util.retry import retry_on_exception

logger = logging.getLogger(__name__)


class WandbStore:
    """WandB implementation of adaptive experiment store."""

    # WandB run states
    # TODO We shuold probably just put this into a string enum
    STATUS_RUNNING = "running"
    STATUS_FINISHED = "finished"
    STATUS_CRASHED = "crashed"
    STATUS_FAILED = "failed"
    STATUS_CANCELLED = "cancelled"

    def __init__(self, entity: str, project: str, evaluator_prefix: Optional[str] = None):
        self.entity = entity
        self.project = project
        # Optional evaluator metrics prefix (e.g., "evaluator/eval_sweep") used to detect eval completion
        # If None, falls back to broad detection of any "evaluator/" metrics
        self.evaluator_prefix = evaluator_prefix
        # Don't store api instance - create fresh one each time to avoid caching

    @retry_on_exception(max_retries=3, initial_delay=1.0, max_delay=30.0)
    def init_run(
        self,
        run_id: str,
        group: str | None = None,
        tags: list[str] | None = None,
        initial_summary: dict[str, Any] | None = None,
    ) -> None:
        """Initialize a new run in WandB with optional initial summary data."""
        logger.info(f"[WandbStore] Initializing run {run_id} for group {group}")

        try:
            # Create the run with specific metadata
            # Note: This creates a placeholder run that will be taken over when training starts
            run = wandb.init(
                entity=self.entity,
                project=self.project,
                id=run_id,  # Use run_id as the WandB run ID
                name=run_id,  # Also use run_id as the display name
                group=group,  # Group by experiment_id for organization
                tags=tags or [],  # Tag runs for organization
                reinit=True,  # Allow reinitializing if run exists
                resume="allow",  # Allow resuming existing runs
            )

            # Mark as initialized but not started
            run.summary["initialized"] = True

            # Set any initial summary data before finishing
            if initial_summary:
                for key, value in initial_summary.items():
                    run.summary[key] = value
                logger.info(f"[WandbStore] Set initial summary data for {run_id}: {list(initial_summary.keys())}")

            # Finish immediately - the actual training process will resume this run
            wandb.finish()

            logger.info(f"[WandbStore] Successfully initialized run {run_id}")

        except Exception as e:
            logger.error(f"[WandbStore] Failed to initialize run {run_id}: {e}", exc_info=True)
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
        if "group" in filters:
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
            logger.error(f"[WandbStore] Error fetching runs: {e}", exc_info=True)
            raise

    @retry_on_exception(max_retries=3, initial_delay=1.0, max_delay=30.0)
    def update_run_summary(self, run_id: str, summary_update: dict) -> bool:
        """Update run summary in WandB."""
        try:
            # Create fresh API instance to avoid caching
            api = wandb.Api()
            run = api.run(f"{self.entity}/{self.project}/{run_id}")

            # Updates come from scheduler metrics; assume they are already serializable
            clean_update = dict(summary_update)

            # Debug log what we're updating
            logger.debug(f"[WandbStore] Updating run {run_id} summary with: {clean_update}")

            # Update the summary
            for key, value in clean_update.items():
                run.summary[key] = value
                logger.debug(f"[WandbStore] Set {key}={value} for run {run_id}")

            run.summary.update()  # Force sync with WandB

            # Verify the update took effect
            refreshed_run = api.run(f"{self.entity}/{self.project}/{run_id}")
            refreshed_summary = normalize_summary(refreshed_run.summary)
            logger.debug(
                "[WandbStore] After update, run %s has_started_eval=%s",
                run_id,
                refreshed_summary.get("has_started_eval"),
            )

            return True
        except Exception as e:
            logger.error(f"[WandbStore] Error updating run summary for run {run_id}: {e}", exc_info=True)
            return False

    @retry_on_exception(max_retries=3, initial_delay=1.0, max_delay=30.0)
    def list_runs(
        self,
        entity: str,
        project: str,
        filters: Optional[dict[str, Any]] = None,
        limit: int = 50,
        per_page: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """List WandB runs with pagination and filtering.

        General-purpose method for listing runs. Returns simple dictionaries
        suitable for API responses. For adaptive system use, see fetch_runs().

        Args:
            entity: WandB entity (user/team)
            project: WandB project name
            filters: Optional WandB filters dictionary (e.g., {"state": "finished"})
            limit: Maximum number of runs to return
            per_page: Page size for API requests (defaults to min(limit, 100))

        Returns:
            List of run metadata dictionaries
        """
        # Create fresh API instance to avoid caching (consistent with fetch_runs)
        api = wandb.Api()

        if per_page is None:
            per_page = min(limit, 100)

        runs = api.runs(
            f"{entity}/{project}",
            filters=filters if filters else None,
            order="-created_at",
            per_page=per_page,
        )

        run_list = []
        count = 0
        for run in runs:
            if count >= limit:
                break

            try:
                run_data = self._extract_run_metadata(run, include_config=False, include_summary=False)
                run_list.append(run_data)
                count += 1
            except Exception as e:
                logger.warning(f"[WandbStore] Error processing run {getattr(run, 'id', 'unknown')}: {e}")
                continue

        return run_list

    @retry_on_exception(max_retries=3, initial_delay=1.0, max_delay=30.0)
    def get_run(
        self,
        entity: str,
        project: str,
        run_id: str,
    ) -> Optional[dict[str, Any]]:
        """Get detailed information about a specific run.

        Args:
            entity: WandB entity (user/team)
            project: WandB project name
            run_id: WandB run ID or name

        Returns:
            Run metadata dictionary with config and summary, or None if not found
        """
        try:
            # Create fresh API instance to avoid caching
            api = wandb.Api()
            run = api.run(f"{entity}/{project}/{run_id}")
            return self._extract_run_metadata(run, include_config=True, include_summary=True)
        except Exception as e:
            logger.warning(f"[WandbStore] Failed to get run {run_id}: {e}")
            return None

    @retry_on_exception(max_retries=3, initial_delay=1.0, max_delay=30.0)
    def get_run_metrics(
        self,
        entity: str,
        project: str,
        run_id: str,
        metric_keys: list[str],
        samples: Optional[int] = None,
    ) -> dict[str, Any]:
        """Get metric time series data for a run.

        Args:
            entity: WandB entity (user/team)
            project: WandB project name
            run_id: WandB run ID or name
            metric_keys: List of metric keys to fetch (e.g., ["overview/reward", "metric/agent_step"])
            samples: Optional limit on number of samples (for large runs)

        Returns:
            Dictionary with metric data including run_id, run_name, metric_keys, samples, and data
        """
        try:
            # Create fresh API instance to avoid caching
            api = wandb.Api()
            run = api.run(f"{entity}/{project}/{run_id}")

            # Use same format as original: pandas=False, keys=metric_keys
            if samples:
                history = run.history(keys=metric_keys, pandas=False, samples=samples)
            else:
                history = run.history(keys=metric_keys, pandas=False)

            metrics_data = list(history)

            return {
                "run_id": run_id,
                "run_name": run.name,
                "metric_keys": metric_keys,
                "samples": len(metrics_data),
                "data": metrics_data,
            }
        except Exception as e:
            logger.warning(f"[WandbStore] Failed to get metrics for run {run_id}: {e}")
            return {
                "run_id": run_id,
                "run_name": run_id,
                "metric_keys": metric_keys,
                "samples": 0,
                "data": [],
            }

    @retry_on_exception(max_retries=3, initial_delay=1.0, max_delay=30.0)
    def discover_run_metrics(
        self,
        entity: str,
        project: str,
        run_id: str,
    ) -> dict[str, Any]:
        """Discover available metrics for a run by sampling its history and summary.

        Args:
            entity: WandB entity (user/team)
            project: WandB project name
            run_id: WandB run ID or name

        Returns:
            Dictionary with summary_metrics, history_metrics, all_metrics, and count
        """
        try:
            # Create fresh API instance to avoid caching
            api = wandb.Api()
            run = api.run(f"{entity}/{project}/{run_id}")

            # Get metrics from summary
            summary_metrics = []
            try:
                if hasattr(run, "summary") and run.summary:
                    summary_metrics = [
                        key
                        for key in run.summary.keys()
                        if not key.startswith("_") and key not in ["_timestamp", "_runtime", "_step", "_wandb"]
                    ]
            except Exception as e:
                logger.debug(f"[WandbStore] Could not extract summary metrics: {e}")

            # Get metrics from history sample
            history_metrics = []
            try:
                sample_history = run.history(samples=1, pandas=True)
                if sample_history is not None and not sample_history.empty:
                    history_metrics = [
                        col
                        for col in sample_history.columns
                        if not col.startswith("_") and col not in ["_step", "_runtime", "_timestamp"]
                    ]
            except Exception as e:
                logger.debug(f"[WandbStore] Could not extract history metrics: {e}")

            # Combine and deduplicate
            all_metrics = list(set(summary_metrics + history_metrics))

            return {
                "run_id": run_id,
                "run_name": run.name,
                "summary_metrics": summary_metrics,
                "history_metrics": history_metrics,
                "all_metrics": sorted(all_metrics),
                "count": len(all_metrics),
            }
        except Exception as e:
            logger.warning(f"[WandbStore] Failed to discover metrics for run {run_id}: {e}")
            return {
                "run_id": run_id,
                "run_name": run_id,
                "summary_metrics": [],
                "history_metrics": [],
                "all_metrics": [],
                "count": 0,
            }

    def _extract_run_metadata(
        self,
        run: Any,
        include_config: bool = False,
        include_summary: bool = False,
    ) -> dict[str, Any]:
        """Extract metadata from a WandB run object.

        Args:
            run: WandB run object
            include_config: Whether to include run config (may be slow)
            include_summary: Whether to include run summary (may be slow)

        Returns:
            Run metadata dictionary
        """
        # Handle created_at - can be datetime or string
        created_at = None
        if hasattr(run, "created_at") and run.created_at:
            if isinstance(run.created_at, str):
                created_at = run.created_at
            else:
                created_at = run.created_at.isoformat()

        # Handle updated_at - can be datetime or string (may not exist)
        updated_at = None
        if hasattr(run, "updated_at") and run.updated_at:
            if isinstance(run.updated_at, str):
                updated_at = run.updated_at
            else:
                updated_at = run.updated_at.isoformat()

        # Handle tags - ensure it's always a list
        tags = run.tags if hasattr(run, "tags") else []
        if not isinstance(tags, list):
            tags = list(tags) if tags else []

        # Lazy load config and summary to avoid slow API calls
        config = {}
        if include_config:
            try:
                config = self._safe_dict_convert(run.config) if hasattr(run, "config") else {}
            except Exception:
                config = {}

        summary = {}
        if include_summary:
            try:
                summary = self._safe_dict_convert(run.summary) if hasattr(run, "summary") else {}
            except Exception:
                summary = {}

        return {
            "id": run.id,
            "name": run.name,
            "display_name": getattr(run, "display_name", run.name),
            "state": run.state,
            "tags": tags,
            "created_at": created_at,
            "updated_at": updated_at,
            "url": run.url,
            "config": config,
            "summary": summary,
        }

    def _safe_dict_convert(self, obj: Any) -> dict[str, Any]:
        """Safely convert WandB objects to dictionaries.

        Args:
            obj: Object to convert (config, summary, etc.)

        Returns:
            Dictionary representation
        """
        if obj is None:
            return {}

        if isinstance(obj, dict):
            return {k: self._safe_dict_convert(v) for k, v in obj.items()}

        # Handle WandB specific types
        if hasattr(obj, "__dict__"):
            try:
                if hasattr(obj, "item"):  # numpy scalars
                    return obj.item()
                return str(obj)
            except Exception:
                return str(obj)

        # Handle standard JSON-serializable types
        if isinstance(obj, (str, int, float, bool, list)):
            if isinstance(obj, list):
                return [self._safe_dict_convert(item) for item in obj]
            return obj

        # Default: convert to string
        return str(obj)

    def _convert_run_to_info(self, run: Any) -> RunInfo:
        """Convert WandB run to RunInfo."""
        summary = normalize_summary(run.summary)

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
        runtime = float(summary.get("_runtime", 0))
        if runtime == 0 and hasattr(run, "duration"):
            try:
                runtime = float(run.duration) if run.duration else 0.0
            except Exception:
                runtime = 0.0

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

        # Check evaluation status (regardless of run state)
        # This needs to be outside the elif block because eval can cause run to go back to RUNNING
        if "has_started_eval" in summary and summary["has_started_eval"] is True:  # type: ignore
            has_started_eval = True
            logger.debug(f"[WandbStore] Run {run.id} has_started_eval flag found and set to True")
        else:
            eval_value = summary.get("has_started_eval") if "has_started_eval" in summary else "missing"
            logger.debug(f"[WandbStore] Run {run.id} has_started_eval flag not found or not True. Value: {eval_value}")

        # Check for evaluator metrics under configured namespace (if provided) to avoid false positives
        if self.evaluator_prefix:
            has_evaluator_metrics = any(k.startswith(self.evaluator_prefix) for k in summary.keys())  # type: ignore
        else:
            # Backward-compatible behavior: any evaluator/* metric counts
            has_evaluator_metrics = any(k.startswith("evaluator/") for k in summary.keys())  # type: ignore

        if has_evaluator_metrics:
            has_started_eval = True
            has_been_evaluated = True

        # Extract runtime and cost
        # Runtime is stored under _runtime in WandB
        runtime = float(summary.get("_runtime", 0.0))  # type: ignore
        if runtime == 0.0 and hasattr(run, "duration"):
            runtime = float(run.duration) if run.duration else 0.0

        # Cost is stored under monitor/cost/accrued_total in WandB; default to 0
        cost = summary.get("monitor/cost/accrued_total", 0)  # type: ignore

        # Note: observation field is no longer used - sweep data is stored in summary instead

        # Extract training progress metrics
        total_timesteps = None
        current_steps = None

        # Get total_timesteps from config. Handle both flat and nested configs.
        config_dict = normalize_config(getattr(run, "config", None))

        def _extract_total_steps(d: dict) -> int | None:
            if "trainer" in d and isinstance(d["trainer"], dict):
                value = d["trainer"].get("total_timesteps")
                if value is not None:
                    try:
                        return int(value)
                    except (TypeError, ValueError):
                        return None
            for val in d.values():
                if isinstance(val, dict):
                    result = _extract_total_steps(val)
                    if result is not None:
                        return result
            return None

        maybe_total = _extract_total_steps(config_dict)
        if maybe_total is not None:
            total_timesteps = maybe_total

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

        # Convert created_at to datetime if it's a string
        created_at = None
        if hasattr(run, "created_at"):
            if isinstance(run.created_at, str):
                try:
                    # Parse ISO format datetime string
                    created_at = parser.parse(run.created_at)
                    # Ensure it has UTC timezone
                    if created_at.tzinfo is None:
                        created_at = created_at.replace(tzinfo=timezone.utc)
                except Exception:
                    created_at = None
            elif isinstance(run.created_at, datetime):
                created_at = run.created_at
                # Ensure it has UTC timezone
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=timezone.utc)
            else:
                created_at = None

        # Calculate last_updated_at (always with UTC timezone)
        timestamp = summary.get("_timestamp")
        if timestamp is not None:
            try:
                last_updated_at = datetime.fromtimestamp(float(timestamp), tz=timezone.utc)
            except (TypeError, ValueError):
                last_updated_at = created_at or datetime.now(timezone.utc)
        elif created_at:
            last_updated_at = created_at
        else:
            last_updated_at = datetime.now(timezone.utc)

        # Create RunInfo with all fields set in constructor
        info = RunInfo(
            run_id=run.id,
            group=run.group if hasattr(run, "group") else None,
            tags=run.tags if hasattr(run, "tags") else None,
            created_at=created_at,
            started_at=None,  # WandB doesn't have separate started_at
            completed_at=None,  # Could be derived from state change
            last_updated_at=last_updated_at,
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
        )

        return info


def normalize_summary(summary: Any) -> dict[str, Any]:
    """Best-effort conversion of a WandB run summary to a plain dict."""

    if summary is None:
        return {}

    # WandB HTTPSummary exposes _json_dict, which may itself be a dict or JSON string
    if hasattr(summary, "_json_dict"):
        raw = summary._json_dict  # type: ignore[attr-defined]
        return normalize_summary(raw)

    if isinstance(summary, Mapping):
        return dict(summary)

    if isinstance(summary, str):
        summary = summary.strip()
        if not summary:
            return {}
        try:
            parsed = json.loads(summary)
        except json.JSONDecodeError:
            logger.warning("WandB summary string is not valid JSON; ignoring.")
            return {}
        if isinstance(parsed, Mapping):
            return dict(parsed)
        logger.warning("WandB summary JSON parsed to %s, not dict; ignoring.", type(parsed).__name__)
        return {}

    # Other types (lists/scalars) aren't expected here; treat as empty
    return {}


def normalize_config(config: Any) -> dict[str, Any]:
    """Convert a WandB run config to a plain dict."""

    if config is None:
        return {}

    if isinstance(config, Mapping):
        return dict(config)

    if hasattr(config, "to_dict"):
        try:
            return dict(config.to_dict())
        except Exception:
            pass

    if hasattr(config, "json_config"):
        try:
            parsed = json.loads(config.json_config)
            if isinstance(parsed, Mapping):
                return dict(parsed)
        except Exception:
            pass

    if hasattr(config, "key_vals"):
        try:
            return dict(config.key_vals)
        except Exception:
            pass

    if hasattr(config, "_wandb"):
        try:
            return dict(config._wandb)
        except Exception:
            pass

    if isinstance(config, str):
        config = config.strip()
        if not config:
            return {}
        try:
            parsed = json.loads(config)
            if isinstance(parsed, Mapping):
                return dict(parsed)
        except json.JSONDecodeError:
            return {}

    try:
        return dict(config)
    except Exception:
        return {}
