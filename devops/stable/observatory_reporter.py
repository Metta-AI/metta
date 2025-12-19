"""Observatory integration for stable release jobs."""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING, Callable, TypeVar

import httpx

if TYPE_CHECKING:
    from devops.stable.runner import Job

from metta.app_backend.clients.stats_client import StatsClient

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _is_transient_httpx_error(exc: Exception) -> bool:
    # StatsClient uses httpx and calls raise_for_status().
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError, httpx.ConnectError, httpx.ReadError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status in (429, 500, 502, 503, 504)
    # Fallback heuristic for wrapped errors.
    msg = str(exc)
    return any(s in msg for s in (" 502 ", "502", "Bad Gateway", " 503 ", " 504 ", "429", "timeout", "timed out"))


def _retry(op_name: str, fn: Callable[[], T], attempts: int = 3, base_sleep_s: float = 1.0) -> T:
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if attempt < attempts and _is_transient_httpx_error(e):
                sleep_s = base_sleep_s * (2 ** (attempt - 1))
                print(
                    f"[observatory] transient error during {op_name} (attempt {attempt}/{attempts}): {str(e)[:160]} ... retrying"
                )
                time.sleep(sleep_s)
                continue
            raise
    assert last_exc is not None
    raise last_exc


class ObservatoryReporter:
    """Reports stable release job results to Observatory."""

    def __init__(self, stats_server_uri: str, machine_token: str | None = None):
        """Initialize the Observatory reporter.

        Args:
            stats_server_uri: Base URL for the Observatory backend API
            machine_token: Optional authentication token (will be read from env if not provided)
        """
        self._stats_server_uri = stats_server_uri
        self._machine_token = machine_token or os.environ.get("OBSERVATORY_TOKEN")
        self._client: StatsClient | None = None

    def _get_client(self) -> StatsClient:
        """Lazy initialize the StatsClient."""
        if self._client is None:
            if not self._machine_token:
                raise ValueError("OBSERVATORY_TOKEN not found in environment")
            self._client = StatsClient(backend_url=self._stats_server_uri, machine_token=self._machine_token)
        return self._client

    def report_jobs(self, jobs: dict[str, Job], suite: str | None = None, dry_run: bool = False) -> None:
        """Report completed stable release jobs to Observatory.

        Args:
            jobs: Dictionary of job name to Job objects
            suite: Suite name (e.g., "stable", "ci")
            dry_run: If True, only log what would be done without making API calls
        """
        if not self._machine_token:
            logger.warning("OBSERVATORY_TOKEN not set, skipping Observatory reporting")
            print("[observatory] OBSERVATORY_TOKEN not set; skipping reporting")
            return

        reportable_jobs = self._filter_reportable_jobs(jobs)
        if not reportable_jobs:
            logger.info("No jobs to report to Observatory")
            print("[observatory] no reportable jobs")
            return

        logger.info(f"Reporting {len(reportable_jobs)} jobs to Observatory")
        print(f"[observatory] reporting {len(reportable_jobs)} jobs")

        for job in reportable_jobs:
            try:
                self._report_single_job(job, suite=suite, dry_run=dry_run)
            except Exception as e:
                # Don't fail the workflow if Observatory reporting fails
                logger.error(f"Failed to report job {job.name} to Observatory: {e}", exc_info=True)

    def _filter_reportable_jobs(self, jobs: dict[str, Job]) -> list[Job]:
        """Filter jobs that should be reported to Observatory.

        We report jobs that:
        - Completed successfully (SUCCEEDED status)
        - Have a WandB run (indicating they produced trackable results)
        """
        from devops.stable.runner import JobStatus

        reportable = []
        for job in jobs.values():
            if job.status != JobStatus.SUCCEEDED:
                continue
            if not job.wandb_run_name:
                continue
            reportable.append(job)
        return reportable

    def _report_single_job(self, job: Job, suite: str | None, dry_run: bool) -> None:
        """Report a single job to Observatory.

        Creates a policy version tagged with stable release metadata.
        """
        if dry_run:
            logger.info(f"[DRY RUN] Would report job to Observatory: {job.name}")
            logger.info(f"  WandB run: {job.wandb_run_name}")
            logger.info(f"  Duration: {job.duration_s:.1f}s")
            logger.info(f"  Acceptance passed: {job.acceptance_passed}")
            return

        # Create policy with job name
        client = self._get_client()
        policy_response = _retry(
            "create_policy",
            lambda: client.create_policy(
                name=job.name,
                attributes={"stable_release": True, "suite": suite or "unknown"},
                is_system_policy=False,
            ),
        )
        policy_id = policy_response.id

        # Create policy version with job metadata
        # Note: For stable release jobs, we use the WandB run as a reference
        # The actual policy spec would need to be retrieved from WandB or the job output
        policy_spec = {
            "type": "stable_release_job",
            "wandb_run": job.wandb_run_name,
            "wandb_url": job.wandb_url,
        }

        version_attributes = {
            "job_name": job.name,
            "duration_seconds": job.duration_s,
            "exit_code": job.exit_code,
        }

        if job.acceptance_passed is not None:
            version_attributes["acceptance_passed"] = job.acceptance_passed

        if job.metrics:
            # Include acceptance metrics if available
            version_attributes["metrics"] = job.metrics

        version_response = _retry(
            "create_policy_version",
            lambda: client.create_policy_version(
                policy_id=policy_id,
                policy_spec=policy_spec,
                attributes=version_attributes,
            ),
        )
        policy_version_id = version_response.id

        # Tag the policy version
        tags = {"stable-release": "true"}
        if suite:
            tags["suite"] = suite
        if job.acceptance_passed is not None:
            tags["acceptance-passed"] = "true" if job.acceptance_passed else "false"

        _retry(
            "update_policy_version_tags",
            lambda: client.update_policy_version_tags(policy_version_id, tags),
        )

        logger.info(
            f"Reported job {job.name} to Observatory: policy_version_id={policy_version_id}, "
            f"acceptance_passed={job.acceptance_passed}"
        )

        # Store the policy version ID on the job for reference
        job.observatory_policy_version_id = str(policy_version_id)
