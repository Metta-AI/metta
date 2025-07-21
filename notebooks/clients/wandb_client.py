import itertools
from enum import Enum
from typing import Any

import wandb
from pydantic import BaseModel, Field


class WandBStatus(str, Enum):
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"
    CRASHED = "crashed"
    KILLED = "killed"
    UNKNOWN = "unknown"

    @classmethod
    def from_raw(cls, raw: str) -> "WandBStatus":
        raw_lower = raw.lower()
        for status in cls:
            if status.value == raw_lower:
                return status
        return cls.UNKNOWN


class RunConfig(BaseModel):
    git_hash: str | None = Field(default=None)
    command_args: str | None = Field(default=None)
    config: dict[str, Any] | None = Field(default=None)


class WandBRunData(RunConfig):
    run_id: str
    url: str
    status: WandBStatus
    project: str = "metta"
    entity: str = "metta-research"
    created_at: str | None = None


# Run sdk: https://docs.wandb.ai/ref/python/public-api/runs/


class WandBClient:
    def __init__(self):
        self._api = None

    @property
    def api(self):
        if self._api is None:
            self._api = wandb.Api()
        return self._api

    def get_run(self, run_id: str, project: str = "metta", entity: str = "metta-research") -> WandBRunData | None:
        # Try direct path first
        run_path = f"{entity}/{project}/{run_id}"

        try:
            run = self.api.run(run_path)
            return self._run_to_data(run)
        except Exception:
            pass

        # If not found by ID, try searching by name
        try:
            runs = self.api.runs(f"{entity}/{project}", filters={"display_name": run_id})
            for run in runs:
                if run.name == run_id or run.id == run_id:
                    return self._run_to_data(run)
        except Exception:
            pass

        return None

    def _run_to_data(self, run) -> WandBRunData:
        return WandBRunData(
            run_id=run.id,
            url=run.url,
            status=WandBStatus.from_raw(run.state),
            project=run.project,
            entity=run.entity,
            config=dict(run.config) if run.config else {},
            git_hash=run.commit or "",
            created_at=run.created_at if hasattr(run, "created_at") else None,
        )

    def get_runs_by_ids(
        self, run_ids: list[str], project: str = "metta", entity: str = "metta-research"
    ) -> dict[str, WandBRunData]:
        """Batch fetch multiple runs by their IDs."""
        results = {}

        # For known IDs, we can try to fetch them individually
        for run_id in run_ids:
            if data := self.get_run(run_id, project, entity):
                # Use the run name as key if available, otherwise use ID
                key = data.run_id
                # Check if this run matches by ID
                for rid in run_ids:
                    if rid == data.run_id:
                        key = rid
                        break
                results[key] = data

        return results

    def discover_recent_runs(
        self, project: str = "metta", entity: str = "metta-research", limit: int = 50, states: list[str] | None = None
    ) -> dict[str, WandBRunData]:
        """Discover recent runs, optionally filtered by state."""
        results = {}

        filters = {}
        if states:
            filters["state"] = {"$in": states}

        try:
            runs = self.api.runs(f"{entity}/{project}", filters=filters, order="-created_at")

            for i, run in enumerate(runs):
                if i >= limit:
                    break

                data = self._run_to_data(run)
                # Use run name as key if available
                key = run.name or run.id
                results[key] = data

        except Exception as e:
            print(f"Error discovering W&B runs: {e}")

        return results

    def list_all(
        self,
        project: str = "metta",
        entity: str = "metta-research",
        limit: int = 20,
        exclude_ids: set[str] | None = None,
    ) -> list[str]:
        exclude_ids = exclude_ids or set()
        discovered = []

        # Get runs iterator ordered by creation time (most recent first)
        # This ensures we get the most relevant runs even with a limit
        runs_iter = self.api.runs(
            f"{entity}/{project}",
            order="-created_at",  # Descending order by creation time
        )

        # Use islice to stop iteration after we've seen 'limit' runs
        # This prevents the API from fetching all runs
        for run in itertools.islice(runs_iter, limit):
            if run.id not in exclude_ids and run.name not in exclude_ids:
                discovered.append(run.name or run.id)

        return discovered
