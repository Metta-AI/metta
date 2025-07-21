import subprocess
from enum import Enum

from pydantic import BaseModel

from notebooks.clients.utils import memoize_with_expiry


class SkyStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def from_raw(cls, raw: str) -> "SkyStatus":
        raw_upper = raw.upper()
        for status in cls:
            if status.value in raw_upper:
                return status
        return cls.UNKNOWN


class SkyPilotJobData(BaseModel):
    job_id: str
    job_name: str = ""
    status: SkyStatus
    resources: str = ""
    duration: str = ""


class SkyPilotClient:
    @memoize_with_expiry(ttl_seconds=10)
    def get_all(self) -> dict[str, SkyPilotJobData]:
        result = subprocess.run(["sky", "jobs", "queue", "--refresh"], capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            return {}

        return self._parse_sky_output(result.stdout)

    def _parse_sky_output(self, output: str) -> dict[str, SkyPilotJobData]:
        jobs = {}
        lines = output.strip().split("\n")

        if len(lines) < 2:
            return jobs

        header_line = None
        data_start = 0
        for i, line in enumerate(lines):
            if "ID" in line and "NAME" in line and "STATUS" in line:
                header_line = line
                data_start = i + 1
                break

        if not header_line:
            return jobs

        headers = ["ID", "NAME", "SUBMITTED", "STATUS", "DURATION", "RESOURCES"]
        col_positions = {}
        for header in headers:
            if header in header_line:
                col_positions[header] = header_line.index(header)

        for line in lines[data_start:]:
            if not line.strip():
                continue

            raw_job = {}
            for header in headers:
                if header in col_positions:
                    start = col_positions[header]
                    next_starts = [col_positions[h] for h in headers if col_positions.get(h, -1) > start]
                    end = min(next_starts) if next_starts else len(line)
                    raw_job[header] = line[start:end].strip()

            if raw_job.get("ID") and raw_job.get("STATUS"):
                job_id = raw_job["ID"]
                jobs[job_id] = SkyPilotJobData(
                    job_id=job_id,
                    job_name=raw_job.get("NAME", ""),
                    status=SkyStatus.from_raw(raw_job.get("STATUS", "")),
                    resources=raw_job.get("RESOURCES", ""),
                    duration=raw_job.get("DURATION", ""),
                )

        return jobs

    def get_job_by_name(self, name: str) -> SkyPilotJobData | None:
        jobs = self.get_jobs_by_names([name])
        return jobs.get(name)

    def get_jobs_by_names(self, names: list[str]) -> dict[str, SkyPilotJobData]:
        """Batch fetch multiple jobs by their names or IDs."""
        results = {}

        # Query SkyPilot for all jobs
        result = subprocess.run(["sky", "jobs", "queue", "--refresh"], capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            all_jobs = self._parse_sky_output(result.stdout)

            # Match by both job name and job ID
            for name in names:
                # Check by job name
                for _job_id, job in all_jobs.items():
                    if job.job_name == name:
                        results[name] = job
                        break

                # Also check if the name itself is a job ID
                if name not in results and name in all_jobs:
                    results[name] = all_jobs[name]

        return results

    def discover_recent_jobs(self, states: list[str] | None = None) -> dict[str, SkyPilotJobData]:
        """Discover all jobs, optionally filtered by state."""
        # Get all jobs from SkyPilot
        all_jobs = self.get_all()

        if not states:
            return all_jobs

        # Filter by states
        filtered = {}
        state_set = {s.upper() for s in states}

        for job_id, job in all_jobs.items():
            if job.status.value in state_set:
                # Use job name as key if available, otherwise job ID
                key = job.job_name or job_id
                filtered[key] = job

        return filtered

    def discover_jobs(self, exclude_names: set[str] | None = None) -> list[str]:
        exclude_names = exclude_names or set()
        discovered = []

        all_jobs = self.get_all()
        for job in all_jobs.values():
            if job.job_name and job.job_name not in exclude_names:
                discovered.append(job.job_name)

        return discovered
