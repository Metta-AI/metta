from collections import defaultdict
from dataclasses import dataclass, field

import sky
import sky.exceptions
import sky.jobs

from devops.cost_alerter.roster import RosterEntry, get_roster
from devops.skypilot.cost import _cost_info_from_job


@dataclass
class UserJobStats:
    email: str
    running_jobs: int = 0
    total_hourly_cost: float = 0.0
    job_names: list[str] = field(default_factory=list)


@dataclass
class Alert:
    employee_name: str
    employee_email: str
    mentor_email: str | None
    running_jobs: int
    job_threshold: int
    daily_cost: float
    cost_threshold: float
    job_names: list[str]

    @property
    def exceeded_jobs(self) -> bool:
        return self.running_jobs > self.job_threshold

    @property
    def exceeded_cost(self) -> bool:
        return self.daily_cost > self.cost_threshold

    def format_message(self) -> str:
        lines = []
        lines.append(f"**Cost Alert for {self.employee_name}** ({self.employee_email})")
        lines.append("")

        if self.exceeded_jobs:
            lines.append(f"- Running jobs: **{self.running_jobs}** (threshold: {self.job_threshold})")

        if self.exceeded_cost:
            lines.append(f"- Daily cost: **${self.daily_cost:.2f}** (threshold: ${self.cost_threshold:.2f})")

        if self.job_names:
            lines.append("")
            lines.append("Jobs:")
            for name in self.job_names[:10]:
                lines.append(f"  - {name}")
            if len(self.job_names) > 10:
                lines.append(f"  - ... and {len(self.job_names) - 10} more")

        if self.mentor_email:
            lines.append("")
            lines.append(f"Mentor: {self.mentor_email}")

        return "\n".join(lines)


def get_running_jobs_by_user() -> dict[str, UserJobStats]:
    try:
        jobs = sky.get(sky.jobs.queue(refresh=False, all_users=True))
    except sky.exceptions.ClusterNotUpError:
        print("Jobs controller is not running")
        return {}

    stats: dict[str, UserJobStats] = defaultdict(lambda: UserJobStats(email=""))

    for job in jobs:
        if job.get("end_at") is not None:
            continue

        user_email = job.get("user_name", "unknown")
        if "@" not in user_email:
            user_email = f"{user_email}@softmax.com"

        if stats[user_email].email == "":
            stats[user_email].email = user_email

        stats[user_email].running_jobs += 1
        stats[user_email].job_names.append(job.get("job_name", "unnamed"))

        cost_info = _cost_info_from_job(job)
        if cost_info.cost is not None:
            duration_hours = job.get("job_duration", 0) / 3600.0 if job.get("job_duration") else 0
            if duration_hours > 0:
                hourly_rate = cost_info.cost / duration_hours
                stats[user_email].total_hourly_cost += hourly_rate

    return dict(stats)


def check_thresholds(
    roster: list[RosterEntry],
    job_stats: dict[str, UserJobStats],
) -> list[Alert]:
    alerts = []

    roster_by_email = {entry.email.lower(): entry for entry in roster}

    for email, stats in job_stats.items():
        email_lower = email.lower()
        entry = roster_by_email.get(email_lower)

        if entry:
            cost_threshold = entry.cost_threshold_daily
            job_threshold = entry.job_threshold
            mentor_email = entry.mentor_email
            name = entry.name
        else:
            cost_threshold = 200.0
            job_threshold = 5
            mentor_email = None
            name = email.split("@")[0]

        daily_cost = stats.total_hourly_cost * 24

        if stats.running_jobs > job_threshold or daily_cost > cost_threshold:
            alerts.append(
                Alert(
                    employee_name=name,
                    employee_email=email,
                    mentor_email=mentor_email,
                    running_jobs=stats.running_jobs,
                    job_threshold=job_threshold,
                    daily_cost=daily_cost,
                    cost_threshold=cost_threshold,
                    job_names=stats.job_names,
                )
            )

    return alerts


def run_check(skip_roster: bool = False) -> list[Alert]:
    if skip_roster:
        print("Skipping Asana roster (using default thresholds for all users)")
        roster = []
    else:
        print("Loading roster from Asana...")
        try:
            roster = get_roster()
        except Exception as e:
            print(f"Warning: Failed to load roster from Asana: {e}")
            print("Using default thresholds for all users")
            roster = []

    print("Querying SkyPilot for running jobs...")
    job_stats = get_running_jobs_by_user()
    print(f"Found {len(job_stats)} users with running jobs")

    for email, stats in job_stats.items():
        print(f"  {email}: {stats.running_jobs} jobs, ${stats.total_hourly_cost:.2f}/hr")

    print("Checking thresholds...")
    alerts = check_thresholds(roster, job_stats)

    if alerts:
        print(f"Generated {len(alerts)} alerts")
    else:
        print("No threshold violations found")

    return alerts
