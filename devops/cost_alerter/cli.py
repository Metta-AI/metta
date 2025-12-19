
import typer

from devops.cost_alerter.alerter import run_check
from metta.common.util.discord import send_to_discord

app = typer.Typer(
    help="Monitor SkyPilot job costs and alert mentors when thresholds are exceeded",
    no_args_is_help=True,
)


@app.command()
def check(
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print alerts without sending to Discord",
    ),
    skip_roster: bool = typer.Option(
        False,
        "--skip-roster",
        help="Skip Asana roster lookup, use default thresholds for all users",
    ),
    webhook_url: str | None = typer.Option(
        None,
        "--webhook-url",
        envvar="DISCORD_WEBHOOK_URL",
        help="Discord webhook URL for alerts",
    ),
) -> None:
    """Check job costs against thresholds and send alerts."""
    alerts = run_check(skip_roster=skip_roster)

    if not alerts:
        print("No alerts to send")
        return

    for alert in alerts:
        message = alert.format_message()
        print("\n" + "=" * 60)
        print(message)
        print("=" * 60)

        if not dry_run:
            if not webhook_url:
                print("Warning: No Discord webhook URL configured, skipping notification")
                continue

            print("Sending alert to Discord...")
            success = send_to_discord(webhook_url, message)
            if success:
                print("Alert sent successfully")
            else:
                print("Failed to send alert")


@app.command()
def list_roster() -> None:
    """List all employees from the Asana Roster project."""
    from devops.cost_alerter.roster import get_roster

    roster = get_roster()

    print(f"\n{'Name':<30} {'Email':<35} {'Mentor':<35} {'Cost $':<10} {'Jobs':<5}")
    print("-" * 120)

    for entry in roster:
        mentor = entry.mentor_email or "-"
        cost = entry.cost_threshold_daily
        jobs = entry.job_threshold
        print(f"{entry.name:<30} {entry.email:<35} {mentor:<35} {cost:<10.0f} {jobs:<5}")


@app.command()
def list_jobs() -> None:
    """List running jobs by user."""
    from devops.cost_alerter.alerter import get_running_jobs_by_user

    stats = get_running_jobs_by_user()

    if not stats:
        print("No running jobs found")
        return

    print(f"\n{'User':<40} {'Jobs':<10} {'Hourly Cost':<15}")
    print("-" * 70)

    for email, s in sorted(stats.items(), key=lambda x: x[1].total_hourly_cost, reverse=True):
        print(f"{email:<40} {s.running_jobs:<10} ${s.total_hourly_cost:<14.2f}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
