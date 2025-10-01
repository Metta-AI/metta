from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import typer

from devops.aws.cost_reporting.config import CostReportingSettings
from devops.aws.cost_reporting.cost_analyzer import basic_recommendations, compute_trends, detect_simple_anomalies
from devops.aws.cost_reporting.cost_collector import collect_and_store
from devops.aws.cost_reporting.report_generator import build_reports
from metta.common.util.log_config import init_logging

app = typer.Typer(help="AWS cost reporting CLI", no_args_is_help=True)


def _parse_date(s: str) -> date:
    try:
        return date.fromisoformat(s)
    except Exception as e:
        raise typer.BadParameter(f"Invalid date: {s}") from e


@app.command()
def collect(
    start: Optional[str] = typer.Option(None, help="Start date (YYYY-MM-DD). Default: 30 days ago"),
    end: Optional[str] = typer.Option(None, help="End date (YYYY-MM-DD). Default: today"),
    db: Optional[str] = typer.Option(None, help="DuckDB database path"),
) -> None:
    """Collect costs from AWS Cost Explorer and store to DuckDB."""
    init_logging()
    settings = CostReportingSettings()
    start_date = _parse_date(start) if start else date.today() - timedelta(days=30)
    end_date = _parse_date(end) if end else date.today()

    written = collect_and_store(settings=settings, start=start_date, end=end_date, db_path=db)
    typer.echo(f"Wrote {written} rows to DuckDB")


@app.command()
def analyze(
    db: str = typer.Option("devops/aws/cost_reporting/data/cost.duckdb", help="DuckDB database path"),
    show_recommendations: bool = typer.Option(True, help="Print basic recommendations"),
) -> None:
    """Run baseline trend and anomaly analyses and print a short summary."""
    init_logging()
    trends = compute_trends(db)
    anomalies = detect_simple_anomalies(db)
    typer.echo(f"Totals points: {len(trends.total)}, Service points: {len(trends.by_service)}")
    typer.echo(f"Anomalies: {len(anomalies)}")
    if show_recommendations:
        recs = basic_recommendations(db)
        typer.echo("Recommendations:")
        for r in recs:
            typer.echo(f"- {r}")


@app.command()
def report(
    db: str = typer.Option("devops/aws/cost_reporting/data/cost.duckdb", help="DuckDB database path"),
    out: str = typer.Option("devops/aws/cost_reporting/out", help="Output directory"),
    title: str = typer.Option("AWS Cost Report", help="Report title"),
) -> None:
    """Generate HTML and CSV reports from DuckDB."""
    init_logging()
    outdir = Path(out)
    paths = build_reports(db, outdir, title=title)
    typer.echo(f"HTML: {paths.html_path}")
    typer.echo(f"CSV:  {paths.csv_summary_path}")


def main() -> None:
    try:
        app()
    except Exception as e:
        typer.echo(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
