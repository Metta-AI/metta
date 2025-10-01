from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd


@dataclass
class ReportPaths:
    output_dir: Path
    html_path: Path | None = None
    csv_summary_path: Path | None = None


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _read_summary_frames(db_path: str) -> dict[str, pd.DataFrame]:
    con = duckdb.connect(database=db_path, read_only=True)
    try:
        totals = con.execute(
            "SELECT start AS day, SUM(UnblendedCost) AS unblended FROM staging_cost GROUP BY day ORDER BY day"
        ).df()
        services = con.execute(
            """
            SELECT start AS day,
                   JSON_EXTRACT_STRING(dimensions, '$."0"') AS service,
                   SUM(UnblendedCost) AS unblended
            FROM staging_cost
            WHERE JSON_EXTRACT_STRING(dimensions, '$."0"') IS NOT NULL
            GROUP BY day, service
            ORDER BY day, service
            """
        ).df()
        return {"totals": totals, "services": services}
    finally:
        con.close()


def generate_csv_summaries(db_path: str, output_dir: Path) -> Path:
    _ensure_dir(output_dir)
    frames = _read_summary_frames(db_path)
    out = output_dir / "summary.csv"
    # Flatten: first write totals, then services separated by blank line
    with out.open("w", encoding="utf-8") as f:
        frames["totals"].to_csv(f, index=False)
        f.write("\n\n")
        frames["services"].to_csv(f, index=False)
    return out


def generate_html_report(db_path: str, output_dir: Path, title: str = "AWS Cost Report") -> Path:
    _ensure_dir(output_dir)
    frames = _read_summary_frames(db_path)

    totals = frames["totals"].to_dict(orient="records")
    top_services = frames["services"].groupby("service")["unblended"].sum().sort_values(ascending=False).head(10)

    html = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'><title>",
        title,
        "</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;} table{border-collapse:collapse;} th,td{border:1px solid #ddd;padding:6px;} th{background:#f5f5f5;} h2{margin-top:28px;}</style>",
        "</head><body>",
        f"<h1>{title}</h1>",
        "<h2>Daily totals (Unblended)</h2>",
        "<table><tr><th>Day</th><th>Unblended</th></tr>",
    ]
    for row in totals:
        html.append(f"<tr><td>{row['day']}</td><td>{row['unblended']:.2f}</td></tr>")
    html.append("</table>")

    html.append("<h2>Top services by spend</h2>")
    html.append("<table><tr><th>Service</th><th>Total</th></tr>")
    for service, total in top_services.items():
        html.append(f"<tr><td>{service}</td><td>{float(total):.2f}</td></tr>")
    html.append("</table>")

    html.append("</body></html>")
    out = output_dir / "report.html"
    out.write_text("".join(html), encoding="utf-8")
    return out


def build_reports(db_path: str, output_dir: Path, *, title: str = "AWS Cost Report") -> ReportPaths:
    paths = ReportPaths(output_dir=output_dir)
    paths.csv_summary_path = generate_csv_summaries(db_path, output_dir)
    paths.html_path = generate_html_report(db_path, output_dir, title=title)
    return paths
