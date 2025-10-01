from __future__ import annotations

import json
from datetime import date
from typing import Any, Iterable

import boto3
import duckdb
import pandas as pd

from .config import CostReportingSettings


def _ce_client(region: str):
    return boto3.client("ce", region_name=region)


def _daterange_str(start: date, end: date) -> tuple[str, str]:
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def _normalize_results(
    results: list[dict[str, Any]],
    metrics: list[str],
    tag_keys: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in results:
        time_period = item.get("TimePeriod", {})
        start = time_period.get("Start")
        end = time_period.get("End")
        groups: list[dict[str, Any]] = item.get("Groups", [])
        if not groups:
            # Ungrouped totals
            amounts = {m: float(item.get("Metrics", {}).get(m, {}).get("Amount", 0.0)) for m in metrics}
            unit = next(
                (item.get("Metrics", {}).get(m, {}).get("Unit") for m in metrics if m in item.get("Metrics", {})), "USD"
            )
            rows.append(
                {
                    "start": start,
                    "end": end,
                    "dimensions": json.dumps({}),
                    "tags": json.dumps({}),
                    "unit": unit,
                    **amounts,
                }
            )
            continue
        for g in groups:
            keys: list[str] = g.get("Keys", [])
            metrics_map: dict[str, Any] = g.get("Metrics", {})
            amounts = {m: float(metrics_map.get(m, {}).get("Amount", 0.0)) for m in metrics}
            unit = next((metrics_map.get(m, {}).get("Unit") for m in metrics if m in metrics_map), "USD")

            # Split tag values vs dimension values (`tag:<Key>` is tag group key in CE)
            dim_values: dict[str, str] = {}
            tag_values: dict[str, str] = {}
            for k in keys:
                if k.startswith("tag:"):
                    # Example: "tag:Project$research" or "tag:Team$ml"
                    try:
                        tag, val = k.split("$", 1)
                        tag_key = tag.removeprefix("tag:")
                        tag_values[tag_key] = val
                    except ValueError:
                        # Fallback: store as-is
                        tag_values[k] = ""
                else:
                    # Common dimensions: SERVICE, LINKED_ACCOUNT, REGION, USAGE_TYPE, etc.
                    dim_values[str(len(dim_values))] = k

            # Keep only requested tag keys (if provided)
            if tag_keys:
                tag_values = {k: v for k, v in tag_values.items() if k in tag_keys}

            rows.append(
                {
                    "start": start,
                    "end": end,
                    "dimensions": json.dumps(dim_values),
                    "tags": json.dumps(tag_values),
                    "unit": unit,
                    **amounts,
                }
            )
    return rows


def collect_cost_and_usage(
    *,
    settings: CostReportingSettings,
    start: date,
    end: date,
    group_by: Iterable[dict[str, str]] | None = None,
) -> pd.DataFrame:
    """Collect cost and usage data from AWS Cost Explorer.

    - Returns a normalized DataFrame with columns: start, end, dimensions (JSON), tags (JSON),
      metric columns (e.g., UnblendedCost), and unit.
    - If `settings.tag_keys` is set, ensures tag groupings are included.
    """

    client = _ce_client(settings.aws_region)
    start_str, end_str = _daterange_str(start, end)

    group_defs: list[dict[str, str]] = []
    if group_by:
        group_defs.extend(list(group_by))
    # Include selected tag keys as groups
    for tag_key in settings.tag_keys:
        group_defs.append({"Type": "TAG", "Key": tag_key})

    token: str | None = None
    all_results: list[dict[str, Any]] = []
    while True:
        params: dict[str, Any] = {
            "TimePeriod": {"Start": start_str, "End": end_str},
            "Granularity": settings.granularity,
            "Metrics": settings.metrics,
        }
        if group_defs:
            params["GroupBy"] = group_defs
        if token:
            params["NextPageToken"] = token

        resp = client.get_cost_and_usage(**params)
        all_results.extend(resp.get("ResultsByTime", []))

        token = resp.get("NextPageToken")
        if not token:
            break

    rows = _normalize_results(all_results, settings.metrics, settings.tag_keys)
    df = pd.DataFrame(rows)
    # Ensure metric columns exist even if empty
    for m in settings.metrics:
        if m not in df.columns:
            df[m] = 0.0
    return df


def write_to_duckdb(
    *,
    df: pd.DataFrame,
    db_path: str | None = None,
) -> None:
    """Append collected rows into a DuckDB database table `staging_cost`.

    If `db_path` is None, writes to a default local path under devops/aws/cost_reporting/data.
    """
    default_db = "devops/aws/cost_reporting/data/cost.duckdb"
    db = db_path or default_db
    con = duckdb.connect(database=db)
    try:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS staging_cost (
                start DATE,
                end DATE,
                dimensions JSON,
                tags JSON,
                unit VARCHAR,
                UnblendedCost DOUBLE,
                AmortizedCost DOUBLE,
                BlendedCost DOUBLE
            );
            """
        )

        # Ensure missing columns exist in df
        for col in ["UnblendedCost", "AmortizedCost", "BlendedCost"]:
            if col not in df.columns:
                df[col] = 0.0

        con.register("input_df", df)
        con.execute(
            """
            INSERT INTO staging_cost
            SELECT
                CAST(start AS DATE),
                CAST(end AS DATE),
                CAST(dimensions AS JSON),
                CAST(tags AS JSON),
                unit,
                CAST(UnblendedCost AS DOUBLE),
                CAST(AmortizedCost AS DOUBLE),
                CAST(BlendedCost AS DOUBLE)
            FROM input_df
            """
        )
    finally:
        con.close()


def collect_and_store(
    *,
    settings: CostReportingSettings,
    start: date,
    end: date,
    db_path: str | None = None,
    group_by_dimensions: list[str] | None = None,
) -> int:
    """High-level helper to collect and persist cost data.

    Returns the number of rows written.
    """
    dims = group_by_dimensions or ["SERVICE", "LINKED_ACCOUNT"]
    group_by = [{"Type": "DIMENSION", "Key": key} for key in dims]
    df = collect_cost_and_usage(settings=settings, start=start, end=end, group_by=group_by)
    write_to_duckdb(df=df, db_path=db_path)
    return int(len(df))
