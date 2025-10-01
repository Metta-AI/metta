from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import duckdb


@dataclass
class TrendResult:
    by_service: list[dict[str, Any]]
    total: list[dict[str, Any]]


def compute_trends(db_path: str) -> TrendResult:
    """Compute simple day-over-day totals and service breakdowns.

    Returns structured dicts suitable for reporting. This serves as a baseline
    analysis to be iterated on.
    """
    con = duckdb.connect(database=db_path, read_only=True)
    try:
        total = (
            con.execute(
                """
            SELECT start AS day, SUM(UnblendedCost) AS unblended
            FROM staging_cost
            GROUP BY day
            ORDER BY day
            """
            )
            .df()
            .to_dict(orient="records")
        )

        by_service = (
            con.execute(
                """
            SELECT start AS day,
                   JSON_EXTRACT_STRING(dimensions, '$."0"') AS service,
                   SUM(UnblendedCost) AS unblended
            FROM staging_cost
            WHERE JSON_EXTRACT_STRING(dimensions, '$."0"') IS NOT NULL
            GROUP BY day, service
            ORDER BY day, service
            """
            )
            .df()
            .to_dict(orient="records")
        )
    finally:
        con.close()

    return TrendResult(by_service=by_service, total=total)


def detect_simple_anomalies(db_path: str, z_threshold: float = 3.0) -> list[dict[str, Any]]:
    """Very simple anomaly detection: z-score on total daily cost.

    Returns records with day, unblended, zscore for flagged anomalies.
    """
    con = duckdb.connect(database=db_path, read_only=True)
    try:
        df = con.execute(
            """
            WITH daily AS (
                SELECT start AS day, SUM(UnblendedCost) AS unblended
                FROM staging_cost
                GROUP BY day
                ORDER BY day
            ), stats AS (
                SELECT AVG(unblended) AS mu, STDDEV_POP(unblended) AS sigma FROM daily
            )
            SELECT daily.day,
                   daily.unblended,
                   CASE WHEN stats.sigma > 0 THEN (daily.unblended - stats.mu) / stats.sigma ELSE 0 END AS zscore
            FROM daily, stats
            ORDER BY daily.day
            """
        ).df()
        anomalies = df[df["zscore"].abs() >= z_threshold]
        return anomalies.to_dict(orient="records")
    finally:
        con.close()


def basic_recommendations(db_path: str) -> list[str]:
    """Placeholder heuristics for recommendations.

    These will be replaced with more robust analyses (RI/SP, right-sizing, etc.).
    """
    con = duckdb.connect(database=db_path, read_only=True)
    try:
        # Identify services with rising trend over last 7 days
        recs: list[str] = []
        df = con.execute(
            """
            WITH svc AS (
                SELECT start AS day,
                       JSON_EXTRACT_STRING(dimensions, '$."0"') AS service,
                       SUM(UnblendedCost) AS cost
                FROM staging_cost
                WHERE JSON_EXTRACT_STRING(dimensions, '$."0"') IS NOT NULL
                GROUP BY day, service
            ), last7 AS (
                SELECT service, SUM(cost) AS last7
                FROM svc
                WHERE day >= (SELECT MAX(start) - INTERVAL 7 DAY FROM staging_cost)
                GROUP BY service
            ), prev7 AS (
                SELECT service, SUM(cost) AS prev7
                FROM svc
                WHERE day < (SELECT MAX(start) - INTERVAL 7 DAY FROM staging_cost)
                  AND day >= (SELECT MAX(start) - INTERVAL 14 DAY FROM staging_cost)
                GROUP BY service
            )
            SELECT COALESCE(last7.service, prev7.service) AS service,
                   COALESCE(last7.last7, 0) AS last7,
                   COALESCE(prev7.prev7, 0) AS prev7
            FROM last7
            FULL OUTER JOIN prev7 ON last7.service = prev7.service
            """
        ).df()

        for _, row in df.iterrows():
            service = str(row.get("service") or "unknown")
            last7 = float(row.get("last7") or 0.0)
            prev7 = float(row.get("prev7") or 0.0)
            if last7 > prev7 * 1.25 and last7 > 10.0:
                recs.append(f"Rising spend in {service}: +{last7 - prev7:.2f} vs prior 7 days")
        return recs
    finally:
        con.close()
