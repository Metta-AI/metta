"""Automated AWS cost collection, analysis, and reporting.

Modules:
- config: Settings and configuration model (env-driven)
- cost_collector: AWS Cost Explorer collection to Parquet
- cost_analyzer: Trends, anomalies, and optimization heuristics
- report_generator: HTML/CSV report generation from aggregated data
- templates: HTML/CSS templates used by report generator
- schedulers: Scheduling helpers (e.g., for CI or cron)
"""

__all__ = [
    "config",
    "cost_collector",
    "cost_analyzer",
    "report_generator",
]
