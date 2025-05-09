from typing import Any, Dict, List, Literal

from metta.util.config import Config


class Metric(Config):
    metric: str
    filters: Dict[str, Any] = {}


class AnalysisConfig(Config):
    metrics: List[Metric]
    filters: Dict[str, Any] = {}
    baseline_policies: List[str] | None = None
    log_all: bool = True
    metric_patterns: list | None = None
    queries: list | None = None


class AnalyzerConfig(Config):
    num_output_policies: int | Literal["all"] = 20
    view_type: Literal["latest", "policy_versions", "chronological", "all"] = "latest"
    metric: str
    policy_uri: str
    output_path: str
    eval_db_uri: str | None = None

    analysis: AnalysisConfig
