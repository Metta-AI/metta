from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from ..models import MetricKind, MetricSample
from ..utils import parse_iso8601
from .base import BaseCollector


@dataclass
class EvalHealthCollectorConfig:
    source_file: Path | None

    @staticmethod
    def from_env() -> "EvalHealthCollectorConfig":
        path_str = os.environ.get("EVAL_HEALTH_FILE")
        return EvalHealthCollectorConfig(source_file=Path(path_str) if path_str else None)


class EvalHealthCollector(BaseCollector):
    slug = "eval"
    metric_namespace = "metta.infra.cron"
    workflow_category = "evaluation"

    def __init__(self) -> None:
        super().__init__()
        self.config = EvalHealthCollectorConfig.from_env()

    def collect(self) -> list[MetricSample]:
        records = self._load_records()
        samples: List[MetricSample] = []
        for record in records:
            timestamp = parse_iso8601(record["timestamp"]) if record.get("timestamp") else None
            samples.append(
                self.build_sample(
                    metric=record["metric"],
                    value=record["value"],
                    workflow_name=record["workflow_name"],
                    task=record["task"],
                    check=record["check"],
                    condition=record["condition"],
                    status=record.get("status", "unknown"),
                    metric_kind=MetricKind(record.get("metric_kind", MetricKind.GAUGE)),
                    tags=record.get("tags"),
                    timestamp=timestamp,
                )
            )
        if not samples:
            self.logger.warning(
                "Eval health source %s produced zero metrics. Did eval pipeline publish results?",
                self.config.source_file or "<unset>",
            )
        return samples

    def _load_records(self) -> List[Dict[str, Any]]:
        if not self.config.source_file:
            self.logger.warning("EVAL_HEALTH_FILE env var not set; skipping eval metrics.")
            return []
        if not self.config.source_file.exists():
            self.logger.warning("Eval health file %s not found.", self.config.source_file)
            return []
        with self.config.source_file.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
        if isinstance(payload, dict):
            records = payload.get("records", [])
        else:
            records = payload
        if not isinstance(records, list):
            self.logger.error("Expected list of eval health records, got %s", type(records))
            return []
        return [record for record in records if self._validate_record(record)]

    def _validate_record(self, record: Dict[str, Any]) -> bool:
        required = ["metric", "value", "workflow_name", "task", "check", "condition"]
        missing = [field for field in required if field not in record]
        if missing:
            self.logger.error("Eval record missing fields %s: %s", missing, record)
            return False
        return True
