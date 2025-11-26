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
class TrainingHealthCollectorConfig:
    source_file: Path | None

    @staticmethod
    def from_env() -> "TrainingHealthCollectorConfig":
        path_str = os.environ.get("TRAINING_HEALTH_FILE")
        return TrainingHealthCollectorConfig(source_file=Path(path_str) if path_str else None)


class TrainingHealthCollector(BaseCollector):
    slug = "training"
    metric_namespace = "metta.infra.stablesuite"
    workflow_category = "training"
    source = "stable_suite"

    def __init__(self) -> None:
        super().__init__()
        self.config = TrainingHealthCollectorConfig.from_env()

    def collect(self) -> list[MetricSample]:
        records = self._load_records()
        samples: List[MetricSample] = []
        for record in records:
            metric_kind = MetricKind(record.get("metric_kind", MetricKind.GAUGE))
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
                    metric_kind=metric_kind,
                    tags=record.get("tags"),
                    timestamp=timestamp,
                )
            )
        if not samples:
            self.logger.warning(
                "Training health source %s produced zero metrics. Did stable_suite publish results?",
                self.config.source_file or "<unset>",
            )
        return samples

    def _load_records(self) -> List[Dict[str, Any]]:
        if not self.config.source_file:
            self.logger.warning("TRAINING_HEALTH_FILE env var not set; skipping training metrics.")
            return []
        if not self.config.source_file.exists():
            self.logger.warning("Training health file %s not found.", self.config.source_file)
            return []
        with self.config.source_file.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
        if isinstance(payload, dict):
            records = payload.get("records", [])
        else:
            records = payload
        if not isinstance(records, list):
            self.logger.error("Expected list of training health records, got %s", type(records))
            return []
        return [record for record in records if self._validate_record(record)]

    def _validate_record(self, record: Dict[str, Any]) -> bool:
        required = ["metric", "value", "workflow_name", "task", "check", "condition"]
        missing = [field for field in required if field not in record]
        if missing:
            self.logger.error("Training record missing fields %s: %s", missing, record)
            return False
        return True
