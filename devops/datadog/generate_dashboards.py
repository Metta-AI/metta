from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple, TypedDict

from devops.datadog.metric_schema import METRIC_SCHEMA, CategoryDefinition, MetricDefinition

DASHBOARD_DIR = Path(__file__).parent / "dashboards"
SUMMARY_PATH = DASHBOARD_DIR / "infra_summary.json"
DETAILED_PATH = DASHBOARD_DIR / "infra_detailed.json"


class ConditionClause(TypedDict):
    label: str
    comparator: str
    value: float


def parse_condition(condition: str) -> Dict[str, ConditionClause]:
    """Return clauses keyed by label ('pass', 'warn', 'fail')."""
    clauses: Dict[str, ConditionClause] = {}
    parts = [p.strip() for p in (condition or "").split("|") if p.strip()]
    if not parts:
        parts = ["> 0"]
    for raw in parts:
        label = "pass"
        lowered = raw.lower()
        remainder = raw
        if lowered.startswith("warn"):
            label = "warn"
            remainder = raw[4:].strip()
        elif lowered.startswith("fail"):
            label = "fail"
            remainder = raw[4:].strip()
        comparator, value = _split_comparator(remainder)
        clauses[label] = {"label": label, "comparator": comparator, "value": value}
    if "pass" not in clauses and clauses:
        clauses["pass"] = next(iter(clauses.values()))
    return clauses


def _split_comparator(text: str) -> Tuple[str, float]:
    text = text.strip()
    for op in ("<=", ">=", "==", "!=", "<", ">", "="):
        if text.startswith(op):
            value = float(text[len(op) :].strip())
            return ("=" if op == "==" else op, value)
    raise ValueError(f"Unable to parse comparator from '{text}'")


def build_summary_dashboard() -> Dict:
    widgets = []
    col_width = 4
    col_height = 2
    for idx, category in enumerate(METRIC_SCHEMA):
        widget = build_summary_widget(category, x=idx * col_width, width=col_width, height=col_height)
        widgets.append(widget)
    return {
        "title": "Metta Infra Health – Summary",
        "description": "Auto-generated summary view of workflow categories.",
        "layout_type": "ordered",
        "template_variables": [],
        "widgets": widgets,
    }


def build_summary_widget(category: CategoryDefinition, x: int, width: int, height: int) -> Dict:
    queries = []
    expressions = []
    counter = 1
    for workflow in category["workflows"]:
        for metric in workflow["metrics"]:
            query_name = f"m{counter}"
            queries.append(
                {
                    "name": query_name,
                    "data_source": "metrics",
                    "query": f"avg:{metric['metric']}{{*}}",
                }
            )
            clause = parse_condition(metric["condition"]).get("pass", {"comparator": ">=", "value": 0})
            expressions.append(f"({query_name} {clause['comparator']} {clause['value']})")
            counter += 1
    if not expressions:
        expressions = ["0"]
    formula = " * ".join(expressions)
    return {
        "definition": {
            "type": "query_value",
            "title": f"{category['category']} summary",
            "title_size": "16",
            "title_align": "left",
            "precision": 0,
            "requests": [
                {
                    "formulas": [{"formula": formula}],
                    "queries": queries,
                    "response_format": "scalar",
                    "conditional_formats": [
                        {"comparator": ">=", "value": 1, "palette": "green_on_white"},
                        {"comparator": "<", "value": 1, "palette": "red_on_white"},
                    ],
                }
            ],
            "autoscale": True,
            "time": {"live_span": "1h"},
        },
        "layout": {"x": x, "y": 0, "width": width, "height": height},
    }


def build_detailed_dashboard() -> Dict:
    widgets = []
    x = 0
    y = 0
    width = 6
    height = 6
    for category in METRIC_SCHEMA:
        for workflow in category["workflows"]:
            for metric in workflow["metrics"]:
                widget = build_timeseries_widget(category["category"], workflow["workflow"], metric)
                widget["layout"] = {"x": x, "y": y, "width": width, "height": height}
                widgets.append(widget)
                x += width
                if x >= 12:
                    x = 0
                    y += height
        if x != 0:
            x = 0
            y += height
    return {
        "title": "Metta Infra Health – Detailed",
        "description": "Auto-generated detailed view for all workflows and metrics.",
        "layout_type": "ordered",
        "template_variables": [],
        "widgets": widgets,
    }


def build_timeseries_widget(category: str, workflow: str, metric: MetricDefinition) -> Dict:
    queries = [
        {
            "name": "query1",
            "data_source": "metrics",
            "query": f"avg:{metric['metric']}{{*}}",
        }
    ]
    markers = build_markers(metric["condition"])
    title = f"{category} — {workflow} — {metric['task']} — {metric['check']} (Condition: {metric['condition']})"
    return {
        "definition": {
            "type": "timeseries",
            "title": title,
            "requests": [
                {
                    "display_type": "line",
                    "formulas": [{"formula": "query1"}],
                    "queries": queries,
                    "response_format": "timeseries",
                    "style": {"palette": "dog_classic", "line_type": "solid", "line_width": "normal"},
                }
            ],
            "markers": markers,
            "show_legend": False,
        }
    }


def build_markers(condition: str) -> List[Dict]:
    clauses = parse_condition(condition)
    markers: List[Dict] = []
    pass_clause = clauses.get("pass")
    warn_clause = clauses.get("warn")
    if pass_clause:
        markers.append(
            {
                "label": "pass boundary",
                "value": f"{pass_clause['comparator']} {pass_clause['value']}",
                "display_type": "ok",
            }
        )
    if warn_clause:
        markers.append(
            {
                "label": "warn",
                "value": f"{warn_clause['comparator']} {warn_clause['value']}",
                "display_type": "warning",
            }
        )
    return markers


def write_dashboard(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)
        fp.write("\n")


def main() -> None:
    summary = build_summary_dashboard()
    detailed = build_detailed_dashboard()
    write_dashboard(SUMMARY_PATH, summary)
    write_dashboard(DETAILED_PATH, detailed)
    print(f"Wrote {SUMMARY_PATH}")
    print(f"Wrote {DETAILED_PATH}")


if __name__ == "__main__":
    main()
