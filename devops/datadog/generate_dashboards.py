from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

from devops.datadog.metric_schema import (
    METRIC_SCHEMA,
    CategoryDefinition,
    MetricDefinition,
    WorkflowDefinition,
)

DASHBOARD_DIR = Path(__file__).parent / "dashboards"
SUMMARY_PATH = DASHBOARD_DIR / "infra_summary.json"
DETAILED_PATH = DASHBOARD_DIR / "infra_detailed.json"
COLLECTOR_DIR = Path(__file__).parent / "collectors"

VALID_AGGREGATIONS = {"avg", "sum", "min", "max"}
ALLOWED_PREFIXES = (
    "metta.infra.cron.ci.",
    "metta.infra.cron.eval.",
    "metta.infra.cron.training.",
    "metta.infra.stablesuite.",
)


def _format_float(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value}".rstrip("0").rstrip(".") if isinstance(value, float) else str(value)


def _flatten_metrics(workflows: Sequence[WorkflowDefinition]) -> List[MetricDefinition]:
    metrics: List[MetricDefinition] = []
    for workflow in workflows:
        metrics.extend(list(workflow.metrics))
    return metrics


def _load_collector_sources() -> str:
    contents = []
    for path in COLLECTOR_DIR.glob("*.py"):
        contents.append(path.read_text(encoding="utf-8"))
    return "\n".join(contents)


def _metric_defined(metric_name: str, corpus: str) -> bool:
    if metric_name.startswith(ALLOWED_PREFIXES):
        return True
    terms = {metric_name}
    if "metta." in metric_name:
        terms.add(metric_name.split("metta.", 1)[1])
    if "cron." in metric_name:
        terms.add(metric_name.split("cron.", 1)[1])
    if "stablesuite." in metric_name:
        terms.add(metric_name.split("stablesuite.", 1)[1])
    return any(term and term in corpus for term in terms)


def validate_schema(schema: Sequence[CategoryDefinition]) -> None:
    corpus = _load_collector_sources()
    if not corpus:
        raise RuntimeError("Unable to read collector sources for schema validation.")
    for category in schema:
        for workflow in category.workflows:
            for metric in workflow.metrics:
                if metric.aggregation not in VALID_AGGREGATIONS:
                    raise ValueError(f"Unsupported aggregation '{metric.aggregation}' for {metric.metric_name}")
                if not metric.metric_name.startswith("metta."):
                    raise ValueError(f"Metric names must start with 'metta.': {metric.metric_name}")
                if not _metric_defined(metric.metric_name, corpus):
                    raise ValueError(
                        f"Metric {metric.metric_name} not found in collector sources. "
                        "Update the schema or emit the metric first."
                    )


def build_summary_dashboard() -> Dict:
    widgets = []
    col_width = 4
    col_height = 2
    slot = 0
    for category in METRIC_SCHEMA:
        metrics = _flatten_metrics(category.workflows)
        if not metrics:
            continue
        widget = build_summary_widget(
            category=category,
            metrics=metrics,
            x=slot * col_width,
            width=col_width,
            height=col_height,
        )
        widgets.append(widget)
        slot += 1
    return {
        "title": "Metta Infra Health – Summary",
        "description": "Auto-generated summary view of workflow categories.",
        "layout_type": "ordered",
        "template_variables": [],
        "widgets": widgets,
    }


def build_summary_widget(
    *,
    category: CategoryDefinition,
    metrics: Sequence[MetricDefinition],
    x: int,
    width: int,
    height: int,
) -> Dict:
    # Query all metrics (no status filter) to show both pass and fail
    # Use two queries per metric: one for pass, one for fail
    queries = []
    pass_terms: List[str] = []
    fail_terms: List[str] = []

    for idx, metric in enumerate(metrics, start=1):
        # Query for passing metrics
        pass_query_name = f"pass_query{idx}"
        queries.append(
            {
                "name": pass_query_name,
                "data_source": "metrics",
                "query": f"{metric.aggregation}:{metric.metric_name}{{status:pass}}",
            }
        )
        pass_terms.append(f"default_zero({pass_query_name})")

        # Query for failing metrics
        fail_query_name = f"fail_query{idx}"
        queries.append(
            {
                "name": fail_query_name,
                "data_source": "metrics",
                "query": f"{metric.aggregation}:{metric.metric_name}{{status:fail}}",
            }
        )
        fail_terms.append(f"default_zero({fail_query_name})")

    # Calculate total pass and fail counts
    pass_expression = "sum(" + ", ".join(pass_terms) + ")" if pass_terms else "0"
    fail_expression = "sum(" + ", ".join(fail_terms) + ")" if fail_terms else "0"

    # Display: show fail count, but color based on whether there are any failures
    # Formula: if fail_count > 0, show fail_count in red, else show 0 in green
    display_formula = f"if({fail_expression} > 0, {fail_expression}, 0)"

    return {
        "definition": {
            "type": "query_value",
            "title": f"{category.display_name} summary",
            "title_size": "16",
            "title_align": "left",
            "precision": 0,
            "requests": [
                {
                    "formulas": [{"formula": display_formula}],
                    "queries": queries,
                    "response_format": "scalar",
                    "conditional_formats": [
                        {
                            "comparator": "<",
                            "value": 1,
                            "palette": "green_on_white",
                        },
                        {
                            "comparator": ">=",
                            "value": 1,
                            "palette": "red_on_white",
                        },
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
    height = 4
    for category in METRIC_SCHEMA:
        for workflow in category.workflows:
            for metric in workflow.metrics:
                widget = build_timeseries_widget(
                    category=category.display_name,
                    workflow=workflow.display_name,
                    metric=metric,
                )
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


def build_timeseries_widget(*, category: str, workflow: str, metric: MetricDefinition) -> Dict:
    queries = [
        {
            "name": "query1",
            "data_source": "metrics",
            "query": f"{metric.aggregation}:{metric.metric_name}{{*}}",
        }
    ]
    condition_text = f"{metric.comparator} {_format_float(metric.threshold)}"
    title = f"{category} — {workflow} — {metric.task} — {metric.check} (Condition: {condition_text})"
    markers = [
        {
            "label": "pass boundary",
            "value": condition_text,
            "display_type": "ok",
        }
    ]
    if metric.warn_threshold is not None:
        markers.append(
            {
                "label": "warn",
                "value": f"{metric.comparator} {_format_float(metric.warn_threshold)}",
                "display_type": "warning",
            }
        )
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


def write_dashboard(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)
        fp.write("\n")


def main() -> None:
    validate_schema(METRIC_SCHEMA)
    summary = build_summary_dashboard()
    detailed = build_detailed_dashboard()
    write_dashboard(SUMMARY_PATH, summary)
    write_dashboard(DETAILED_PATH, detailed)
    print(f"Wrote {SUMMARY_PATH}")
    print(f"Wrote {DETAILED_PATH}")


if __name__ == "__main__":
    main()
