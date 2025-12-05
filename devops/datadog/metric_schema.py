from __future__ import annotations

"""
Central definition of all metrics that feed the Datadog infrastructure health dashboards.

Each entry describes a workflow category, its workflows, and the metrics (with success
conditions) that populate both the summary and detailed dashboards.
"""

from typing import List, TypedDict


class MetricDefinition(TypedDict):
    metric: str
    task: str
    check: str
    condition: str


class WorkflowDefinition(TypedDict):
    workflow: str
    metrics: List[MetricDefinition]


class CategoryDefinition(TypedDict):
    category: str
    workflows: List[WorkflowDefinition]


METRIC_SCHEMA: List[CategoryDefinition] = [
    {
        "category": "Training",
        "workflows": [
            {
                "workflow": "Multigpu arena basic easy shaped",
                "metrics": [
                    {
                        "metric": "container.cpu.usage",
                        "task": "Multigpu arena basic easy shaped",
                        "check": "Runs successfully",
                        "condition": "> 0",
                    },
                    {
                        "metric": "container.cpu.limit",
                        "task": "Multigpu arena basic easy shaped",
                        "check": "Hearts",
                        "condition": "> 0.5",
                    },
                    {
                        "metric": "container.cpu.throttled",
                        "task": "Multigpu arena basic easy shaped",
                        "check": "SPS",
                        "condition": "> 40000",
                    },
                ],
            },
            {
                "workflow": "Multinode learning progress",
                "metrics": [
                    {
                        "metric": "container.cpu.partial_stall",
                        "task": "Multinode learning progress",
                        "check": "Runs successfully",
                        "condition": "> 0",
                    },
                    {
                        "metric": "container.cpu.system",
                        "task": "Multinode learning progress",
                        "check": "Hearts",
                        "condition": "> 0.5",
                    },
                    {
                        "metric": "container.cpu.throttled.periods",
                        "task": "Multinode learning progress",
                        "check": "Shaped",
                        "condition": "> 40000",
                    },
                ],
            },
            {
                "workflow": "Local arena basic easy shaped",
                "metrics": [
                    {
                        "metric": "cri.uptime",
                        "task": "Local arena basic easy shaped",
                        "check": "Runs to first checkpoint at 10000 steps",
                        "condition": "> 0",
                    },
                    {
                        "metric": "datadog.agent.started",
                        "task": "Local arena basic easy shaped",
                        "check": "Continues from checkpoint and runs another 10000 steps",
                        "condition": "> 0",
                    },
                ],
            },
            {
                "workflow": "Bugs",
                "metrics": [
                    {
                        "metric": "datadog.agent.running",
                        "task": "Bugs",
                        "check": "Num tickets in Bugs project with \"Training\" label",
                        "condition": "< 1|warn< 3",
                    },
                ],
            },
        ],
    },
    {
        "category": "CI",
        "workflows": [
            {
                "workflow": "Latest state of main",
                "metrics": [
                    {
                        "metric": "commits.hotfix",
                        "task": "Latest state of main",
                        "check": "Tests that block merge passing",
                        "condition": "> 0",
                    },
                    {
                        "metric": "commits.reverts",
                        "task": "Latest state of main",
                        "check": "Benchmarks passing",
                        "condition": "> 0",
                    },
                    {
                        "metric": "metta.infra.ci.workflow.other_failing",
                        "task": "Latest state of main",
                        "check": "Num other workflows whose latest run off main is failing",
                        "condition": "< 2|warn< 4",
                    },
                ],
            },
            {
                "workflow": "Commit history",
                "metrics": [
                    {
                        "metric": "metta.infra.ci.github.hotfix.count",
                        "task": "Commit history",
                        "check": "Weekly num hotfix commits",
                        "condition": "< 5|warn< 7",
                    },
                    {
                        "metric": "metta.infra.ci.github.force_merge.count",
                        "task": "Commit history",
                        "check": "Weekly num force merges",
                        "condition": "< 7|warn< 10",
                    },
                    {
                        "metric": "metta.infra.ci.github.reverts.count",
                        "task": "Commit history",
                        "check": "Weekly num reverts",
                        "condition": "< 1|warn< 2",
                    },
                ],
            },
            {
                "workflow": "CI smoothness",
                "metrics": [
                    {
                        "metric": "metta.infra.ci.workflow.duration.p90",
                        "task": "CI smoothness",
                        "check": "P90 pre-merge CI checks duration minutes",
                        "condition": "< 5|warn< 8",
                    },
                    {
                        "metric": "metta.infra.ci.workflow.cancelled",
                        "task": "CI smoothness",
                        "check": "Weekly num jobs canceled due to timeout",
                        "condition": "< 10|warn< 15",
                    },
                    {
                        "metric": "metta.infra.ci.workflow.flaky_tests",
                        "task": "CI smoothness",
                        "check": "Weekly num times a check failed then succeeded",
                        "condition": "< 10|warn< 15",
                    },
                ],
            },
        ],
    },
    {
        "category": "Eval",
        "workflows": [
            {
                "workflow": "Local runs",
                "metrics": [
                    {
                        "metric": "metta.infra.eval.local.success",
                        "task": "Local runs",
                        "check": "`./tools/run.py …` exits with code 0",
                        "condition": "> 0",
                    },
                    {
                        "metric": "metta.infra.eval.local.heart_delta_pct",
                        "task": "Local runs",
                        "check": "`./tools/run.py …` avg hearts % diff",
                        "condition": "< 10|warn< 15",
                    },
                ],
            },
            {
                "workflow": "Runs remotely meets known bar",
                "metrics": [
                    {
                        "metric": "metta.infra.eval.remote.success",
                        "task": "Runs remotely meets known bar",
                        "check": "`./tools/request_eval.py …` succeeds",
                        "condition": "> 0",
                    },
                    {
                        "metric": "metta.infra.eval.remote.heart_delta_pct",
                        "task": "Runs remotely meets known bar",
                        "check": "`./tools/request_eval.py …` avg hearts % diff",
                        "condition": "< 10|warn< 15",
                    },
                    {
                        "metric": "metta.infra.eval.remote.duration_minutes",
                        "task": "Runs remotely meets known bar",
                        "check": "`./tools/request_eval.py …` duration minutes",
                        "condition": "<= 5|warn<= 7",
                    },
                ],
            },
        ],
    },
]
