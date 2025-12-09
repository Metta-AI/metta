from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Sequence

Comparator = Literal[">", ">=", "<", "<=", "=="]
Aggregation = Literal["avg", "sum", "min", "max"]


@dataclass(frozen=True)
class MetricDefinition:
    display_name: str
    metric_name: str
    task: str
    check: str
    comparator: Comparator
    threshold: float
    aggregation: Aggregation = "avg"
    description: str | None = None

    @property
    def condition_text(self) -> str:
        return f"{self.comparator} {self.threshold}"


@dataclass(frozen=True)
class WorkflowDefinition:
    name: str
    display_name: str
    metrics: Sequence[MetricDefinition]


@dataclass(frozen=True)
class CategoryDefinition:
    name: str
    display_name: str
    workflows: Sequence[WorkflowDefinition]


def _ci_workflows() -> List[WorkflowDefinition]:
    latest_state_metrics = [
        MetricDefinition(
            display_name="Tests that block merge passing",
            metric_name="metta.infra.cron.ci.workflow.tests_blocking_merge",
            task="Tests that block merge passing",
            check="Workflow passing signal",
            comparator=">=",
            threshold=1,
            aggregation="avg",
            description="Binary 1/0 metric indicating whether all blocking tests are passing.",
        ),
        MetricDefinition(
            display_name="Benchmarks passing",
            metric_name="metta.infra.cron.ci.workflow.benchmarks",
            task="Benchmarks passing",
            check="Workflow passing signal",
            comparator=">=",
            threshold=1,
            aggregation="avg",
            description="Binary 1/0 metric indicating benchmark workflow health.",
        ),
        MetricDefinition(
            display_name="Num other workflows failing",
            metric_name="metta.infra.cron.ci.workflow.other_failing",
            task="Other workflows failing",
            check="Count of failing workflows",
            comparator="<",
            threshold=2,
            aggregation="avg",
            description="Count of workflows on main that are currently failing.",
        ),
        MetricDefinition(
            display_name="Latest state of main success ratio",
            metric_name="metta.infra.cron.ci.workflow.success",
            task="Latest state success ratio",
            check="Binary success signal",
            comparator=">=",
            threshold=1,
            aggregation="avg",
            description="1 when 100% of runs in the observation window succeed, 0 otherwise.",
        ),
        MetricDefinition(
            display_name="Flaky test reruns",
            metric_name="metta.infra.cron.ci.workflow.flaky_tests",
            task="Flaky reruns per window",
            check="Count of reruns",
            comparator="<",
            threshold=10,
            aggregation="avg",
            description="Number of workflow reruns triggered by flaky behavior.",
        ),
    ]

    commit_history_metrics = [
        MetricDefinition(
            display_name="Weekly num hotfix commits",
            metric_name="metta.infra.cron.github.hotfix.count",
            task="Weekly hotfix commits",
            check="Hotfix count",
            comparator="<",
            threshold=5,
            aggregation="avg",
            description="Merged PRs labeled as hotfixes over the rolling weekly window.",
        ),
        MetricDefinition(
            display_name="Weekly num force merges",
            metric_name="metta.infra.cron.github.force_merge.count",
            task="Weekly force merges",
            check="Force merge count",
            comparator="<",
            threshold=7,
            aggregation="avg",
            description="Merged PRs that bypassed protections via force merge labels.",
        ),
        MetricDefinition(
            display_name="Weekly num reverts",
            metric_name="metta.infra.cron.github.reverts.count",
            task="Weekly reverts",
            check="Revert count",
            comparator="<=",
            threshold=1,
            aggregation="avg",
            description="Reverts merged into main over the rolling weekly window.",
        ),
    ]

    ci_smoothness_metrics = [
        MetricDefinition(
            display_name="P90 pre-merge CI checks duration minutes",
            metric_name="metta.infra.cron.ci.workflow.duration.p90",
            task="Pre-merge duration",
            check="P90 length (minutes)",
            comparator="<",
            threshold=5,
            aggregation="avg",
            description="P90 runtime of CI workflows in minutes.",
        ),
        MetricDefinition(
            display_name="Weekly num jobs canceled due to timeout",
            metric_name="metta.infra.cron.ci.workflow.cancelled",
            task="Jobs cancelled",
            check="Cancellation count",
            comparator="<",
            threshold=10,
            aggregation="avg",
            description="Number of CI jobs cancelled during the observation window.",
        ),
    ]

    return [
        WorkflowDefinition(
            name="latest_state_of_main",
            display_name="Latest state of main",
            metrics=latest_state_metrics,
        ),
        WorkflowDefinition(
            name="commit_history",
            display_name="Commit history",
            metrics=commit_history_metrics,
        ),
        WorkflowDefinition(
            name="ci_smoothness",
            display_name="CI smoothness",
            metrics=ci_smoothness_metrics,
        ),
    ]


def _training_workflows() -> List[WorkflowDefinition]:
    core_metrics = [
        MetricDefinition(
            display_name="StableSuite SPS",
            metric_name="metta.infra.stablesuite.sps",
            task="Throughput",
            check="Steps per second",
            comparator=">=",
            threshold=40000,
            aggregation="avg",
            description="Steps per second from StableSuite runs.",
        ),
        MetricDefinition(
            display_name="StableSuite Hearts",
            metric_name="metta.infra.stablesuite.hearts",
            task="Hearts",
            check="Avg hearts",
            comparator=">=",
            threshold=0.5,
            aggregation="avg",
            description="Mean hearts across agents; must stay above threshold.",
        ),
        MetricDefinition(
            display_name="StableSuite failure ratio",
            metric_name="metta.infra.stablesuite.failure_ratio",
            task="Failure ratio",
            check="Failure ratio",
            comparator="<",
            threshold=0.2,
            aggregation="avg",
            description="Failure ratio for StableSuite runs; lower is better.",
        ),
    ]

    pipeline_metrics = [
        MetricDefinition(
            display_name="Training pipeline success",
            metric_name="metta.infra.stablesuite.training.pipeline.success",
            task="Pipeline status",
            check="Binary success",
            comparator=">=",
            threshold=1,
            aggregation="avg",
            description="1 if the latest training pipeline run succeeded, 0 otherwise.",
        ),
        MetricDefinition(
            display_name="Training pipeline runtime (minutes)",
            metric_name="metta.infra.stablesuite.training.pipeline.runtime",
            task="Pipeline runtime",
            check="Runtime minutes",
            comparator="<=",
            threshold=90,
            aggregation="avg",
            description="Total runtime of the training pipeline in minutes.",
        ),
        MetricDefinition(
            display_name="Training environment checks passing",
            metric_name="metta.infra.stablesuite.training.env.checks",
            task="Environment checks",
            check="Checks passing",
            comparator=">=",
            threshold=1,
            aggregation="avg",
            description="Binary signal indicating environment checks are passing.",
        ),
    ]

    return [
        WorkflowDefinition(name="stablesuite_core", display_name="StableSuite core", metrics=core_metrics),
        WorkflowDefinition(name="training_pipeline", display_name="Training pipeline", metrics=pipeline_metrics),
    ]


def _eval_workflows() -> List[WorkflowDefinition]:
    eval_metrics = [
        MetricDefinition(
            display_name="Remote eval success",
            metric_name="metta.infra.cron.eval.remote.success",
            task="Remote eval",
            check="Success signal",
            comparator=">=",
            threshold=1,
            aggregation="avg",
            description="Binary success signal for remote eval runs.",
        ),
        MetricDefinition(
            display_name="Remote eval failures",
            metric_name="metta.infra.cron.eval.remote.failure",
            task="Remote eval",
            check="Failure count",
            comparator="<",
            threshold=1,
            aggregation="avg",
            description="Failures observed in remote eval; should remain zero.",
        ),
        MetricDefinition(
            display_name="Eval score",
            metric_name="metta.infra.cron.eval.score",
            task="Eval quality",
            check="Score",
            comparator=">=",
            threshold=0.5,
            aggregation="avg",
            description="Evaluation score aggregated across runs.",
        ),
        MetricDefinition(
            display_name="Eval runtime (minutes)",
            metric_name="metta.infra.cron.eval.runtime",
            task="Eval runtime",
            check="Runtime minutes",
            comparator="<=",
            threshold=60,
            aggregation="avg",
            description="Runtime of eval workflows in minutes.",
        ),
    ]

    return [WorkflowDefinition(name="remote_eval", display_name="Remote eval", metrics=eval_metrics)]


METRIC_SCHEMA: List[CategoryDefinition] = [
    CategoryDefinition(name="ci", display_name="CI", workflows=_ci_workflows()),
    CategoryDefinition(name="training", display_name="Training", workflows=_training_workflows()),
    CategoryDefinition(name="evaluation", display_name="Eval", workflows=_eval_workflows()),
]


__all__ = [
    "Aggregation",
    "CategoryDefinition",
    "Comparator",
    "METRIC_SCHEMA",
    "MetricDefinition",
    "WorkflowDefinition",
]
