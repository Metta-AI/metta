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
    warn_threshold: float | None = None

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
            warn_threshold=8,
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
    data_availability_metrics = [
        MetricDefinition(
            display_name="Training data missing",
            metric_name="metta.infra.cron.training.data_missing",
            task="Data availability",
            check="S3 data found",
            comparator="<",
            threshold=1,
            aggregation="avg",
            description="Sentinel metric: 1 when training data is missing from S3, 0 when data is available.",
        ),
    ]

    multigpu_metrics = [
        MetricDefinition(
            display_name="Runs successfully",
            metric_name="metta.infra.cron.training.multigpu.runs_success",
            task="Runs successfully",
            check="Binary success",
            comparator=">=",
            threshold=1,
            aggregation="avg",
            description="Multigpu arena basic easy shaped run success signal.",
        ),
        MetricDefinition(
            display_name="Hearts",
            metric_name="metta.infra.cron.training.multigpu.hearts",
            task="Hearts",
            check="Avg hearts",
            comparator=">=",
            threshold=0.5,
            aggregation="avg",
            description="Average hearts for multigpu run.",
        ),
        MetricDefinition(
            display_name="SPS",
            metric_name="metta.infra.cron.training.multigpu.sps",
            task="SPS",
            check="Steps per second",
            comparator=">=",
            threshold=40000,
            aggregation="avg",
            description="Throughput (steps per second) for multigpu run.",
        ),
    ]

    multinode_metrics = [
        MetricDefinition(
            display_name="Runs successfully",
            metric_name="metta.infra.cron.training.multinode.runs_success",
            task="Runs successfully",
            check="Binary success",
            comparator=">=",
            threshold=1,
            aggregation="avg",
            description="Multinode learning progress run success signal.",
        ),
        MetricDefinition(
            display_name="Hearts",
            metric_name="metta.infra.cron.training.multinode.hearts",
            task="Hearts",
            check="Avg hearts",
            comparator=">=",
            threshold=0.5,
            aggregation="avg",
            description="Average hearts for multinode run.",
        ),
        MetricDefinition(
            display_name="Shaped",
            metric_name="metta.infra.cron.training.multinode.shaped",
            task="Shaped",
            check="Shaped SPS",
            comparator=">=",
            threshold=40000,
            aggregation="avg",
            description="Shaped throughput for multinode run.",
        ),
    ]

    local_arena_metrics = [
        MetricDefinition(
            display_name="Runs to first checkpoint at 10000 steps",
            metric_name="metta.infra.cron.training.local_arena.first_checkpoint",
            task="Runs to first checkpoint",
            check="Binary success",
            comparator=">=",
            threshold=1,
            aggregation="avg",
            description="Local arena basic easy shaped reaches first checkpoint.",
        ),
        MetricDefinition(
            display_name="Continues from checkpoint and runs another 10000 steps",
            metric_name="metta.infra.cron.training.local_arena.continues",
            task="Continues from checkpoint",
            check="Binary success",
            comparator=">=",
            threshold=1,
            aggregation="avg",
            description="Local arena continues after checkpoint.",
        ),
    ]

    bugs_metrics = [
        MetricDefinition(
            display_name='Num tickets in Bugs project with "Training" as workflow label',
            metric_name="metta.infra.cron.training.bugs.count",
            task="Bugs tickets",
            check="Count",
            comparator="<",
            threshold=1,
            aggregation="avg",
            description='Tickets in Bugs project labeled "Training".',
        ),
    ]

    return [
        WorkflowDefinition(
            name="training_data_availability",
            display_name="Training data availability",
            metrics=data_availability_metrics,
        ),
        WorkflowDefinition(
            name="multigpu_arena_basic_easy_shaped",
            display_name="Multigpu arena basic easy shaped",
            metrics=multigpu_metrics,
        ),
        WorkflowDefinition(
            name="multinode_learning_progress",
            display_name="Multinode learning progress",
            metrics=multinode_metrics,
        ),
        WorkflowDefinition(
            name="local_arena_basic_easy_shaped",
            display_name="Local arena basic easy shaped",
            metrics=local_arena_metrics,
        ),
        WorkflowDefinition(
            name="training_bugs",
            display_name="Bugs",
            metrics=bugs_metrics,
        ),
    ]


def _eval_workflows() -> List[WorkflowDefinition]:
    data_availability_metrics = [
        MetricDefinition(
            display_name="Eval data missing",
            metric_name="metta.infra.cron.eval.data_missing",
            task="Data availability",
            check="S3 eval artifacts found",
            comparator="<",
            threshold=1,
            aggregation="avg",
            description="Sentinel metric: 1 when eval data is missing from S3, 0 when data is available.",
        ),
    ]

    eval_metrics = [
        MetricDefinition(
            display_name="Local eval success",
            metric_name="metta.infra.cron.eval.local.success",
            task="Local eval",
            check="Success signal",
            comparator=">=",
            threshold=1,
            aggregation="avg",
            description="Binary success signal for local eval runs.",
        ),
        MetricDefinition(
            display_name="Local eval heart delta pct",
            metric_name="metta.infra.cron.eval.local.heart_delta_pct",
            task="Local eval",
            check="Heart delta pct",
            comparator=">=",
            threshold=0,
            aggregation="avg",
            description="Heart delta percentage for local eval.",
        ),
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
            display_name="Remote eval heart delta pct",
            metric_name="metta.infra.cron.eval.remote.heart_delta_pct",
            task="Remote eval",
            check="Heart delta pct",
            comparator=">=",
            threshold=0,
            aggregation="avg",
            description="Heart delta percentage for remote eval.",
        ),
        MetricDefinition(
            display_name="Remote eval duration minutes",
            metric_name="metta.infra.cron.eval.remote.duration_minutes",
            task="Remote eval",
            check="Runtime minutes",
            comparator="<=",
            threshold=60,
            warn_threshold=90,
            aggregation="avg",
            description="Runtime of remote eval workflows in minutes.",
        ),
    ]

    return [
        WorkflowDefinition(
            name="eval_data_availability",
            display_name="Eval data availability",
            metrics=data_availability_metrics,
        ),
        WorkflowDefinition(name="remote_eval", display_name="Remote eval", metrics=eval_metrics),
    ]


METRIC_SCHEMA: List[CategoryDefinition] = [
    CategoryDefinition(name="training", display_name="Training", workflows=_training_workflows()),
    CategoryDefinition(name="ci", display_name="CI", workflows=_ci_workflows()),
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
