"""Ensure default-zero env metrics are injected before rolling averages."""

from types import SimpleNamespace
from typing import Any

from metta.rl.training.stats_reporter import StatsReporter, StatsReporterConfig


class _DummyTimer:
    def get_all_elapsed(self) -> dict[str, float]:
        return {}

    def get_elapsed(self) -> float:
        return 0.0

    def lap_all(self, agent_step: int, exclude_global: bool = False) -> dict[str, float]:
        return {"global": 0.0}

    def get_lap_steps(self) -> int:
        return 0

    def get_rate(self, agent_step: int) -> float:
        return 0.0

    def __call__(self, name: str):
        # Context manager stub
        from contextlib import nullcontext

        return nullcontext()


class _DummyExperience:
    def stats(self) -> dict[str, Any]:
        return {}


class _DummyOptimizer:
    param_groups = []


def _make_reporter() -> StatsReporter:
    cfg = StatsReporterConfig()
    reporter = StatsReporter(config=cfg, wandb_run=None)
    # Minimal context needed for _collect_parameters
    reporter._context = SimpleNamespace(  # type: ignore[attr-defined, assignment]
        config=SimpleNamespace(optimizer=SimpleNamespace(learning_rate=0.0, type="adam")),
        stopwatch=_DummyTimer(),
        experience=_DummyExperience(),
        policy=None,
        optimizer=_DummyOptimizer(),
        epoch=0,
        agent_step=0,
        run_name=None,
    )
    return reporter


def test_missing_metric_is_zero_filled() -> None:
    reporter = _make_reporter()

    payload = reporter._build_wandb_payload(
        losses_stats={},
        experience=reporter.context.experience,
        trainer_cfg=reporter.context.config,
        policy=None,
        agent_step=0,
        epoch=0,
        timer=reporter.context.stopwatch,
        optimizer=reporter.context.optimizer,
    )

    assert payload["env_agent/heart.gained"] == 0.0


def test_existing_metric_is_not_overwritten() -> None:
    reporter = _make_reporter()
    # Preload rollout stats to simulate env emission
    reporter._state.rollout_stats["agent/heart.gained"] = [5.0]

    payload = reporter._build_wandb_payload(
        losses_stats={},
        experience=reporter.context.experience,
        trainer_cfg=reporter.context.config,
        policy=None,
        agent_step=0,
        epoch=0,
        timer=reporter.context.stopwatch,
        optimizer=reporter.context.optimizer,
    )

    assert payload["env_agent/heart.gained"] == 5.0
