"""Minimal checks that default-zero env metrics are injected and not overwritten."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from metta.rl.training.stats_reporter import StatsReporter, StatsReporterConfig


def _reporter(existing_heart: float | None = None) -> StatsReporter:
    cfg = StatsReporterConfig()
    reporter = StatsReporter(config=cfg, wandb_run=None)

    timer = SimpleNamespace(
        get_all_elapsed=lambda: {},
        get_elapsed=lambda: 0.0,
        lap_all=lambda agent_step, exclude_global=False: {"global": 0.0},
        get_lap_steps=lambda: 0,
        get_rate=lambda agent_step: 0.0,
        __call__=lambda self, name: nullcontext(),
    )

    reporter._context = SimpleNamespace(  # type: ignore[attr-defined, assignment]
        config=SimpleNamespace(
            optimizer=SimpleNamespace(learning_rate=0.0, type="adam"),
            nodes={},
        ),
        stopwatch=timer,
        experience=SimpleNamespace(stats=lambda: {}),
        policy=None,
        optimizer=None,
        epoch=0,
        agent_step=0,
        run_name=None,
    )

    if existing_heart is not None:
        reporter._state.rollout_stats["game/assembler.heart.created"] = [existing_heart]

    return reporter


@pytest.mark.parametrize(
    ("existing", "expected"),
    [(None, 0.0), (5.0, 5.0)],
)
def test_heart_metric_zero_fill_and_preserve(existing: float | None, expected: float) -> None:
    reporter = _reporter(existing)

    payload = reporter._build_wandb_payload(
        graph_stats={},
        experience=reporter.context.experience,
        trainer_cfg=reporter.context.config,
        policy=None,
        agent_step=0,
        epoch=0,
        timer=reporter.context.stopwatch,
        optimizer=SimpleNamespace(param_groups=[]),
    )

    assert payload["env_game/assembler.heart.created"] == expected
