"""Helpers for formatting and replaying evaluation results."""

from __future__ import annotations

import logging
import re
import shlex
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from rich.console import Console
from rich.table import Table

from metta.eval.eval_request_config import EvalResults
from metta.rl.training.progress_logger import should_use_rich_console
from metta.sim.runner import (
    SimulationRunResult,
)
from metta.sim.simulation_config import SimulationConfig
from mettagrid.types import EpisodeStats

logger = logging.getLogger(__name__)

HEART_STAT_KEYS: tuple[str, ...] = (
    "heart.gained",
    "chest.heart.deposited",
    "hearts.delivered",
    "heart.delivered",
)


@dataclass(slots=True)
class EvalSummaryRow:
    suite: str
    name: str
    reward: float
    hearts: float | None
    replay_url: str | None
    repro_command: str | None

    @property
    def label(self) -> str:
        return f"{self.suite}/{self.name}"


def build_eval_summary_rows(
    *,
    policy_uri: str,
    simulations: list[SimulationConfig],
    rollout_results: list[SimulationRunResult],
    evaluation_results: EvalResults,
    replay_dir: str | None,
) -> list[EvalSummaryRow]:
    rows: list[EvalSummaryRow] = []
    for sim_config, run_result in zip(simulations, rollout_results, strict=True):
        reward = evaluation_results.scores.simulation_scores.get((sim_config.suite, sim_config.name), 0.0)
        hearts = _extract_heart_statistic(run_result.results.stats)
        replay_url = _select_replay_url(sim_config, run_result, evaluation_results)
        config_path = _persist_simulation_config(sim_config, replay_dir)
        repro_command = _build_repro_command(config_path, policy_uri)
        rows.append(
            EvalSummaryRow(
                suite=sim_config.suite,
                name=sim_config.name,
                reward=reward,
                hearts=hearts,
                replay_url=replay_url,
                repro_command=repro_command,
            )
        )
    return rows


def render_eval_summary(rows: Iterable[EvalSummaryRow]) -> None:
    rows = list(rows)
    if not rows:
        return

    if should_use_rich_console():
        console = Console()
        table = Table(
            title="[bold cyan]Local Evaluation Summary[/bold cyan]",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Simulation", style="cyan")
        table.add_column("Hearts", justify="right", style="green")
        table.add_column("Reward", justify="right", style="yellow")
        table.add_column("Replay", style="bright_blue")
        table.add_column("Local Repro", style="magenta")
        for row in rows:
            reward_display = f"{row.reward:.3f}"
            hearts_display = f"{row.hearts:.2f}" if row.hearts is not None else "—"
            table.add_row(
                row.label,
                hearts_display,
                reward_display,
                row.replay_url or "—",
                row.repro_command or "—",
            )
        console.print(table)
        return

    for row in rows:
        reward_display = f"{row.reward:.3f}"
        hearts_display = f"{row.hearts:.2f}" if row.hearts is not None else "n/a"
        logger.info(
            "Eval %s | reward=%s hearts=%s replay=%s repro=%s",
            row.label,
            reward_display,
            hearts_display,
            row.replay_url or "—",
            row.repro_command or "—",
        )


def _extract_heart_statistic(stats: list[EpisodeStats]) -> float | None:
    values: list[float] = []
    for episode in stats:
        game_stats = episode.get("game", {})
        for key in HEART_STAT_KEYS:
            if key in game_stats:
                values.append(float(game_stats[key]))
                break
    if not values:
        return None
    return sum(values) / len(values)


def _select_replay_url(
    sim_config: SimulationConfig,
    run_result: SimulationRunResult,
    evaluation_results: EvalResults,
) -> str | None:
    if run_result.replay_urls:
        iterator = iter(run_result.replay_urls.values())
        return next(iterator, None)
    lookup_key = f"{sim_config.suite}.{sim_config.name}"
    urls = evaluation_results.replay_urls.get(lookup_key)
    if urls:
        return urls[0]
    return None


def _build_repro_command(config_path: Path | None, policy_uri: str) -> str | None:
    if not config_path:
        return None
    quoted_config = shlex.quote(str(config_path))
    quoted_policy = shlex.quote(policy_uri)
    return (
        "uv run python -m metta.rl.training.format_eval_results "
        f"--sim-config {quoted_config} --policy-uri {quoted_policy}"
    )


def _persist_simulation_config(sim_config: SimulationConfig, replay_dir: str | None) -> Path | None:
    if not replay_dir:
        return None

    try:
        root = Path(replay_dir).expanduser()
        target_dir = root / "sim_configs"
        target_dir.mkdir(parents=True, exist_ok=True)
        slug = _slugify(f"{sim_config.suite}_{sim_config.name}")
        file_path = target_dir / f"{slug}_{uuid.uuid4().hex[:6]}.json"
        file_path.write_text(sim_config.model_dump_json(indent=2), encoding="utf-8")
        return file_path
    except Exception as exc:  # pragma: no cover - best effort logging only
        logger.debug("Unable to persist simulation config for %s/%s: %s", sim_config.suite, sim_config.name, exc)
        return None


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9_-]+", "-", value.lower()).strip("-")
    return slug or "simulation"
