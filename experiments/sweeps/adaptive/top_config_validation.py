"""Top-config validation experiment.

Select the top-K configurations from a WandB sweep group (using
summary.sweep/suggestion.* for the config), run each config for N
random seeds to a fixed training horizon, then evaluate with many
episodes and produce CSVs + plots.

Usage example:
  uv run ./tools/run.py experiments.sweeps.adaptive.top_config_validation.validate \
    sweep_name=my_sweep \
    K=5 N=3 \
    metric=overview/reward \
    target_timesteps=5000000000 \
    recipe_module=experiments.recipes.arena_basic_easy_shaped \
    train_entrypoint=train \
    eval_entrypoint=evaluate \
    eval_num_episodes=30 \
    --out-dir ./train_dir/validation/my_sweep
"""

from __future__ import annotations

import importlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from metta.adaptive import AdaptiveConfig, AdaptiveController
from metta.adaptive.dispatcher import LocalDispatcher, SkypilotDispatcher
from metta.adaptive.models import JobDefinition, JobTypes, RunInfo
from metta.adaptive.protocols import ExperimentScheduler
from metta.adaptive.stores import WandbStore
from metta.common.util.constants import (
    METTA_WANDB_ENTITY,
    METTA_WANDB_PROJECT,
    SOFTMAX_S3_POLICY_PREFIX,
)


def _flatten_suggestion_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Extract sweep/suggestion.* keys from a summary and map to dotted overrides.

    Example: {"sweep/suggestion.trainer.batch_size": 524288} ->
             {"trainer.batch_size": 524288}
    """
    out: dict[str, Any] = {}
    for k, v in (summary or {}).items():
        if not isinstance(k, str):
            continue
        if k.startswith("sweep/suggestion."):
            out[k[len("sweep/suggestion.") :]] = v
    return out


def _select_top_k(runs: List[RunInfo], metric: str, k: int) -> List[RunInfo]:
    with_metric: List[RunInfo] = []
    for r in runs:
        summary = r.summary or {}
        if metric in summary:
            try:
                float(summary[metric])
                with_metric.append(r)
            except Exception:
                pass

    if not with_metric:
        return []

    with_metric.sort(
        key=lambda rr: float((rr.summary or {}).get(metric, float("-inf"))),
        reverse=True,
    )
    return with_metric[:k]


def _infer_eval_sim_count(recipe_module: str, eval_entrypoint: str) -> int:
    mod = importlib.import_module(recipe_module)
    fn = getattr(mod, eval_entrypoint)
    tool = fn()  # type: ignore[call-arg]
    sims = getattr(tool, "simulations", None)
    return len(list(sims)) if sims is not None else 0


def _make_eval_overrides(sim_count: int, episodes: int) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    # Set num_episodes for all simulations if possible
    for i in range(sim_count):
        overrides[f"simulations.{i}.num_episodes"] = int(episodes)
    return overrides


@dataclass
class ValidationPlan:
    sweep_name: str
    num_configs: int
    num_seeds: int
    metric: str
    target_timesteps: int
    recipe_module: str
    train_entrypoint: str
    eval_entrypoint: str
    eval_num_episodes: int
    dispatcher_type: str  # "local" or "skypilot"
    gpus: int
    nodes: int
    max_parallel: int
    monitoring_interval: int
    local_test: bool
    out_dir: str | None


class TopConfigValidationScheduler(ExperimentScheduler):
    """Schedules training for K configs × N seeds, then high-episode eval.

    - Uses sweep suggestion fields to construct training overrides.
    - New run IDs: {sweep_name}.final_validation.cfg{idx:02d}.seed{seed}
    - Group for all jobs: {sweep_name}.final_validation
    - Schedules eval with policy_uri=:latest and num_episodes overrides.
    - Finishes when all evals have been dispatched and runs are present.
    """

    def __init__(self, plan: ValidationPlan, store: WandbStore):
        self.plan = plan
        self.store = store
        self.source_group = plan.sweep_name
        self.validation_group = f"{plan.sweep_name}.final_validation"
        self._seeds: List[int] = []
        self._winners: List[RunInfo] | None = None
        self._scheduled_train: set[str] = set()
        self._scheduled_eval: set[str] = set()
        self._sim_count_cache: int | None = None

    def _ensure_seeds(self) -> None:
        if self._seeds:
            return
        # Deterministic seed generation
        rng = random.Random(1337)
        self._seeds = sorted(
            {rng.randint(1, 2**31 - 1) for _ in range(self.plan.num_seeds)}
        )

    def _select_winners(self) -> None:
        if self._winners is not None:
            return
        runs = self.store.fetch_runs(filters={"group": self.source_group})
        winners = _select_top_k(runs, self.plan.metric, self.plan.num_configs)
        self._winners = winners

    def _winner_overrides(self, run: RunInfo) -> dict[str, Any]:
        return _flatten_suggestion_summary(run.summary or {})

    def _train_run_id(self, cfg_idx: int, seed: int) -> str:
        return f"{self.validation_group}.cfg{cfg_idx:02d}.seed{seed}"

    def _policy_uri_for_run(self, run_id: str) -> str:
        if self.plan.local_test:
            # Local trainer dir policy path
            run_dir = Path("./train_dir") / run_id / "checkpoints"
            return f"file://{run_dir.resolve()}/:latest"
        # Assume remote S3 prefix for production
        return f"{SOFTMAX_S3_POLICY_PREFIX}/{run_id}/:latest"

    def _sim_count(self) -> int:
        if self._sim_count_cache is None:
            try:
                self._sim_count_cache = _infer_eval_sim_count(
                    self.plan.recipe_module, self.plan.eval_entrypoint
                )
            except Exception:
                self._sim_count_cache = 0
        return self._sim_count_cache

    def schedule(
        self, runs: List[RunInfo], available_training_slots: int
    ) -> List[JobDefinition]:
        self._ensure_seeds()
        self._select_winners()

        jobs: List[JobDefinition] = []

        if not self._winners:
            return jobs

        # Build lookup of current runs in validation group
        current = {r.run_id: r for r in runs if r.group == self.validation_group}

        # 1) Schedule missing training jobs
        for cfg_idx, winner in enumerate(self._winners):
            overrides_base = self._winner_overrides(winner)

            for seed in self._seeds:
                run_id = self._train_run_id(cfg_idx, seed)
                if len(jobs) >= available_training_slots:
                    break
                if run_id in self._scheduled_train:
                    continue
                r = current.get(run_id)
                if (
                    r is not None
                    and r.has_started_training
                    and not r.has_completed_training
                ):
                    continue

                # Construct overrides
                overrides: dict[str, Any] = dict(overrides_base)
                overrides["trainer.total_timesteps"] = int(
                    50000 if self.plan.local_test else self.plan.target_timesteps
                )
                overrides["training_env.seed"] = int(seed)

                job = JobDefinition(
                    run_id=run_id,
                    cmd=f"{self.plan.recipe_module}.{self.plan.train_entrypoint}",
                    gpus=self.plan.gpus,
                    nodes=self.plan.nodes,
                    args={
                        "run": run_id,
                        "group": self.validation_group,
                    },
                    overrides=overrides,
                    type=JobTypes.LAUNCH_TRAINING,
                    metadata={
                        "validation/from_group": self.source_group,
                        "validation/cfg_index": cfg_idx,
                        "validation/seed": seed,
                        "validation/metric": self.plan.metric,
                    },
                )
                jobs.append(job)
                self._scheduled_train.add(run_id)

        # 2) Schedule evaluations for completed training
        sim_count = self._sim_count()
        eval_overrides = _make_eval_overrides(
            sim_count, 2 if self.plan.local_test else self.plan.eval_num_episodes
        )

        for run_id, r in current.items():
            if not (r.has_completed_training and not r.has_been_evaluated):
                continue
            if run_id in self._scheduled_eval:
                continue

            policy_uri = self._policy_uri_for_run(run_id)
            eval_job = JobDefinition(
                run_id=run_id,
                cmd=f"{self.plan.recipe_module}.{self.plan.eval_entrypoint}",
                args={"policy_uri": policy_uri, "group": self.validation_group},
                overrides=eval_overrides,
                type=JobTypes.LAUNCH_EVAL,
                metadata={"validation/policy_uri": policy_uri},
            )
            jobs.append(eval_job)
            self._scheduled_eval.add(run_id)

        return jobs

    def is_experiment_complete(self, runs: List[RunInfo]) -> bool:
        self._select_winners()
        if not self._winners:
            return True

        self._ensure_seeds()
        total_expected = len(self._winners) * len(self._seeds)

        # Consider it complete when all expected runs exist in the validation group
        val_runs = [r for r in runs if r.group == self.validation_group]
        if len(val_runs) < total_expected:
            return False

        # And all have been evaluated
        for r in val_runs:
            if not r.has_been_evaluated:
                return False
        return True


def _fetch_timeseries_for_runs(
    run_ids: Iterable[str],
    metric: str,
) -> List[Tuple[str, pd.DataFrame]]:
    import wandb

    api = wandb.Api()
    results: List[Tuple[str, pd.DataFrame]] = []
    for rid in run_ids:
        try:
            run = api.run(f"{METTA_WANDB_ENTITY}/{METTA_WANDB_PROJECT}/{rid}")
            # Fetch minimal columns: our metric + agent_step
            df = run.history(keys=[metric, "metric/agent_step"], pandas=True)
            if "metric/agent_step" not in df.columns:
                # Fall back to row index if agent_step missing
                df["metric/agent_step"] = np.arange(len(df))
            df = df[["metric/agent_step", metric]].dropna()
            results.append((rid, df))
        except Exception:
            continue
    return results


def _parse_cfg_seed(run_id: str) -> Tuple[int | None, int | None]:
    # Expect pattern ...cfgXX.seedYY...
    cfg_idx = None
    seed = None
    try:
        parts = run_id.split(".cfg")
        if len(parts) > 1:
            rest = parts[1]
            cfg_idx = int(rest[:2])
        if ".seed" in run_id:
            seed = int(run_id.split(".seed")[1].split(".")[0])
    except Exception:
        pass
    return cfg_idx, seed


def _write_csv_and_plots(
    group: str,
    metric: str,
    out_dir: str | None,
) -> None:
    store = WandbStore(entity=METTA_WANDB_ENTITY, project=METTA_WANDB_PROJECT)
    runs = store.fetch_runs(filters={"group": group})
    run_ids = [r.run_id for r in runs]

    series = _fetch_timeseries_for_runs(run_ids, metric)

    target_dir = Path(out_dir or (Path("./train_dir") / "validation" / group))
    target_dir.mkdir(parents=True, exist_ok=True)

    # Per-seed CSV
    rows: List[dict[str, Any]] = []
    for rid, df in series:
        cfg_idx, seed = _parse_cfg_seed(rid)
        for _, row in df.iterrows():
            rows.append(
                {
                    "run_id": rid,
                    "config_index": cfg_idx,
                    "seed": seed,
                    "agent_step": int(row["metric/agent_step"]),
                    "metric": float(row[metric]),
                }
            )

    if rows:
        per_seed_df = pd.DataFrame(rows)
        per_seed_df.to_csv(target_dir / "timeseries_per_seed.csv", index=False)

        # Plot mean/std/CI per config across seeds
        for cfg_idx, cfg_df in per_seed_df.groupby("config_index"):
            # Align on agent_step by grouping
            grouped = cfg_df.groupby("agent_step")["metric"]
            mean = grouped.mean()
            std = grouped.std().fillna(0.0)
            count = grouped.count().clip(lower=1)
            ci = 1.96 * std / np.sqrt(count)

            plt.figure(figsize=(8, 5))
            x = mean.index.values
            y = mean.values
            plt.plot(x, y, label=f"cfg{cfg_idx:02d}")
            plt.fill_between(
                x, y - std.values, y + std.values, alpha=0.2, label="±1 std"
            )
            plt.fill_between(x, y - ci.values, y + ci.values, alpha=0.2, label="95% CI")
            plt.xlabel("agent_step")
            plt.ylabel(metric)
            plt.title(f"Validation: cfg{cfg_idx:02d} ({group})")
            plt.legend()
            plt.tight_layout()
            plt.savefig(target_dir / f"cfg{cfg_idx:02d}_{metric.replace('/', '_')}.png")
            plt.close()


def validate(
    *,
    sweep_name: str,
    K: int,
    N: int,
    metric: str,
    target_timesteps: int,
    recipe_module: str = "experiments.recipes.arena_basic_easy_shaped",
    train_entrypoint: str = "train",
    eval_entrypoint: str = "evaluate",
    eval_num_episodes: int = 30,
    # Dispatcher + infra knobs (match SweepTool defaults)
    dispatcher_type: str = "skypilot",  # "local" or "skypilot"
    gpus: int = 1,
    nodes: int = 1,
    max_parallel: int = 6,
    monitoring_interval: int = 60,
    local_test: bool = False,
    out_dir: str | None = None,
) -> None:
    """Validate top-K configs from a sweep across N seeds, then plot results.

    Args mirror SweepTool where practical; new runs are created in
    group = {sweep_name}.final_validation.
    """

    store = WandbStore(entity=METTA_WANDB_ENTITY, project=METTA_WANDB_PROJECT)

    plan = ValidationPlan(
        sweep_name=sweep_name,
        num_configs=int(K),
        num_seeds=int(N),
        metric=metric,
        target_timesteps=int(target_timesteps),
        recipe_module=recipe_module,
        train_entrypoint=train_entrypoint,
        eval_entrypoint=eval_entrypoint,
        eval_num_episodes=int(eval_num_episodes),
        dispatcher_type=dispatcher_type.lower(),
        gpus=int(gpus),
        nodes=int(nodes),
        max_parallel=int(max_parallel),
        monitoring_interval=int(monitoring_interval),
        local_test=bool(local_test),
        out_dir=out_dir,
    )

    # Dispatcher
    if plan.local_test or plan.dispatcher_type == "local":
        dispatcher = LocalDispatcher(capture_output=True)
    else:
        dispatcher = SkypilotDispatcher()

    scheduler = TopConfigValidationScheduler(plan, store)

    controller = AdaptiveController(
        experiment_id=f"{sweep_name}.final_validation",
        scheduler=scheduler,
        dispatcher=dispatcher,
        store=store,
        config=AdaptiveConfig(
            max_parallel=plan.max_parallel,
            monitoring_interval=plan.monitoring_interval,
            resume=True,
        ),
    )

    controller.run()

    # Aggregate results and write outputs
    _write_csv_and_plots(
        group=f"{sweep_name}.final_validation", metric=metric, out_dir=plan.out_dir
    )
