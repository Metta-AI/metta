#!/usr/bin/env -S uv run
"""
Evaluate a specified policy across spanning eval missions.

Examples:
  uv run packages/cogames/scripts/evaluate_spanning_policy.py \
    --policy cogames.policy.fast_agents.agents.ThinkyAgentsMultiPolicy --cogs 4

  uv run packages/cogames/scripts/evaluate_spanning_policy.py \
    --policy cogames.policy.fast_agents.agents.ThinkyAgentsMultiPolicy \
    --eval-module cogames.cogs_vs_clips.evals.spanning_evals \
    --experiments unclipping_easy distant_resources_standard --cogs 4 --repeats 3
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import random as pyrandom
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from typing import Iterable, List

from mettagrid.policy.loader import initialize_or_load_policy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator.rollout import Rollout

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_eval_missions(module_path: str):
    module = importlib.import_module(module_path)
    missions = getattr(module, "EVAL_MISSIONS", None)
    if missions is None:
        raise AttributeError(f"Module '{module_path}' does not define EVAL_MISSIONS")
    return missions


def _ensure_vibe_supports_gear(env_cfg) -> None:
    """
    Converter maps IDs from the first N default vibes.
    If assembler protocols require 'gear' but number_of_vibes is too small,
    bump to at least 8 so 'gear' is included.
    """
    try:
        assembler = env_cfg.game.objects.get("assembler")
        uses_gear = False
        if assembler is not None and hasattr(assembler, "protocols"):
            for proto in assembler.protocols:
                if any(v == "gear" for v in getattr(proto, "vibes", [])):
                    uses_gear = True
                    break
        if uses_gear:
            change_vibe = env_cfg.game.actions.change_vibe
            if getattr(change_vibe, "number_of_vibes", 0) < 8:
                change_vibe.number_of_vibes = 8
    except Exception:
        # Best-effort; if anything fails, leave as-is
        pass


@contextmanager
def suppress_output(enabled: bool):
    if not enabled:
        yield
        return
    devnull = open(os.devnull, "w")
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = devnull, devnull
        yield
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        devnull.close()


@contextmanager
def suppress_output_fds(enabled: bool):
    if not enabled:
        yield
        return
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_RDWR)
    try:
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        yield
    finally:
        try:
            os.dup2(saved_stdout, 1)
            os.dup2(saved_stderr, 2)
        finally:
            os.close(saved_stdout)
            os.close(saved_stderr)
            os.close(devnull_fd)


def _policy_label(policy_path: str) -> str:
    try:
        return policy_path.rsplit(".", 1)[-1]
    except Exception:
        return policy_path


def _evaluate_mission_policy(
    mission_name: str,
    eval_module: str,
    policy_path: str,
    cogs: int,
    repeats: int,
    seed: int,
    quiet: bool,
) -> tuple[str, str, float, float, int, str | None]:
    """
    Worker-safe evaluation: reconstruct mission by name, run repeats, compute mean/std/success.
    Returns (mission_name, policy_label, mean, std, successes, error_message_or_None).
    """
    lab = _policy_label(policy_path)
    try:
        missions = load_eval_missions(eval_module)
        mission = next(m for m in missions if m.name == mission_name)
        m = mission.model_copy(deep=True)
        m.num_cogs = cogs

        env = m.make_env()
        _ensure_vibe_supports_gear(env)

        pei = PolicyEnvInterface.from_mg_cfg(env)
        policy = initialize_or_load_policy(pei, policy_path, None)
        agent_policies = [policy.agent_policy(i) for i in range(env.game.num_agents)]

        rewards: List[float] = []
        for r in range(repeats):
            with suppress_output(quiet), suppress_output_fds(quiet):
                ro = Rollout(env, agent_policies, render_mode="none", seed=seed + r, pass_sim_to_policies=True)
                ro.run_until_done()
            avg = sum(ro._sim.episode_rewards) / env.game.num_agents
            rewards.append(avg)
        mean_r = statistics.mean(rewards) if rewards else 0.0
        std_r = statistics.stdev(rewards) if len(rewards) > 1 else 0.0
        succ = sum(1 for x in rewards if x > 0)
        return mission_name, lab, mean_r, std_r, succ, None
    except Exception as e:
        return mission_name, lab, float("nan"), 0.0, 0, str(e)


def evaluate_spanning(
    policy_path: str,
    eval_module: str,
    experiments: Iterable[str] | None,
    cogs: int,
    repeats: int,
    seed: int,
    quiet: bool,
    sample: int | None,
    policy_paths: list[str] | None = None,
    jobs: int = 1,
) -> None:
    # Resolve policy list (backward compatible with --policy)
    paths = list(policy_paths) if policy_paths else [policy_path]
    labels = [_policy_label(p) for p in paths]

    missions = load_eval_missions(eval_module)
    if experiments:
        names = set(experiments)
        missions = [m for m in missions if m.name in names]

    # Optional deterministic sampling
    if sample is not None and 0 < sample < len(missions):
        rng = pyrandom.Random(seed)
        total = len(missions)
        missions = rng.sample(missions, k=sample)
        print(f"Sampling {len(missions)} of {total} missions (seed={seed})")

    if quiet:
        logging.getLogger().setLevel(logging.WARNING)
        logger.setLevel(logging.WARNING)

    # Accumulate per-policy mission results for final summary
    rows_by_policy: dict[str, list[tuple[str, float]]] = {lab: [] for lab in labels}
    # Prepare task list
    mission_names = [m.name for m in missions]
    tasks: list[tuple[str, str]] = [(mn, p) for mn in mission_names for p in paths]

    results: dict[tuple[str, str], tuple[float, float, int, str | None]] = {}
    if jobs and jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = {
                ex.submit(_evaluate_mission_policy, mn, eval_module, pth, cogs, repeats, seed, quiet): (mn, pth)
                for mn, pth in tasks
            }
            for fut in as_completed(futs):
                mn, pth = futs[fut]
                mname, lab, mean_r, std_r, succ, err = fut.result()
                results[(mname, lab)] = (mean_r, std_r, succ, err)
    else:
        for mn, pth in tasks:
            mname, lab, mean_r, std_r, succ, err = _evaluate_mission_policy(
                mn, eval_module, pth, cogs, repeats, seed, quiet
            )
            results[(mname, lab)] = (mean_r, std_r, succ, err)

    # Print per-mission summary lines in mission order
    for mn in mission_names:
        per_policy_summaries: list[str] = []
        for lab, _pth in zip(labels, paths, strict=False):
            mean_r, std_r, succ, err = results.get((mn, lab), (float("nan"), 0.0, 0, "missing"))
            if err is None:
                rows_by_policy[lab].append((mn, mean_r))
                per_policy_summaries.append(f"{lab}: mean={mean_r:.2f} std={std_r:.2f} succ={succ}/{repeats}")
            else:
                rows_by_policy[lab].append((mn, float("nan")))
                per_policy_summaries.append(f"{lab}: ERROR: {err}")
        print(f"{mn:24s} " + " | ".join(per_policy_summaries))

    # Final summary per policy
    print("\n=== SUMMARY ===")
    for lab in labels:
        rows = rows_by_policy[lab]
        vals = [v for _, v in rows if v == v]
        overall_mean = statistics.mean(vals) if vals else float("nan")
        successes = sum(1 for v in vals if v > 0)
        print(f"[{lab:>16s}] missions={len(rows):2d} mean={overall_mean:.2f} successes={successes}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a policy across spanning eval missions")
    parser.add_argument(
        "--policy",
        type=str,
        default="cogames.policy.fast_agents.agents.ThinkyAgentsMultiPolicy",
        help="Fully-qualified policy class path",
    )
    parser.add_argument(
        "--eval-module",
        type=str,
        default="cogames.cogs_vs_clips.evals.spanning_evals",
        help="Module path containing EVAL_MISSIONS (default: spanning_evals)",
    )
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=None,
        help="Subset of mission names to run (default: all in EVAL_MISSIONS)",
    )
    parser.add_argument("--cogs", type=int, default=4, help="Number of agents")
    parser.add_argument("--repeats", type=int, default=3, help="Runs per mission with seed offsets")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--quiet", action="store_true", help="Suppress rollout stdout/stderr and per-mission logs")
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Randomly sample this many missions (deterministic with --seed)",
    )
    parser.add_argument(
        "--policies",
        nargs="*",
        default=None,
        help="Compare multiple policies (list of fully-qualified class paths). Overrides --policy if provided.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Parallel workers for mission×policy evaluations (default: 1).",
    )

    args = parser.parse_args()
    evaluate_spanning(
        policy_path=args.policy,
        eval_module=args.eval_module,
        experiments=args.experiments,
        cogs=args.cogs,
        repeats=args.repeats,
        seed=args.seed,
        quiet=args.quiet,
        sample=args.sample,
        policy_paths=args.policies,
        jobs=max(1, int(args.jobs)),
    )


if __name__ == "__main__":
    main()
