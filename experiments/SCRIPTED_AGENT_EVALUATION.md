# Scripted Agent Evaluation Report

**Date**: November 1, 2025
**Agent Version**: Visited-aware exploration + adaptive energy tracking
**Evaluation Suites Executed**:
- Full scripted-agent sweep via `evaluate_scripted_agent.py` with agent counts limited to 2, 4, and 8
- Runtime sampling on `CollectResourcesBase` (medium difficulty) for 1000-step episodes at 2/4/8 agents

---

## Executive Summary

- ✅ **Full sweep completed (2/4/8 agents only)**: 4,095 evaluation cases executed with an overall success rate of **20.1 %**.
- ✅ **Moderate-team scenarios remain the strongest**: 4-agent runs edge out 2-agent runs (22.4 % vs 21.2 % success) with comparable runtime cost.
- ⚠️ **High-population missions still struggle**: 8-agent configurations succeed only 15.9 % of the time and frequently stall on oxygen scarcity or charger contention.
- ⚠️ **Single-use and clip-heavy missions remain unsolved**: `ClipOxygen` and `SingleUseSwarm` produced zero successes across all tested presets.
- ⏱️ **Runtime scales roughly linearly with team size**: 1000-step episodes average ~1.35 s (2 agents) up to ~5.12 s (8 agents) on the current workstation.

---

## Full Evaluation Sweep (4,095 configs)

Agent counts were restricted to {2, 4, 8}; all experiments, difficulties, clip settings, and hyperparameter presets were otherwise left at their defaults.

| Cogs | Success / Total | Success % | Avg Reward |
|------|-----------------|-----------|------------|
| 2    | 311 / 1,470     | 21.2 %    | 1.10       |
| 4    | 329 / 1,470     | 22.4 %    | 0.88       |
| 8    | 184 / 1,155     | 15.9 %    | 0.38       |

| Difficulty    | Success / Total | Success % | Avg Reward |
|---------------|-----------------|-----------|------------|
| easy          | 135 / 585       | 23.1 %    | 0.94       |
| medium        | 135 / 585       | 23.1 %    | 0.94       |
| hard          | 135 / 585       | 23.1 %    | 0.94       |
| extreme       | 135 / 585       | 23.1 %    | 0.94       |
| energy_crisis | 135 / 585       | 23.1 %    | 0.94       |
| speed_run     | 135 / 585       | 23.1 %    | 0.94       |
| single_use    | 14 / 585        | 2.4 %     | 0.07       |

| Hyperparameter Preset | Success / Total | Success % |
|-----------------------|-----------------|-----------|
| explorer_long         | 171 / 819       | 20.9 %    |
| balanced              | 165 / 819       | 20.1 %    |
| efficiency_heavy      | 165 / 819       | 20.1 %    |
| sequential_baseline   | 165 / 819       | 20.1 %    |
| greedy_conservative   | 158 / 819       | 19.3 %    |

### Mission Highlights

- **Highest success (≥10 runs)**
  - `OxygenBottleneck`: 98 / 315 (31.1 %)
  - `GoTogether`: 90 / 315 (28.6 %)
  - `CollectResourcesClassic`: 90 / 315 (28.6 %)
  - `ExtractorHub30` & `CollectResourcesBase`: 84 / 315 each (26.7 %)

- **Lowest success (≥10 runs)**
  - `ClipOxygen`: 0 / 315 (0.0 %)
  - `SingleUseSwarm`: 0 / 315 (0.0 %)
  - `ExtractorHub100`: 24 / 210 (11.4 %)
  - `ExtractorHub80`: 30 / 210 (14.3 %)

---

## Runtime Benchmarks (1000-step episodes)

Measured on `CollectResourcesBase` with medium difficulty and the `balanced` preset; each entry averages 3 trials.

| Cogs | Mean Time (s) | σ (s)  | Trial Durations (s)            |
|------|---------------|--------|--------------------------------|
| 2    | 1.35          | 0.006  | 1.36, 1.35, 1.35               |
| 4    | 2.62          | 0.045  | 2.67, 2.63, 2.56               |
| 8    | 5.12          | 0.032  | 5.12, 5.15, 5.08               |

These timings align with a roughly linear cost increase per additional agent pair. End-to-end full-suite runtime (4,095 configs) was just under 4 hours on the evaluation machine.

---

## Observations & Next Steps

1. **Oxygen scarcity dominates failures**: many missions log “No available oxygen extractors,” indicating extractor detection and rotation still need targeted fixes (especially in clip scenarios).
2. **Single-use missions require bespoke strategy**: zero successes on `SingleUseSwarm`/`ClipOxygen` highlight the need for deterministic unclipping and single-use routing policies.
3. **8-agent coordination**: charger congestion and shared pathfinding slowdowns continue to suppress success rates and runtime efficiency; charger load balancing and decentralized exploration should be prioritized.
4. **Runtime headroom**: 8-agent runs are ~4× slower than 2-agent runs; any future evaluation automation should parallelize across agent counts to keep turnaround reasonable.

---

## Reproduction

```bash
# Full evaluation restricted to 2/4/8 agents (saves results to JSON)
LOG_LEVEL=WARNING uv run python -u packages/cogames/scripts/evaluate_scripted_agent.py --output evaluation_full_cogs248.json full --cogs 2 4 8

# Runtime sampling (replicates the table above)
uv run python - <<'PY'
import json, statistics, time, numpy as np
from cogames.cogs_vs_clips.evals import DIFFICULTY_LEVELS, apply_difficulty
from cogames.cogs_vs_clips.evals.eval_missions import CollectResourcesBase
from cogames.policy.scripted_agent import HYPERPARAMETER_PRESETS, ScriptedAgentPolicy
from mettagrid import MettaGridEnv, dtype_actions

TRIALS, STEPS = 3, 1000
preset = HYPERPARAMETER_PRESETS["balanced"]
mission_cls = CollectResourcesBase
rows = []
for cogs in (2, 4, 8):
    timings = []
    for _ in range(TRIALS):
        mission = mission_cls()
        apply_difficulty(mission, DIFFICULTY_LEVELS["medium"])
        mission = mission.instantiate(mission.site.map_builder if mission.site else None, num_cogs=cogs)
        cfg = mission.make_env()
        cfg.game.max_steps = STEPS
        env = MettaGridEnv(cfg)
        policy = ScriptedAgentPolicy(env, hyperparams=preset)
        obs, info = env.reset()
        policy.reset(obs, info)
        agents = [policy.agent_policy(i) for i in range(env.num_agents)]
        start = time.perf_counter()
        for _ in range(STEPS):
            actions = np.zeros(env.num_agents, dtype=dtype_actions)
            for i in range(env.num_agents):
                actions[i] = int(agents[i].step(obs[i]))
            obs, rewards, done, truncated, _ = env.step(actions)
            if all(done) or all(truncated):
                break
        timings.append(time.perf_counter() - start)
        env.close()
    rows.append({
        "cogs": cogs,
        "mean_sec": statistics.mean(timings),
        "stdev_sec": statistics.pstdev(timings),
        "samples": timings,
    })
print(json.dumps(rows, indent=2))
PY
```

Artifacts: `evaluation_full_cogs248.json` (4,095-row aggregate) plus the runtime JSON emitted by the sampling script.
