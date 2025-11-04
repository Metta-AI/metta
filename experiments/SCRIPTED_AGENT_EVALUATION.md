 # Scripted Agent Evaluation Report

**Date**: November 1, 2025
**Evaluation Suites Executed**:
- Full scripted-agent sweep via `evaluate_scripted_agent.py` with agent counts limited to 2, 4, and 8 (aggregates exclude clipped runs; see note below)
- Runtime sampling on `CollectResourcesBase` (medium difficulty) for 1000-step episodes at 2/4/8 agents

---

## Executive Summary

- ✅ **Full sweep completed (2/4/8 agents only)**: 2,730 unclipped configs ran with an overall success rate of **30.2 %**.
- ✅ **Moderate-team scenarios remain the strongest**: 4-agent runs edge out 2-agent runs (**33.6 %** vs **31.7 %**) with manageable runtime cost.
- ⚠️ **High-population missions still struggle**: 8-agent configurations clear only **23.9 %** of missions and frequently stall on oxygen scarcity or charger contention.
- ⚠️ **Single-use and clip-heavy missions remain unsolved**: `ClipOxygen` and `SingleUseSwarm` produced zero successes across all tested presets.
- ⏱️ **Runtime scales roughly linearly with team size**: 1000-step episodes average ~1.35 s (2 agents) up to ~5.12 s (8 agents) on the current workstation.

---

## Full Evaluation Sweep (2,730 unclipped configs)

Agent counts were restricted to {2, 4, 8}. The sweep covered every evaluation mission, difficulty variant, clip profile, clip rate, and hyperparameter preset, but all clipped runs (`clip_rate = 0.25`) produced zero reward and are omitted from the aggregated tables below.

| Cogs | Success / Total | Success % | Avg Reward |
|------|-----------------|-----------|------------|
| 2    | 311 / 980       | 31.7 %    | 1.65       |
| 4    | 329 / 980       | 33.6 %    | 1.31       |
| 8    | 184 / 770       | 23.9 %    | 0.57       |

| Difficulty Tier (legacy name) | Success / Total | Success % | Avg Reward |
|-------------------------------|-----------------|-----------|------------|
| story_mode (easy)             | 135 / 390       | 34.6 %    | 1.41       |
| standard (medium)             | 135 / 390       | 34.6 %    | 1.41       |
| hard                          | 135 / 390       | 34.6 %    | 1.41       |
| brutal (extreme)              | 135 / 390       | 34.6 %    | 1.41       |
| energy_crisis                 | 135 / 390       | 34.6 %    | 1.41       |
| speed_run                     | 135 / 390       | 34.6 %    | 1.41       |
| single_use                    | 14 / 390        | 3.6 %     | 0.11       |

*These per-tier results reflect the pre-overhaul run with legacy difficulty settings; updated numbers will be generated after the new tiers are re-evaluated.*

**New canonical tiers:** `story_mode` (former easy), `standard` (former medium), `hard`, and `brutal` (former extreme), plus specialised tiers `single_use`, `speed_run`, and `energy_crisis`.

| Hyperparameter Preset | Success / Total | Success % |
|-----------------------|-----------------|-----------|
| explorer_long         | 171 / 546       | 31.3 %    |
| balanced              | 165 / 546       | 30.2 %    |
| efficiency_heavy      | 165 / 546       | 30.2 %    |
| sequential_baseline   | 165 / 546       | 30.2 %    |
| greedy_conservative   | 158 / 546       | 28.9 %    |

### Mission Highlights

- **Highest success (≥10 runs)**
  - `OxygenBottleneck`: 98 / 210 (46.7 %)
  - `GoTogether`: 90 / 210 (42.9 %)
  - `CollectResourcesClassic`: 90 / 210 (42.9 %)
  - `ExtractorHub30` & `CollectResourcesBase`: 84 / 210 each (40.0 %)

- **Lowest success (≥10 runs)**
  - `ClipOxygen`: 0 / 210 (0.0 %)
  - `SingleUseSwarm`: 0 / 210 (0.0 %)
  - `ExtractorHub100`: 24 / 140 (17.1 %)
  - `ExtractorHub80`: 30 / 140 (21.4 %)

### Per-Mission Breakdown (clip_rate = 0.0 only)

| Mission                     | Success / Total | Success % | Avg Reward | Avg Hearts |
|-----------------------------|-----------------|-----------|------------|------------|
| OxygenBottleneck            | 98 / 210        | 46.7 %    | 1.43       | 0.50       |
| ExtractorHub50              | 60 / 140        | 42.9 %    | 1.97       | 0.77       |
| CollectResourcesClassic     | 90 / 210        | 42.9 %    | 2.63       | 1.26       |
| GoTogether                  | 90 / 210        | 42.9 %    | 2.03       | 1.09       |
| ExtractorHub30              | 84 / 210        | 40.0 %    | 1.54       | 0.43       |
| CollectResourcesBase        | 84 / 210        | 40.0 %    | 2.54       | 1.00       |
| ExtractorHub70              | 72 / 210        | 34.3 %    | 1.00       | 0.40       |
| CollectResourcesSpread      | 72 / 210        | 34.3 %    | 1.23       | 0.57       |
| DivideAndConquer            | 66 / 210        | 31.4 %    | 0.80       | 0.40       |
| CollectFar                  | 54 / 210        | 25.7 %    | 0.49       | 0.17       |
| ExtractorHub80              | 30 / 140        | 21.4 %    | 0.94       | 0.13       |
| ExtractorHub100             | 24 / 140        | 17.1 %    | 0.43       | 0.00       |
| SingleUseSwarm              | 0 / 210         | 0.0 %     | 0.00       | 0.00       |
| ClipOxygen                  | 0 / 210         | 0.0 %     | 0.00       | 0.00       |

Clipped scenarios (`clip_rate = 0.25`) were removed from these aggregates after repeated zero-success outcomes; see Observations for next steps.

## Evaluation Missions & Quick Play

Launch any evaluation environment locally with:

```bash
uv run python -m cogames.cli.mission play --mission <site.mission> --cogs <N>
```

| Mission                | Description                                                                    | Map                                   | Quick Play Example |
|------------------------|--------------------------------------------------------------------------------|---------------------------------------|--------------------|
| `evals.EnergyStarved`  | Low regen; requires careful charging and routing.                             | `evals/eval_energy_starved.map`       | `... --mission evals.energy_starved --cogs 4` |
| `evals.OxygenBottleneck` | Oxygen paces assembly; batch other resources.                               | `evals/eval_oxygen_bottleneck.map`    | `... --mission evals.oxygen_bottleneck --cogs 4` |
| `evals.ExtractorHub30` | Small 30×30 extractor hub.                                                     | `evals/extractor_hub_30x30.map`       | `... --mission evals.extractor_hub_30 --cogs 4` |
| `evals.ExtractorHub50` | Medium 50×50 extractor hub.                                                    | `evals/extractor_hub_50x50.map`       | `... --mission evals.extractor_hub_50 --cogs 4` |
| `evals.ExtractorHub70` | Large 70×70 extractor hub.                                                     | `evals/extractor_hub_70x70.map`       | `... --mission evals.extractor_hub_70 --cogs 4` |
| `evals.ExtractorHub80` | Large 80×80 extractor hub.                                                     | `evals/extractor_hub_80x80.map`       | `... --mission evals.extractor_hub_80 --cogs 4` |
| `evals.ExtractorHub100`| Extra large 100×100 extractor hub.                                             | `evals/extractor_hub_100x100.map`     | `... --mission evals.extractor_hub_100 --cogs 4` |
| `evals.CollectResourcesBase` | Collect near-base resources; single carrier deposits.                   | `evals/eval_collect_resources_easy.map` | `... --mission evals.collect_resources_base --cogs 4` |
| `evals.CollectResourcesClassic` | Classic layout with balanced routing near base.                     | `evals/eval_collect_resources.map`    | `... --mission evals.collect_resources_classic --cogs 4` |
| `evals.CollectResourcesSpread` | Scattered resources; assemble at hub.                                | `evals/eval_collect_resources_medium.map` | `... --mission evals.collect_resources_spread --cogs 4` |
| `evals.CollectFar`     | Widely separated resources; heavy routing coordination.                       | `evals/eval_collect_resources_hard.map` | `... --mission evals.collect_far --cogs 4` |
| `evals.DivideAndConquer` | Region-partitioned resources; specialization required.                     | `evals/eval_divide_and_conquer.map`   | `... --mission evals.divide_and_conquer --cogs 4` |
| `evals.GoTogether`     | Collective glyphing and synchronized travel.                                  | `evals/eval_balanced_spread.map`      | `... --mission evals.go_together --cogs 4` |
| `evals.SingleUseSwarm` | Every extractor single-use; team must fan out and reconverge.                 | `evals/eval_single_use_world.map`     | `... --mission evals.single_use_swarm --cogs 4` |
| `evals.ClipOxygen`     | Oxygen extractor starts clipped; requires crafted decoder to unlock.          | `evals/eval_clip_oxygen.map`          | `... --mission evals.clip_oxygen --cogs 4` |

*(Replace the ellipsis with `uv run python -m cogames.cli.mission play`.)*

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
2. **Difficulty tiers overhauled**: the legacy “easy/medium/hard/extreme” presets converged to identical behaviour, so we replaced them with `story_mode`, `standard`, `hard`, and `brutal` (plus specialised tiers). Re-run the full sweep to validate the new overrides once agent scaling fixes land.
3. **Clipped & single-use missions require bespoke strategy**: zero successes on every `clip_rate = 0.25` scenario and on `SingleUseSwarm` underscore the need for deterministic unclipping and single-use routing policies.
4. **8-agent coordination**: charger congestion and shared pathfinding slowdowns continue to suppress success rates and runtime efficiency; charger load balancing and decentralized exploration should be prioritized.
5. **Runtime headroom**: 8-agent runs are ~4× slower than 2-agent runs; any future evaluation automation should parallelize across agent counts to keep turnaround reasonable.
6. **Clipped runs deferred**: we deliberately excluded `clip_rate = 0.25` missions from the metrics above because they still yield 0 % success; revisit once unclipping/gear crafting logic is hardened.

---

## Reproduction

```bash
# Full evaluation restricted to 2/4/8 agents (saves results to JSON)
LOG_LEVEL=WARNING uv run python -u packages/cogames/scripts/evaluate_scripted_agent.py --output evaluation_results.json full --cogs 2 4 8

```
