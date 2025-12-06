# Observation Optimization Experiments (1M steps, GPU, serial vecenv, 24 agents, batch 170)

TimerReporter times are seconds (mean of 3 runs per variant unless noted). KSPS from progress logs hover ~66–70 for
optimized runs. Env vars listed apply only to the optimized run; baseline always has `optimized_obs` off.

| Variant (n=3)            | Env toggles (optimized run)                               | Baseline rollout mean (send / inference / td_prep) | Optimized rollout mean (send / inference / td_prep) | Rollout delta |
| ------------------------ | --------------------------------------------------------- | -------------------------------------------------- | --------------------------------------------------- | ------------- |
| Default (all toggles on) | `METTAGRID_OPTIMIZED_OBS=1`                               | 10.20 (6.98 / 2.86 / 0.20)                         | 7.89 (4.69 / 2.83 / 0.18)                           | -2.31 s       |
| No dirty tracking        | `METTAGRID_OPTIMIZED_OBS=1` `METTAGRID_OPT_CACHE_DIRTY=0` | 10.05 (6.89 / 2.82 / 0.17)                         | 9.00 (5.77 / 2.88 / 0.20)                           | -1.05 s       |
| No move swap             | `METTAGRID_OPTIMIZED_OBS=1` `METTAGRID_OPT_MOVE_SWAP=0`   | 10.00 (6.85 / 2.81 / 0.17)                         | 7.81 (4.69 / 2.80 / 0.16)                           | -2.19 s       |
| No memset                | `METTAGRID_OPTIMIZED_OBS=1` `METTAGRID_OPT_MEMSET=0`      | 10.16 (6.93 / 2.88 / 0.19)                         | 7.81 (4.69 / 2.81 / 0.16)                           | -2.35 s       |

Notes

- Dirty tracking + move-swap dominate the send reduction; forcing full refresh erases ~1.2 s of the gain.
- memset vs fill is a small effect (~0.1–0.2 s on send) within noise.
- Prepacked-coordinate toggle remains untested; add runs if we want to isolate it.
- Move-swap bugfix sanity check (single run, default toggles): baseline rollout 10.19 s (send 6.95), optimized rollout
  8.27 s (send 4.94); confirms keeping dst clean preserves the swap benefit.
