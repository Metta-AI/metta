# Observation Optimization Experiments (1M steps, GPU, serial vecenv, 24 agents, batch 170)

TimerReporter times are seconds (mean of `n` runs). KSPS from progress logs hover ~66–70 for optimized runs. Env vars
listed apply only to the optimized run; baseline always has `optimized_obs` off.

| Variant                  | n   | Env toggles (optimized run)                               | Baseline rollout mean (send / inference / td_prep) | Optimized rollout mean (send / inference / td_prep) | Rollout delta |
| ------------------------ | --- | --------------------------------------------------------- | -------------------------------------------------- | --------------------------------------------------- | ------------- |
| Default (all toggles on) | 6   | `METTAGRID_OPTIMIZED_OBS=1`                               | 10.08 (6.90 / 2.84 / 0.18)                         | 7.87 (4.67 / 2.85 / 0.18)                           | -2.21 s       |
| No dirty tracking        | 6   | `METTAGRID_OPTIMIZED_OBS=1` `METTAGRID_OPT_CACHE_DIRTY=0` | 10.19 (6.99 / 2.85 / 0.18)                         | 8.96 (5.74 / 2.86 / 0.20)                           | -1.23 s       |
| No move swap             | 6   | `METTAGRID_OPTIMIZED_OBS=1` `METTAGRID_OPT_MOVE_SWAP=0`   | 10.04 (6.88 / 2.82 / 0.17)                         | 7.85 (4.71 / 2.82 / 0.16)                           | -2.19 s       |

Notes

- Means are averaged over 6 runs (3 prior + 3 new runs in `outputs/optobs_runs/optobs_n6_20251205_234038.log`).
- Dirty tracking + move-swap dominate the send reduction; forcing full refresh erases ~1.2 s of the gain.
- memset vs fill was noise-level; the toggle was removed (no observed benefit).
- Prepacked-coordinate toggle remains untested; add runs if we want to isolate it.
- Move-swap bugfix sanity check (single run, default toggles): baseline rollout 10.19 s (send 6.95), optimized rollout
  8.27 s (send 4.94); confirms keeping dst clean preserves the swap benefit.
