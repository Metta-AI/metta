# Machina Eval Experiments

This suite defines 10 single-map experiments derived from the `machina_eval_template` base. Each experiment scatters a small number of stations outside the central base and adjusts station efficiencies, per-station max uses ("max outputs"), and the agent's energy regeneration to emphasize a distinct strategy.

Legend: `+` charger, `C` carbon, `O` oxygen, `G` germanium, `S` silicon, `&` assembler, `=` heart chest, `@` agent spawn, `#` wall, `.` empty. The central base (containing `&`, `=`, `@`, nearby objects) is left unchanged.

For each experiment below:
- Counts denote objects scattered OUTSIDE the base.
- Efficiencies affect yields/cooldowns.
- Max uses = per-station maximum activations (0 means unlimited for chargers; others respect the number).

## 1) energy_starved (machina_eval_exp01.map)
- Objects: `+ x2, C x3, O x1, G x2, S x3`
- Efficiencies: `charger=80, carbon=100, oxygen=100, germanium=1, silicon=100`
- Max uses: `carbon=100, oxygen=30, germanium=5, silicon=100, charger=0`
- Energy regen: `0`
- Optimal strategy: Tight energy economy. Route between chargers and silicon to fund movement. Batch resources to avoid extra trips.

## 2) oxygen_bottleneck (machina_eval_exp02.map)
- Objects: `+ x2, C x2, O x1, G x3, S x2`
- Efficiencies: `oxygen=50 (slower), others=100`
- Max uses: `oxygen=20, carbon=100, germanium=10, silicon=100, charger=0`
- Energy regen: `1`
- Optimal strategy: Oxygen paces assembly. Time oxygen taps; gather others while oxygen cools down.

## 3) germanium_rush (machina_eval_exp03.map)
- Objects: `+ x2, C x2, O x2, G x6, S x2`
- Efficiencies: `all=100`
- Max uses: `germanium=10, carbon=100, oxygen=50, silicon=100, charger=0`
- Energy regen: `1`
- Optimal strategy: Sprint to germanium early; avoid detours; bring exact bundles to assembler.

## 4) silicon_workbench (machina_eval_exp04.map)
- Objects: `+ x4, C x2, O x2, G x2, S x8`
- Efficiencies: `silicon=150, others=100`
- Max uses: `silicon=200, oxygen=50, carbon=100, germanium=10, charger=0`
- Energy regen: `1`
- Optimal strategy: Turn energy into silicon; ensure charger coverage; silicon-first pipeline.

## 5) carbon_desert (machina_eval_exp05.map)
- Objects: `+ x3, C x1, O x3, G x3, S x3`
- Efficiencies: `all=100`
- Max uses: `carbon=30, oxygen=50, germanium=10, silicon=100, charger=0`
- Energy regen: `1`
- Optimal strategy: Plan around the lone carbon; stage other resources to minimize carbon trips.

## 6) single_use_world (machina_eval_exp06.map)
- Objects: `+ x4, C x4, O x3, G x4, S x4`
- Efficiencies: `all=100`
- Max uses: `charger=1, carbon=1, oxygen=1, germanium=1, silicon=1`
- Energy regen: `1`
- Optimal strategy: One pass only. Exact ordering; no retries.

## 7) slow_oxygen (machina_eval_exp07.map)
- Objects: `+ x4, C x3, O x3, G x3, S x4`
- Efficiencies: `oxygen=25, others=100`
- Max uses: `oxygen=100, carbon=100, germanium=10, silicon=100, charger=0`
- Energy regen: `2`
- Optimal strategy: Interleave partial-usage oxygen taps with other trips.

## 8) high_regen_sprint (machina_eval_exp08.map)
- Objects: `+ x1, C x2, O x2, G x2, S x2`
- Efficiencies: `all=100`
- Max uses: `carbon=100, oxygen=50, germanium=10, silicon=100, charger=0`
- Energy regen: `3`
- Optimal strategy: High regen reduces charger dependence; take efficient long routes.

## 9) sparse_balanced (machina_eval_exp09.map)
- Objects: `+ x2, C x2, O x2, G x2, S x2`
- Efficiencies: `all=100`
- Max uses: `carbon=50, oxygen=50, germanium=10, silicon=50, charger=0`
- Energy regen: `1`
- Optimal strategy: Balanced gathering with minimal backtracking.

## 10) germanium_clutch (machina_eval_exp10.map)
- Objects: `+ x3, C x4, O x4, G x1, S x4`
- Efficiencies: `all=100`
- Max uses: `germanium=2, carbon=100, oxygen=50, silicon=100, charger=0`
- Energy regen: `1`
- Optimal strategy: Beeline to the single germanium line; align other trips around that constraint.
