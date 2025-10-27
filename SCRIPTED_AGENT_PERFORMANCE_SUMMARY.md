# Scripted Agent Performance Summary

## Overall Performance

- **Exploration Experiments**: 32/36 (88.9%) ✅
- **Eval Missions**: 28/40 (70.0%) ✅
- **Combined**: 60/76 (78.9%)

---

## Exploration Experiments - Per-Experiment Results


### EXP1
**Play:** `cogames play -m exp1.baseline -p scripted`
  ✅ baseline       : Reward=2.0, Hearts=2, Steps=1000
  ✅ conservative   : Reward=2.0, Hearts=2, Steps=1000
  ✅ aggressive     : Reward=2.0, Hearts=2, Steps=1000
  ✅ silicon_focused: Reward=2.0, Hearts=2, Steps=1000

### EXP10
**Play:** `cogames play -m exp10.complex_mixed -p scripted`
  ❌ baseline       : Reward=0.0, Hearts=0, Steps=2500
  ❌ conservative   : Reward=0.0, Hearts=0, Steps=2500
  ❌ aggressive     : Reward=0.0, Hearts=0, Steps=2500
  ❌ silicon_focused: Reward=0.0, Hearts=0, Steps=2500

### EXP2
**Play:** `cogames play -m exp2.oxygen_abundance -p scripted`
  ✅ baseline       : Reward=1.0, Hearts=1, Steps=2500
  ✅ conservative   : Reward=1.0, Hearts=1, Steps=2500
  ✅ aggressive     : Reward=1.0, Hearts=1, Steps=2500
  ✅ silicon_focused: Reward=1.0, Hearts=1, Steps=2500

### EXP4
**Play:** `cogames play -m exp4.fast_depletion -p scripted`
  ✅ baseline       : Reward=1.0, Hearts=1, Steps=2500
  ✅ conservative   : Reward=1.0, Hearts=1, Steps=2500
  ✅ aggressive     : Reward=1.0, Hearts=1, Steps=2500
  ✅ silicon_focused: Reward=1.0, Hearts=1, Steps=2500

### EXP5
**Play:** `cogames play -m exp5.energy_abundance -p scripted`
  ✅ baseline       : Reward=2.0, Hearts=2, Steps=2500
  ✅ conservative   : Reward=2.0, Hearts=2, Steps=2500
  ✅ aggressive     : Reward=2.0, Hearts=2, Steps=2500
  ✅ silicon_focused: Reward=2.0, Hearts=2, Steps=2500

### EXP6
**Play:** `cogames play -m exp6.energy_scarcity -p scripted`
  ✅ baseline       : Reward=1.0, Hearts=1, Steps=3000
  ✅ conservative   : Reward=1.0, Hearts=1, Steps=3000
  ✅ aggressive     : Reward=1.0, Hearts=1, Steps=3000
  ✅ silicon_focused: Reward=1.0, Hearts=1, Steps=3000

### EXP7
**Play:** `cogames play -m exp7.high_efficiency -p scripted`
  ✅ baseline       : Reward=1.0, Hearts=1, Steps=3000
  ✅ conservative   : Reward=1.0, Hearts=1, Steps=3000
  ✅ aggressive     : Reward=1.0, Hearts=1, Steps=3000
  ✅ silicon_focused: Reward=1.0, Hearts=1, Steps=3000

### EXP8
**Play:** `cogames play -m exp8.zoned_resources -p scripted`
  ✅ baseline       : Reward=1.0, Hearts=1, Steps=3000
  ✅ conservative   : Reward=1.0, Hearts=1, Steps=3000
  ✅ aggressive     : Reward=1.0, Hearts=1, Steps=3000
  ✅ silicon_focused: Reward=1.0, Hearts=1, Steps=3000

### EXP9
**Play:** `cogames play -m exp9.resource_abundance -p scripted`
  ✅ baseline       : Reward=1.0, Hearts=1, Steps=3000
  ✅ conservative   : Reward=1.0, Hearts=1, Steps=3000
  ✅ aggressive     : Reward=1.0, Hearts=1, Steps=3000
  ✅ silicon_focused: Reward=1.0, Hearts=1, Steps=3000

### Configuration Performance (Exploration)
  aggressive     : 8/9 success, avg_reward=1.11, avg_hearts=1.11
  baseline       : 8/9 success, avg_reward=1.11, avg_hearts=1.11
  conservative   : 8/9 success, avg_reward=1.11, avg_hearts=1.11
  silicon_focused: 8/9 success, avg_reward=1.11, avg_hearts=1.11

---

## Eval Missions - Per-Mission Results


### CARBON_DESERT
**Play:** `cogames play -m machina_eval.carbon_desert -p scripted`
  ✅ baseline       : Reward=2.0, Hearts=2, Steps=1000
  ✅ conservative   : Reward=2.0, Hearts=2, Steps=1000
  ✅ aggressive     : Reward=2.0, Hearts=2, Steps=1000
  ✅ silicon_focused: Reward=2.0, Hearts=2, Steps=1000

### ENERGY_STARVED
**Play:** `cogames play -m machina_eval.energy_starved -p scripted`
  ✅ baseline       : Reward=2.0, Hearts=2, Steps=1000
  ✅ conservative   : Reward=2.0, Hearts=2, Steps=1000
  ✅ aggressive     : Reward=2.0, Hearts=2, Steps=1000
  ✅ silicon_focused: Reward=2.0, Hearts=2, Steps=1000

### GERMANIUM_CLUTCH
**Play:** `cogames play -m machina_eval.germanium_clutch -p scripted`
  ❌ baseline       : Reward=0.0, Hearts=0, Steps=1000
  ❌ conservative   : Reward=0.0, Hearts=0, Steps=1000
  ❌ aggressive     : Reward=0.0, Hearts=0, Steps=1000
  ❌ silicon_focused: Reward=0.0, Hearts=0, Steps=1000

### GERMANIUM_RUSH
**Play:** `cogames play -m machina_eval.germanium_rush -p scripted`
  ✅ baseline       : Reward=2.0, Hearts=2, Steps=1000
  ✅ conservative   : Reward=2.0, Hearts=2, Steps=1000
  ✅ aggressive     : Reward=2.0, Hearts=2, Steps=1000
  ✅ silicon_focused: Reward=2.0, Hearts=2, Steps=1000

### HIGH_REGEN_SPRINT
**Play:** `cogames play -m machina_eval.high_regen_sprint -p scripted`
  ✅ baseline       : Reward=2.0, Hearts=2, Steps=1000
  ✅ conservative   : Reward=2.0, Hearts=2, Steps=1000
  ✅ aggressive     : Reward=2.0, Hearts=2, Steps=1000
  ✅ silicon_focused: Reward=2.0, Hearts=2, Steps=1000

### OXYGEN_BOTTLENECK
**Play:** `cogames play -m machina_eval.oxygen_bottleneck -p scripted`
  ✅ baseline       : Reward=2.0, Hearts=2, Steps=1000
  ✅ conservative   : Reward=2.0, Hearts=2, Steps=1000
  ✅ aggressive     : Reward=2.0, Hearts=2, Steps=1000
  ✅ silicon_focused: Reward=2.0, Hearts=2, Steps=1000

### SILICON_WORKBENCH
**Play:** `cogames play -m machina_eval.silicon_workbench -p scripted`
  ❌ baseline       : Reward=0.0, Hearts=0, Steps=1000
  ❌ conservative   : Reward=0.0, Hearts=0, Steps=1000
  ❌ aggressive     : Reward=0.0, Hearts=0, Steps=1000
  ❌ silicon_focused: Reward=0.0, Hearts=0, Steps=1000

### SINGLE_USE_WORLD
**Play:** `cogames play -m machina_eval.single_use_world -p scripted`
  ❌ baseline       : Reward=0.0, Hearts=0, Steps=1000
  ❌ conservative   : Reward=0.0, Hearts=0, Steps=1000
  ❌ aggressive     : Reward=0.0, Hearts=0, Steps=1000
  ❌ silicon_focused: Reward=0.0, Hearts=0, Steps=1000

### SLOW_OXYGEN
**Play:** `cogames play -m machina_eval.slow_oxygen -p scripted`
  ✅ baseline       : Reward=2.0, Hearts=2, Steps=1000
  ✅ conservative   : Reward=2.0, Hearts=2, Steps=1000
  ✅ aggressive     : Reward=2.0, Hearts=2, Steps=1000
  ✅ silicon_focused: Reward=2.0, Hearts=2, Steps=1000

### SPARSE_BALANCED
**Play:** `cogames play -m machina_eval.sparse_balanced -p scripted`
  ✅ baseline       : Reward=2.0, Hearts=2, Steps=1000
  ✅ conservative   : Reward=2.0, Hearts=2, Steps=1000
  ✅ aggressive     : Reward=2.0, Hearts=2, Steps=1000
  ✅ silicon_focused: Reward=2.0, Hearts=2, Steps=1000

### Configuration Performance (Eval Missions)
  baseline       : 7/10 success, avg_reward=1.40, avg_hearts=1.40
  conservative   : 7/10 success, avg_reward=1.40, avg_hearts=1.40
  aggressive     : 7/10 success, avg_reward=1.40, avg_hearts=1.40
  silicon_focused: 7/10 success, avg_reward=1.40, avg_hearts=1.40

---

## Key Insights

### Successful Patterns
- All configurations perform similarly (baseline, conservative, aggressive, silicon_focused)
- Agent excels at medium-to-large maps (40x40 to 90x100)
- Robust handling of energy scarcity, depleted extractors, and complex layouts
- Dynamic resource collection and stuck detection working well

### Failing Experiments
**Exploration:**
- EXP10 (90x90, mixed efficiency): 0/4 - Oxygen marked as unobtainable

**Eval Missions:**
- germanium_clutch: 0/4
- silicon_workbench: 0/4
- single_use_world: 0/4

### Major Fixes Applied
1. ✅ Critical environment bug: `change_glyph` consuming 100 energy → fixed
2. ✅ Hearts counter not incrementing → fixed
3. ✅ Navigator refactor: Clean pathfinding to stations/extractors
4. ✅ Chest configuration: Accept deposits from any direction (N/S/E/W)
5. ✅ Resource-based stuck detection and blacklisting
6. ✅ Opportunistic resource collection
7. ✅ Dynamic recharge thresholds based on map size

---

## Hyperparameter Configurations

### Default (Baseline) - **RECOMMENDED** ✅
- `energy_buffer: 20` - Safety margin for energy calculations
- `min_energy_for_silicon: 70` - Min energy before silicon harvesting
- `charger_search_threshold: 40` - Search for charger below this
- `prefer_nearby: True` - Prefer closer extractors

**Note**: All 4 tested configs (baseline, conservative, aggressive, silicon_focused) perform identically. The **baseline config** is used by default when running cogames and is recommended.


---

## How to Run Evaluations

**Full exploration experiments evaluation (9 experiments × 4 configs = 36 tests):**
```bash
python packages/cogames/scripts/evaluate_outpost_all_experiments.py
```

**Full eval missions evaluation (10 missions × 4 configs = 40 tests):**
```bash
python packages/cogames/scripts/evaluate_eval_missions.py
```

**Output files:**
- `phase1_evaluation_results.json` - Exploration experiments
- `eval_missions_results.json` - Eval missions
