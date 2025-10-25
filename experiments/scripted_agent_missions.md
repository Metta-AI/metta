# Scripted Agent Exploration Missions

## Overview

10 experiments designed to test different exploration strategies for a single agent. Each experiment uses existing `extractor_hub` maps with different parameter settings (efficiency, max_uses, energy_regen).

**All maps include**: 1 assembler (&), 1 chest (=), 4 agent spawn points (@)

---

## Experiment 1: Baseline

### Environment
- **Map**: `extractor_hub_30x30.map` (30×30)
- **Resources**: 2 Carbon, 2 Oxygen, 3 Germanium, 2 Silicon, 5 Chargers
- **Total**: 9 extractors (smallest map)

### Parameters
- **Efficiency**: 100% (all extractors)
- **Max Uses**: 1000 (all extractors)
- **Energy Regen**: 1/turn (standard)

### Game Mechanics Context
- **Heart recipe**: 20C + 20O + 5G + 50S + 20 energy
- **Resource outputs** (100% efficiency):
  - Carbon: 4/use → need 5 uses (50 turns with 10 turn cooldown)
  - Oxygen: 20/use → need 1 use (**100 turn cooldown = bottleneck**)
  - Germanium: 1/use → need 5 uses (no cooldown)
  - Silicon: 25/use → need 2 uses (50 energy cost)
- **Energy**: 100 capacity, 2 per move, 1/turn regen
- **Heart cycle time**: ~100+ turns (limited by oxygen cooldown)

### Optimal Strategy
1. **Initial production** (base resources):
   - Harvest from base: 5× carbon (50 turns), 1× oxygen (100 turn cooldown), 5× germanium (instant), 2× silicon (50 energy)
   - Charge back to 100 energy
   - Assemble heart (20 energy), deposit in chest

2. **During oxygen cooldown** (100 turns of waiting):
   - Explore to discover outside extractors
   - Map out oxygen locations (2 total: 1 in base + 1 outside)
   - Map out other resources
   - Stay within ~20 tiles of base (40 energy round trip)

3. **Production cycle** with outside resources:
   - Use whichever oxygen is available (rotate between base and outside)
   - Gather carbon from nearest available (2 sources)
   - Gather germanium from base (3 sources, no cooldown so location doesn't matter much)
   - Gather silicon carefully (energy cost)
   - Return to base to assemble and deposit

4. **Energy management**:
   - Always charge to 100 before silicon runs (need 70 total: 50 for silicon + 20 for assembly)
   - 5 chargers available (2 in base, 3 outside)

### Expected Performance
- **Hearts**: 5-8 in 5000 timesteps
- **Map Coverage**: 40-50% (small map, will discover most)
- **Key Behavior**: Opportunistic exploration during oxygen cooldown

### Feasibility Check
✅ **Feasible**: Small map with sparse but adequate resources. Oxygen cooldown is clear bottleneck. Agent can produce hearts steadily while gradually discovering outside extractors.

---

## Experiment 2: Oxygen Abundance

### Environment
- **Map**: `extractor_hub_80x80.map` (80×80)
- **Resources**: 9 Carbon, **13 Oxygen**, 14 Germanium, 10 Silicon, 13 Chargers
- **Total**: 46 extractors (large map with abundant oxygen)

### Parameters
- **Efficiency**: 100% (all extractors)
- **Max Uses**: 1000 (all extractors)
- **Energy Regen**: 1/turn (standard)

### Game Mechanics Context
- Same as Experiment 1, but **13 oxygen sources** vs 2
- Oxygen cooldown (100 turns) is the primary production bottleneck normally
- With 13 oxygen sources, can rotate to always find an available one

### Optimal Strategy
1. **Early exploration phase** (first 500 turns):
   - Discover and map oxygen locations (highest priority!)
   - Map out charger locations (13 total for waypoints)
   - Build mental model of oxygen distribution
   - Start producing hearts using discovered extractors

2. **Rotational oxygen harvesting**:
   - Keep track of which oxygen sources are on cooldown
   - Always move to nearest **available** (not on cooldown) oxygen
   - Never wait for oxygen - there are 13 sources!
   - This eliminates the 100-turn wait bottleneck

3. **Production cycle optimization**:
   - Gather germanium (14 sources, no cooldown - use nearest)
   - Gather carbon from nearest (9 sources)
   - Gather silicon with energy planning (10 sources)
   - **Find nearest available oxygen** (key innovation!)
   - Return to base for assembly/deposit

4. **Energy management**:
   - 13 chargers distributed across map enable long expeditions
   - Can reach distant oxygen if closer ones are on cooldown
   - Use chargers as waypoints for oxygen routes

### Expected Performance
- **Hearts**: 12-15 in 5000 timesteps (2x baseline due to no oxygen waiting!)
- **Map Coverage**: 60-70% (large map, must explore to find oxygen)
- **Key Behavior**: Active oxygen rotation, minimal idle time

### Feasibility Check
✅ **Feasible**: 13 oxygen sources enable true rotation strategy. Agent can maintain ~1 heart per 50-60 turns vs 100+ in baseline. Large map requires exploration but 13 chargers support it. This is the key test of "breaking the bottleneck."

---

## Experiment 3: Low Efficiency

### Environment
- **Map**: `extractor_hub_50x50.map` (50×50)
- **Resources**: 5 Carbon, 4 Oxygen, 8 Germanium, 5 Silicon, 6 Chargers
- **Total**: 22 extractors (medium map)

### Parameters
- **Efficiency**: **75%** (all extractors)
- **Max Uses**: 1000 (all extractors)
- **Energy Regen**: 1/turn (standard)

### Game Mechanics Context
- **75% efficiency changes outputs**:
  - Carbon: 3/use → need **7 uses** (vs 5) = 70 turns
  - Oxygen: 20/use → **133 turn cooldown** (vs 100)
  - Germanium: 1/use → need 5 uses (unchanged)
  - Silicon: 18.75/use → need **3 uses** (vs 2) = **75 energy** (vs 50)
  - Charger: 37.5/use → need **3 charges** (vs 2) to fill = 30 turns
- **Critical**: Silicon now costs 75 energy, assembly costs 20 = 95 total. Must charge mid-cycle!

### Optimal Strategy
1. **Energy-first planning**:
   - Heart production requires 95 energy minimum (75 for silicon + 20 for assembly)
   - Must plan charging carefully - can't do silicon + assembly in one energy bar

2. **Production cycle** (2-phase energy):
   - **Phase 1**: Harvest 2× silicon (50 energy), charge to 100 (need 2-3 charger uses)
   - **Phase 2**: Harvest 1× silicon (25 energy), now have 50 silicon total
   - Charge back to 100 energy
   - Harvest 7× carbon (70 turns), 1× oxygen (133 turn cooldown), 5× germanium
   - Assemble (20 energy), deposit

3. **Multiple extractor usage**:
   - Must use multiple carbon sources (need 7 uses, base can't keep up alone)
   - Must use multiple oxygen sources (4 total, 133 turn cooldown)
   - 8 germanium sources compensate for more uses needed
   - 6 chargers critical for silicon energy needs

4. **Charger dependency**:
   - Visit chargers frequently (need 3 uses to fill)
   - Plan routes to pass chargers before silicon runs
   - 6 chargers across map provide good coverage

### Expected Performance
- **Hearts**: 6-8 in 5000 timesteps (slower than baseline due to efficiency penalty)
- **Map Coverage**: 50-60% (must visit multiple extractors)
- **Key Behavior**: Frequent charging, multiple extractor visits per resource

### Feasibility Check
✅ **Feasible**: 75 energy for silicon is challenging but doable with 6 chargers. Must charge between silicon phases. Lower efficiency forces use of outside extractors (8 germanium, 5 carbon, 4 oxygen provides alternatives). This tests energy budgeting and multi-source harvesting.

---

## Experiment 4: Fast Depletion

### Environment
- **Map**: `extractor_hub_70x70.map` (70×70)
- **Resources**: 4 Carbon, 5 Oxygen, **16 Germanium**, 10 Silicon, 13 Chargers
- **Total**: 35 extractors (large map with many options)

### Parameters
- **Efficiency**: 100% (all extractors)
- **Max Uses**: **50** (all extractors deplete quickly)
- **Energy Regen**: 1/turn (standard)

### Game Mechanics Context
- **Depletion math** (max_uses = 50):
  - Carbon: 4/use × 50 = 200 carbon max = **10 hearts** per extractor
  - Oxygen: 20/use × 50 = 1000 oxygen max = **50 hearts** per extractor
  - Germanium: 1/use × 50 = 50 germanium max = **10 hearts** per extractor
  - Silicon: 25/use × 50 uses, but actual uses = 50/10 = **5 uses** = 125 silicon = **2.5 hearts** per extractor!
- **Silicon depletes first after ~2 hearts!**

### Optimal Strategy
1. **Phase 1 - Base exploitation** (0-2 hearts):
   - Use base extractors for first 2 hearts
   - Base silicon depletes first (only 5 actual uses)
   - Base germanium next (50 uses = ~10 hearts but need 5/heart)

2. **Phase 2 - Forced exploration** (after 2 hearts):
   - **Must explore to find outside silicon** (depleted in base!)
   - Discover silicon locations (10 total across map)
   - Discover other resources as backups
   - Map out charger network (13 chargers for long trips)

3. **Phase 3 - Distributed harvesting** (3+ hearts):
   - Rotate through multiple extractors as they deplete
   - 16 germanium sources provide many options
   - 10 silicon sources enable continued production
   - Track which extractors still have uses left

4. **Exploration priority**:
   - Silicon (highest priority - depletes fastest)
   - Germanium (second priority - need 5 uses per heart)
   - Carbon (10 hearts per extractor, less urgent)
   - Oxygen (50 hearts per extractor, rarely depletes)

### Expected Performance
- **Hearts**: 15-20 in 5000 timesteps (requires using many outside extractors)
- **Map Coverage**: 70-80% (must explore to continue production)
- **Key Behavior**: Depletion triggers exploration, uses most extractors

### Feasibility Check
✅ **Feasible**: Silicon depletion after 2 hearts forces immediate exploration. 10 silicon and 16 germanium sources provide plenty of alternatives. 13 chargers enable reaching distant extractors. This clearly tests anticipatory exploration before resources run out.

---

## Experiment 5: Energy Abundance

### Environment
- **Map**: `extractor_hub_70x70.map` (70×70)
- **Resources**: 4 Carbon, 5 Oxygen, 16 Germanium, 10 Silicon, **13 Chargers**
- **Total**: 35 extractors + 13 chargers (same map as Exp 4)

### Parameters
- **Efficiency**: 100% (all extractors)
- **Max Uses**: 1000 (all extractors)
- **Energy Regen**: **2/turn** (double standard)

### Game Mechanics Context
- **Energy changes**:
  - Normal: 1/turn → 70 turns to recover 70 energy (silicon + assembly)
  - **High**: 2/turn → **35 turns** to recover 70 energy (2x faster!)
  - During oxygen cooldown (100 turns), regen **200 energy** (vs 100)
- **Energy is no longer constraining** - passive regen handles most needs

### Optimal Strategy
1. **Aggressive exploration** (energy unconstrained):
   - Explore freely without energy anxiety
   - Can travel to distant extractors (40+ tiles = 80 energy round trip, but regen quickly)
   - Discover all 35 extractors across large map
   - Map out entire environment

2. **Long-distance expeditions**:
   - Don't need to stay near base/chargers
   - Can harvest from distant extractors efficiently
   - Still use 13 chargers opportunistically but not dependent

3. **Production optimization**:
   - Focus on finding optimal extractor locations (shortest paths)
   - Minimize total travel distance rather than energy management
   - Use nearest extractors regardless of distance from chargers

4. **Time allocation**:
   - More time exploring (60%+ of actions)
   - Less time charging (rare - only for silicon if needed)
   - Energy recovers passively during oxygen cooldown

### Expected Performance
- **Hearts**: 18-20 in 5000 timesteps (energy not limiting)
- **Map Coverage**: 75-85% (aggressive exploration)
- **Key Behavior**: Explores entire map, rarely charges, travels freely

### Feasibility Check
✅ **Feasible**: 2x energy regen removes energy as constraint. During 100-turn oxygen wait, recovers 200 energy (way more than 70 needed). Can explore entire 70×70 map without energy anxiety. This tests what happens when energy constraint is removed.

---

## Experiment 6: Energy Scarcity

### Environment
- **Map**: `extractor_hub_50x50.map` (50×50)
- **Resources**: 5 Carbon, 4 Oxygen, 8 Germanium, 5 Silicon, **6 Chargers**
- **Total**: 22 extractors + 6 chargers (medium map)

### Parameters
- **Efficiency**: 100% (all extractors)
- **Max Uses**: 1000 (all extractors)
- **Energy Regen**: **0.5/turn** (half standard)

### Game Mechanics Context
- **Energy changes**:
  - Normal: 1/turn → 70 turns to recover 70 energy
  - **Low**: 0.5/turn → **140 turns** to recover 70 energy (2x slower!)
  - During oxygen cooldown (100 turns), regen only **50 energy** (vs 100)
- **Cannot produce hearts without active charging** - passive regen insufficient

### Optimal Strategy
1. **Charger network mapping** (highest priority):
   - Discover and memorize all 6 charger locations immediately
   - Chargers are critical infrastructure, not optional
   - Build mental map of safe exploration range from each charger

2. **Constrained exploration**:
   - Never venture >15 tiles from nearest charger (30 energy round trip, have 50-70 buffer)
   - Plan routes that pass through chargers
   - Explore in "hops" between chargers

3. **Energy-intensive heart cycle**:
   - Start at 100 energy
   - Harvest 2× silicon (50 energy) → at 50 energy
   - **Must charge back to 100** (critical!)
   - Harvest other resources
   - **Must charge to 70+** before assembly (needs 20 energy)
   - Assemble and deposit
   - Return to charger

4. **Conservative gathering**:
   - Use nearest extractors only (minimize travel)
   - 22 extractors sufficient but must stay near chargers
   - Abandon distant extractors if no charger nearby
   - 6 chargers provide coverage across 50×50 map

### Expected Performance
- **Hearts**: 8-10 in 5000 timesteps (energy slows everything)
- **Map Coverage**: 45-55% (conservative exploration near chargers)
- **Key Behavior**: Frequent charging (50%+ of energy from chargers), charger-centric routes

### Feasibility Check
✅ **Feasible**: 6 chargers across 50×50 provide adequate coverage. 0.5 regen means agent must actively charge frequently. Cannot produce hearts without charging (50 energy regen in 100 turns < 70 needed). This tests charger-dependent pathfinding and conservative energy management.

---

## Experiment 7: High Efficiency

### Environment
- **Map**: `extractor_hub_50x50.map` (50×50)
- **Resources**: 5 Carbon, 4 Oxygen, 8 Germanium, 5 Silicon, 6 Chargers
- **Total**: 22 extractors (same as Exp 3 and 6)

### Parameters
- **Efficiency**: **200%** (all extractors - double!)
- **Max Uses**: 1000 (all extractors)
- **Energy Regen**: 1/turn (standard)

### Game Mechanics Context
- **200% efficiency changes outputs**:
  - Carbon: 8/use → need **3 uses** (vs 5) = 30 turns
  - Oxygen: 20/use → **50 turn cooldown** (vs 100) = 2x faster!
  - Germanium: 2/use → need **3 uses** (vs 5)
  - Silicon: 50/use → need **1 use** (vs 2) = **25 energy** (vs 50)!
  - Charger: 100/use → **fills in 1 use** (vs 2)!
- **Heart production time cut in half**: ~50 turns vs 100

### Optimal Strategy
1. **Fast gathering cycle**:
   - Harvest 3× carbon (30 turns) - very quick
   - Harvest 1× oxygen (50 turn cooldown) - half the wait!
   - Harvest 3× germanium (instant)
   - Harvest 1× silicon (25 energy) - half the cost!
   - Assemble (20 energy), deposit
   - **Total energy**: 25 + 20 = 45 (easily recoverable)

2. **Time allocation shift**:
   - Gathering takes 50% less time
   - **More time for exploration** (60%+ of actions)
   - Oxygen cooldown still exists but 50 turns (vs 100)
   - Use extra time to discover entire map

3. **Efficient charger usage**:
   - Chargers fill in one use (100 energy)
   - Very satisfying and quick
   - Rarely need charging (only need 45 per heart vs 70)

4. **Exploration optimization**:
   - Discover all 22 extractors
   - Find shortest paths to each resource type
   - Optimize routes to minimize total travel
   - Benefit from efficiency by spending saved time exploring

### Expected Performance
- **Hearts**: 15-18 in 5000 timesteps (2x production rate)
- **Map Coverage**: 70-80% (more exploration time)
- **Key Behavior**: Fast gathering, high exploration rate, efficient charger fills

### Feasibility Check
✅ **Feasible**: 200% efficiency halves gathering time and energy cost. Same map as Exp 3 allows direct comparison. Can produce hearts at ~50 turns per heart vs 100. More exploration time available. This tests how efficiency improvements translate to exploration time.

---

## Experiment 8: Zoned Resources

### Environment
- **Map**: `extractor_hub_100x100.map` (100×100 - largest!)
- **Resources**: 16 Carbon, 13 Oxygen, **37 Germanium**, 15 Silicon, 14 Chargers
- **Total**: 81 extractors (maximum resources)

### Parameters
- **Efficiency**: 100% (all extractors)
- **Max Uses**: 1000 (all extractors)
- **Energy Regen**: 1/turn (standard)

### Game Mechanics Context
- **Huge map**: 100×100 = 10,000 tiles
- **Many extractors**: 81 total, distributed across map
- **Natural corridors** may create zone-like regions
- **Travel is expensive**: Corner to corner = 200 tiles = 400 energy (4 full energy bars!)

### Optimal Strategy
1. **Zone discovery phase** (first 1000 turns):
   - Explore systematically to map extractor locations
   - Identify natural clustering patterns (corridors, open areas)
   - Notice if certain resources concentrate in areas
   - Map charger locations (14 total) as waypoints

2. **Zone recognition**:
   - Group extractors into spatial zones (e.g., "northwest cluster", "south corridor")
   - Associate zones with resource types (e.g., "west has 6 carbon")
   - Notice which zones have what mix of resources
   - Identify zones with complete sets (C+O+G+S nearby)

3. **Batched gathering by zone**:
   - Plan trips to specific zones for resources
   - Gather multiple resources in one zone trip
   - Example: "Going to northeast zone for carbon and oxygen"
   - Minimize cross-map travel by batching requests

4. **Route optimization**:
   - Establish efficient routes between base and key zones
   - Use chargers as waypoints for long trips
   - Return to base for assembly/deposit (or establish forward operating bases mentally)
   - Learn which zones are worth the travel time

### Expected Performance
- **Hearts**: 18-20 in 5000 timesteps (abundant resources but large map)
- **Map Coverage**: 60-70% (large map takes time to explore)
- **Key Behavior**: Zone-specific routes, batched gathering, spatial reasoning

### Feasibility Check
✅ **Feasible**: 81 extractors scattered across 100×100 naturally create spatial patterns. Agent must learn efficient routes rather than random wandering. Batching makes sense with long travel times. 14 chargers enable long expeditions. This tests spatial reasoning and route optimization.

---

## Experiment 9: Resource Abundance

### Environment
- **Map**: `extractor_hub_100x100.map` (100×100 - largest!)
- **Resources**: **16 Carbon, 13 Oxygen, 37 Germanium, 15 Silicon**, 14 Chargers
- **Total**: **81 extractors** (same as Exp 8, maximum resources)

### Parameters
- **Efficiency**: 100% (all extractors)
- **Max Uses**: 1000 (all extractors)
- **Energy Regen**: 1/turn (standard)

### Game Mechanics Context
- Same huge map as Exp 8
- **Key difference**: Strategy focuses on abundance vs spatial reasoning
- **37 germanium sources** = extreme abundance (need 5 uses per heart)
- **13 oxygen sources** = can always find one not on cooldown

### Optimal Strategy
1. **Early exploration for inventory** (first 1500 turns):
   - Discover and catalog all extractor locations
   - Don't focus on zones - just know where everything is
   - Build mental inventory of resource locations
   - Priority: Map oxygen and germanium (most abundant)

2. **Nearest-available policy**:
   - **For oxygen**: Find nearest oxygen extractor that is NOT on cooldown
     - With 13 sources, almost always one available nearby
     - Eliminates wait time completely
   - **For germanium**: Use nearest (37 sources = always one close)
   - **For carbon/silicon**: Use nearest available

3. **Continuous production**:
   - Never wait for cooldowns - just find another source
   - Heart production limited only by travel time, not resource availability
   - Oxygen cooldown (100 turns) becomes irrelevant with 13 sources
   - Can maintain high production rate

4. **Distance optimization**:
   - Always choose nearest available extractor
   - Minimize total travel distance
   - Don't worry about specific zones - abundance means always something nearby
   - Use any of 14 chargers opportunistically

### Expected Performance
- **Hearts**: 25-30 in 5000 timesteps (highest of all experiments)
- **Map Coverage**: 75-85% (must explore to find all resources)
- **Key Behavior**: No cooldown waiting, continuous production, uses many different extractors

### Feasibility Check
✅ **Feasible**: 81 extractors provide extreme abundance. 13 oxygen sources enable true "never wait" strategy. 37 germanium means always one nearby. This is the "ideal" scenario - tests agent performance when resources are not constraining. Should produce most hearts of any experiment.

---

## Experiment 10: Complex Mixed Optimization

### Environment
- **Map**: `extractor_hub_80x80.map` (80×80)
- **Resources**: 9 Carbon, 13 Oxygen, 14 Germanium, 10 Silicon, 13 Chargers
- **Total**: 46 extractors (large balanced map)

### Parameters
- **Efficiency**: **Mixed** - C=150%, O=75%, G=200%, S=125%
- **Max Uses**: **Mixed** - C=500, O=1000, G=300, S=800
- **Energy Regen**: **2/turn** (to help with energy management)

### Game Mechanics Context
- **Carbon (150%)**: 6/use → need 4 uses, max 500 uses = 3000 carbon = 150 hearts per extractor
- **Oxygen (75%)**: 20/use → **133 turn cooldown**, max 1000 uses = 1000 oxygen = 50 hearts per extractor
- **Germanium (200%)**: 2/use → need 3 uses, max 300 uses = 600 germanium = 120 hearts per extractor
- **Silicon (125%)**: 31.25/use → need 2 uses = 50 energy, max 800 uses = 2500 silicon = 50 hearts per extractor
- **Bottlenecks**: Silicon and oxygen will deplete first (~50 hearts each)

### Optimal Strategy
1. **Resource prioritization** (by efficiency):
   - **High priority**: Carbon (150% = 4 uses) and Germanium (200% = 3 uses)
   - **Medium priority**: Silicon (125% = 2 uses, same energy cost)
   - **Low priority**: Oxygen (75% = 133 turn cooldown, slow!)
   - Use high-efficiency extractors preferentially

2. **Depletion awareness** (by max_uses):
   - **First to deplete**: Silicon (50 hearts) and Oxygen (50 hearts)
   - **Later depletion**: Germanium (120 hearts)
   - **Last to deplete**: Carbon (150 hearts)
   - Explore for silicon and oxygen before they run out

3. **Complex cycle planning**:
   - Carbon: Use nearest (efficient at 150%)
   - Oxygen: Accept the 133 turn cooldown (13 sources help but slow)
   - Germanium: Use preferentially (very efficient at 200%)
   - Silicon: Standard (125% only slight improvement)
   - Energy: 2/turn regen helps offset planning complexity

4. **Strategic adaptation**:
   - Early game: Use most efficient extractors (carbon, germanium)
   - Mid game: Discover backups for silicon/oxygen (deplete first)
   - Late game: Spread usage across all extractors to extend lifetime
   - Observe which resources are "better" and prioritize them

### Expected Performance
- **Hearts**: 20-25 in 5000 timesteps (good but complex)
- **Map Coverage**: 65-75% (large map, must find alternatives)
- **Key Behavior**: Strategic prioritization, adaptive behavior, discovers efficiency differences

### Feasibility Check
✅ **Feasible**: Mixed parameters create meaningful optimization problem. Carbon at 150% and germanium at 200% are clearly "better." Oxygen at 75% is clearly "worse" (133 turn cooldown). Silicon and oxygen will deplete first (50 hearts), forcing exploration. Agent must balance efficiency vs availability vs depletion. 2/turn energy regen prevents energy from being the bottleneck. This tests multi-factor decision-making.

---

## Summary Table

| Exp | Strategy | Map Size | Resources | Efficiency | Max Uses | E-Regen | Expected Hearts |
|-----|----------|----------|-----------|------------|----------|---------|-----------------|
| 1 | Wait-based exploration | 30×30 | 9 | 100% | 1000 | 1 | 5-8 |
| 2 | Oxygen rotation | 80×80 | 46 | 100% | 1000 | 1 | 12-15 |
| 3 | Energy management | 50×50 | 22 | **75%** | 1000 | 1 | 6-8 |
| 4 | Depletion-driven | 70×70 | 35 | 100% | **50** | 1 | 15-20 |
| 5 | Aggressive | 70×70 | 35 | 100% | 1000 | **2** | 18-20 |
| 6 | Charger network | 50×50 | 22 | 100% | 1000 | **0.5** | 8-10 |
| 7 | Fast gathering | 50×50 | 22 | **200%** | 1000 | 1 | 15-18 |
| 8 | Zone-based | 100×100 | 81 | 100% | 1000 | 1 | 18-20 |
| 9 | Nearest-available | 100×100 | 81 | 100% | 1000 | 1 | 25-30 |
| 10 | Multi-factor | 80×80 | 46 | **Mixed** | **Mixed** | **2** | 20-25 |

---

## Running Experiments

```bash
cd packages/cogames

# Run single experiment
uv run cogames play --mission exp1.baseline --policy scripted --cogs 1 --steps 5000

# List all experiments
uv run python -c "from cogames.cogs_vs_clips.exploration_experiments import list_experiments; list_experiments()"

# Run all experiments (batch)
for i in {1..10}; do
    echo "Running Experiment $i..."
    uv run cogames play --mission exp${i}.* --policy scripted --cogs 1 --steps 5000
done
```

---

## Key Insights

### Feasibility Verified ✅
All 10 strategies are feasible given their environments:
- Resources match strategy needs (e.g., 13 oxygen for Exp 2 rotation)
- Parameters create meaningful constraints (e.g., 75% efficiency forces multiple visits)
- Map sizes appropriate for strategy complexity
- Energy/depletion mechanics support intended behaviors

### Strategic Diversity ✅
Each experiment tests unique capabilities:
1. Temporal reasoning (when to explore)
2. Resource tracking (rotation across multiple)
3. Energy budgeting (constrained resources)
4. Anticipatory planning (before depletion)
5. Unconstrained exploration (no energy limit)
6. Infrastructure navigation (charger network)
7. Time optimization (efficiency benefits)
8. Spatial reasoning (zones and batching)
9. Availability assessment (nearest-available)
10. Multi-factor optimization (complex tradeoffs)

### Progression ✅
Experiments progress from simple to complex:
- **Tutorial**: Exp 1, 7 (single main variable)
- **Intermediate**: Exp 2, 3, 5, 6 (two factors)
- **Advanced**: Exp 4, 8, 9 (strategic planning)
- **Expert**: Exp 10 (multi-factor optimization)

**All experiments ready for evaluation!** 🚀

