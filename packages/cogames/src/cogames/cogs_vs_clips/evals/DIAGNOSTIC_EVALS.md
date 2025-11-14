# CoGames Diagnostic Evaluations

**Purpose:** Minimal, focused tests that isolate specific agent capabilities. Each diagnostic returns **0.0 or 1.0 reward** (binary pass/fail).

**Reward Normalization:** All diagnostics cap chest capacity at 1 heart and set `chest.heart.amount` reward to 1.0, ensuring:
- ✅ **Pass:** 1 heart deposited = 1.0 reward
- ❌ **Fail:** 0 hearts deposited = 0.0 reward

**Generous Defaults:** Unless otherwise noted, agents start with:
- Large capacity (255) for hearts, cargo, gear, and energy
- Full energy regeneration each step
- Zero cooldowns on all protocols

---

## Test Categories

### 1. Chest Navigation & Deposit (5 evals)

Test basic navigation and chest interaction. Agents start with 1 heart and must deposit it.

| Eval | Agents | Max Steps | Map | Pass Criteria | Tests |
|------|--------|-----------|-----|---------------|-------|
| `diagnostic_chest_navigation1` | **1** | 250 | chest_navigation1 | Deposit 1 heart | Simple navigation to chest |
| `diagnostic_chest_navigation2` | **1** | 250 | chest_navigation2 | Deposit 1 heart | Navigate through obstacles |
| `diagnostic_chest_navigation3` | **1** | 250 | chest_navigation3 | Deposit 1 heart | Navigate obstacles (variant) |
| `diagnostic_chest_deposit_near` | **1** | 250 | chest_near | Deposit 1 heart | Chest adjacent/very close |
| `diagnostic_chest_deposit_search` | **1** | 250 | chest_search | Deposit 1 heart | Find chest outside FOV |

**Inventory Seed:** `{"heart": 1}`

**Multi-Agent Scaling:** ❌ All forced to 1 agent only

**Skills Tested:**
- Basic movement and pathfinding
- Obstacle avoidance
- Chest location discovery
- Chest interaction (deposit action)

**Redundancy Note:** 🔴 Three navigation tests (`navigation1/2/3`) may be redundant unless maps differ significantly. Consider consolidating to 1-2 variants.

---

### 2. Assembly - Seeded Resources (2 evals)

Test assembler interaction and multi-agent coordination. Agents start with all resources needed for 1 heart.

| Eval | Agents | Max Steps | Map | Pass Criteria | Tests |
|------|--------|-----------|-----|---------------|-------|
| `diagnostic_assemble_seeded_near` | **1-4** | 50 | assembler_near | All agents chorus → 1 heart | Chorus coordination (assembler visible) |
| `diagnostic_assemble_seeded_search` | **1-4** | 150 | assembler_search | All agents chorus → 1 heart | Chorus coordination + search |

**Inventory Seed:** `{"carbon": 2, "oxygen": 2, "germanium": 1, "silicon": 3}`

**Dynamic Chorus:** Assembler requires **N agents** with `heart_a` vibe simultaneously (where N = number of agents in env).

**Multi-Agent Scaling:** ✅ Complexity grows with agent count
- **1 agent:** Solo vibe + assembly
- **2 agents:** Both must coordinate vibe timing to chorus together
- **4 agents:** All 4 must synchronize vibes at assembler simultaneously

**Skills Tested:**
- Assembler location discovery
- Vibe changing (to `heart_a`)
- Multi-agent coordination and synchronization
- Chorus glyphing mechanics

---

### 3. Extraction - Missing 1 Resource (4 evals)

Test extractor interaction. Agents start with 3 of 4 resources; must extract the missing one.

| Eval | Agents | Max Steps | Map | Pass Criteria | Tests |
|------|--------|-----------|-----|---------------|-------|
| `diagnostic_extract_missing_carbon` | **1-4** | 130 | extract_lab | Extract carbon → chorus → 1 heart | Carbon extraction + chorus |
| `diagnostic_extract_missing_oxygen` | **1-4** | 130 | extract_lab | Extract oxygen → chorus → 1 heart | Oxygen extraction + chorus |
| `diagnostic_extract_missing_germanium` | **1-4** | 130 | extract_lab | Extract germanium → chorus → 1 heart | Germanium extraction + chorus |
| `diagnostic_extract_missing_silicon` | **1-4** | 130 | extract_lab | Extract silicon → chorus → 1 heart | Silicon extraction + chorus |

**Inventory Seeds:**
- Missing Carbon: `{"oxygen": 2, "germanium": 1, "silicon": 3}`
- Missing Oxygen: `{"carbon": 2, "germanium": 1, "silicon": 3}`
- Missing Germanium: `{"carbon": 2, "oxygen": 2, "silicon": 3}`
- Missing Silicon: `{"carbon": 2, "oxygen": 2, "germanium": 1}`

**Dynamic Chorus:** Requires **N agents** to chorus (where N = number of agents).

**Multi-Agent Scaling:** ✅ Scales to 4 agents
- **1 agent:** Extract resource + solo assembly
- **2-4 agents:** Extract resource + all agents must chorus to assemble

**Skills Tested:**
- Extractor location discovery
- Extractor interaction (move into extractor)
- Resource collection
- Vibe coordination for assembly

**Redundancy Note:** 🟡 All 4 tests use the same map (`extract_lab`), differing only in which resource is missing. Could be parameterized, but current explicit approach is clear for individual tracking.

---

### 4. Unclipping (2 evals)

Test unclipping mechanics. **Number of clipped extractors = number of agents.**

| Eval | Agents | Max Steps | Map | Pass Criteria | Tests |
|------|--------|-----------|-----|---------------|-------|
| `diagnostic_unclip_craft` | **1-4** | 250 | unclip | Craft decoder(s) → unclip → chorus → 1 heart | Full unclip pipeline |
| `diagnostic_unclip_preseed` | **1-4** | 250 | unclip | Unclip (preseeded decoder) → chorus → 1 heart | Unclipping without crafting |

**Inventory Seeds:**
- Craft: `{"carbon": 1, "oxygen": 1, "germanium": 1, "silicon": 1}`
- Preseed: `{"carbon": 2, "oxygen": 2, "germanium": 2, "silicon": 2}` + dynamic decoder provisioning

**Dynamic Clipping:**
- **1 agent:** 1 extractor clipped (e.g., carbon)
- **2 agents:** 2 extractors clipped (e.g., carbon, oxygen)
- **4 agents:** 4 extractors clipped (all resources)

**Unclipping Recipe:** Requires the 3 non-clipped resources (e.g., if carbon is clipped, need `{oxygen:1, germanium:1, silicon:1}` to craft decoder).

**Dynamic Chorus:** Requires **N agents** to chorus.

**Multi-Agent Scaling:** ✅ Complexity grows significantly with agent count
- More extractors to unclip
- More decoders to craft
- More coordination for chorus assembly

**Skills Tested:**
- Clipped extractor detection
- Decoder crafting at assembler
- Unclipping action (use decoder on clipped extractor)
- Resource management under clipping constraints
- Multi-agent coordination

**Difference:**
- `unclip_craft`: Tests full pipeline (craft tools from scratch)
- `unclip_preseed`: Tests unclipping action in isolation (tools provided)

---

### 5. Energy Management (1 eval)

Test energy management and charger usage.

| Eval | Agents | Max Steps | Map | Pass Criteria | Tests |
|------|--------|-----------|-----|---------------|-------|
| `diagnostic_charge_up` | **1** | 250 | charge_up | Charge to sufficient energy → deposit heart | Energy management + charger |

**Inventory Seed:** `{"heart": 1, "energy": 60}`

**Energy Configuration:**
- Starting energy: 60 (low)
- Energy regeneration: **0** (disabled)
- Must use charger to gain energy

**Multi-Agent Scaling:** ❌ Forced to 1 agent

**Skills Tested:**
- Charger location discovery
- Charger interaction
- Energy state tracking
- Planning actions under energy constraints

---

### 6. Memory & Exploration (2 evals)

Test memory, exploration, and full mission cycles.

| Eval | Agents | Max Steps | Map | Pass Criteria | Tests |
|------|--------|-----------|-----|---------------|-------|
| `diagnostic_memory` | **1** | 110 | memory | Remember chest location → deposit heart | Memory/exploration (long distance) |
| `diagnostic_radial` | **1-4** | 250 | radial | Gather from 4 radial extractors → chorus → 1 heart | Full mission with radial layout |

**Inventory Seeds:**
- Memory: `{"heart": 1}`
- Radial: `{"energy": 255}` (full energy)

**Dynamic Chorus (Radial only):** Requires **N agents** to chorus.

**Multi-Agent Scaling:**
- Memory: ❌ Single agent only
- Radial: ✅ Scales to 4 agents with chorus requirement

**Skills Tested:**
- **Memory:** Long-distance navigation, maintaining target position in memory
- **Radial:** Full mission cycle (extract 4 resources, assemble, deposit), radial map navigation

---

### 7. Agility (1 eval)

Test navigation agility in complex terrain.

| Eval | Agents | Max Steps | Map | Pass Criteria | Tests |
|------|--------|-----------|-----|---------------|-------|
| `diagnostic_agile` | **1-4** | 250 | agile | Navigate obstacles → extract → chorus → 1 heart | Complex navigation + full cycle |

**Inventory Seed:** None (empty inventory)

**Extractor Configuration:** Each extractor has `max_uses=1` and outputs exactly the amount needed for 1 heart recipe.

**Dynamic Chorus:** Requires **N agents** to chorus.

**Multi-Agent Scaling:** ✅ Scales to 4 agents

**Skills Tested:**
- Navigation through complex/narrow terrain
- Pathfinding around obstacles
- Single-use resource management
- Full mission cycle under navigation constraints

---

## Multi-Agent Coverage Summary

| Agent Count | # Evals | Eval Names |
|-------------|---------|------------|
| **1 agent only** | 8 | Chest Navigation (3), Chest Deposit (2), Charge Up, Memory |
| **1-4 agents (dynamic)** | 9 | Assemble Seeded (2), Extract Missing (4), Unclip (2), Radial, Agile |
| **Total** | 17 | All diagnostics |

**Dynamic Scaling Pattern:**
- **1 agent:** Tests basic individual capabilities
- **2-4 agents:** Adds chorus coordination requirement (all agents must synchronize)
- Complexity grows with agent count due to coordination overhead

---

## Pass/Fail Criteria

All diagnostics use **binary pass/fail** scoring:

- ✅ **Pass (1.0 reward):** Agent(s) successfully deposit 1 heart into chest
- ❌ **Fail (0.0 reward):** No heart deposited by max_steps

**Partial Credit:** Not available (chest capacity capped at 1 heart, reward normalized)

**Timeout:** Reaching `max_steps` without depositing a heart results in failure (0.0 reward)

---

## Redundancy Analysis & Recommendations

### 🔴 High Redundancy

**Chest Navigation Tests (3 evals):** `chest_navigation1`, `chest_navigation2`, `chest_navigation3`
- All test single-agent navigation to chest with obstacles
- **Recommendation:** Consolidate to 1-2 variants unless maps are significantly different
- **Rationale:** Testing obstacle navigation once is sufficient; multiple variants add evaluation time without new insights

### 🟡 Medium Redundancy

**Extract Missing Tests (4 evals):** All use same map (`extract_lab`), differing only in which resource is missing
- **Recommendation:** Consider parameterizing as `DiagnosticExtractMissing(resource: str)` with 4 instances
- **Rationale:** Current explicit approach is clear for tracking individual resource extraction, but could be DRY-er
- **Keep as-is if:** Individual resource extraction rates differ significantly or need separate tracking/reporting

### ✅ No Redundancy

All other diagnostics are well-differentiated:
- Assembly tests: Near vs Search (different complexity)
- Unclipping tests: Craft vs Preseed (different skill focus)
- Single-purpose tests: Charge, Memory, Agile, Radial (unique mechanics)

---

## Gaps & Future Additions

### Missing Test Coverage

1. **Multi-Agent Resource Contention**
   - Currently no tests for multiple agents competing for same extractor
   - **Suggestion:** Add diagnostic with 4 agents, 1 extractor per resource type (forces queuing/collision handling)

2. **Fixed 2/4 Agent Tests**
   - All multi-agent tests are dynamic (support 1-4)
   - **Suggestion:** Add explicit 2-agent and 4-agent tests to ensure proper scaling validation

3. **Vibe Swapping Under Pressure**
   - No tests for rapid vibe changes or vibe management under time constraints
   - **Suggestion:** Add diagnostic requiring multiple vibe swaps in sequence

4. **Chest Withdrawal**
   - All chest tests are deposits; no tests for withdrawing resources from chests
   - **Suggestion:** Add diagnostic requiring agent to withdraw resources from resource chest

5. **Multi-Station Coordination**
   - No tests requiring agents to coordinate across multiple station types simultaneously
   - **Suggestion:** Add diagnostic with parallel assembly + extraction tasks

---

## Usage

### Running Individual Diagnostics

```bash
# Single diagnostic, 1 agent
uv run cogames play -m evals.diagnostic_chest_navigation1 -c 1

# Single diagnostic, 4 agents (for dynamic tests)
uv run cogames play -m evals.diagnostic_radial -c 4
```

### Running All Diagnostics

```python
from cogames.cogs_vs_clips.evals.diagnostic_evals import DIAGNOSTIC_EVALS

for diagnostic_class in DIAGNOSTIC_EVALS:
    diagnostic = diagnostic_class()
    # Run evaluation...
```

### Using Evaluation Script

```bash
# Evaluate baseline agent on all diagnostics
cd packages/cogames
uv run python scripts/evaluate_scripted_agents.py \
  --agent baseline \
  --mission-set diagnostic_evals \
  --cogs 1 2 4 \
  --steps 300
```

---

## Diagnostic Maps

Maps are located in: `packages/cogames/src/cogames/maps/evals/`

**Map Naming Convention:** `diagnostic_<test_focus>.map`

**Available Maps:**
- `diagnostic_chest_navigation1.map`, `diagnostic_chest_navigation2.map`, `diagnostic_chest_navigation3.map`
- `diagnostic_chest_near.map`, `diagnostic_chest_search.map`
- `diagnostic_assembler_near.map`, `diagnostic_assembler_search.map`
- `diagnostic_extract_lab.map`
- `diagnostic_unclip.map`
- `diagnostic_charge_up.map`
- `diagnostic_memory.map`
- `diagnostic_agile.map`
- `diagnostic_radial.map`

---

## Implementation Details

### Base Class: `_DiagnosticMissionBase`

**Key Configuration Options:**
- `max_steps`: Episode timeout (default: 250)
- `required_agents`: Force specific agent count (default: None = 1-4 flexible)
- `inventory_seed`: Starting inventory for agents
- `communal_chest_hearts`: Pre-stock communal chest with hearts
- `resource_chest_stock`: Pre-stock resource chests
- `clip_extractors`: Set of extractors to start clipped
- `extractor_max_uses`: Max uses per extractor
- `assembler_heart_chorus`: Number of agents required for chorus (1 = solo)
- `dynamic_assembler_chorus`: If True, set chorus = num_agents
- `generous_energy`: If True, full energy regen each step (default: True)

**Automatic Modifications:**
- All capacities set to 255 (hearts, cargo, gear, energy)
- All protocol cooldowns set to 0 (instant interactions)
- Reward normalization: 1 heart deposited = 1.0 reward
- Chest capacity capped at 1 heart per episode

**Customization Hook:**
```python
def configure_env(self, cfg: MettaGridConfig) -> None:
    # Override this method in subclasses to apply custom modifications
    pass
```

---

**Last Updated:** November 14, 2025

**Total Diagnostics:** 17

**Agent Counts Tested:** 1, 2, 4 (1-4 for dynamic tests)

**Total Test Configurations:** 25+ (17 diagnostics × variable agent counts)


