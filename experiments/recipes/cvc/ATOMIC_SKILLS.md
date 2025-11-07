# Atomic Skills Curriculum for CoGs vs Clips

A progressive curriculum that breaks down CVC tasks into fundamental atomic skills, teaching agents core behaviors through structured skill progression.

## Overview

The atomic skills curriculum teaches agents fundamental CVC behaviors by:
1. Starting with simple awareness and recognition tasks
2. Progressively adding complexity through resource management
3. Building up to coordination and multi-agent scenarios
4. Evaluating on the same standard CVC missions as scripted baseline agents

**Key Features:**
- 🎯 8 atomic skill missions that isolate specific behaviors
- 📊 Evaluation every 10 epochs on standard CVC missions
- 🔄 Difficulty sweep across 13 canonical difficulty levels
- 📈 Learning progress curriculum with bidirectional scoring
- 🤝 Direct comparison with scripted baseline agents (41.5% success rate)

---

## Atomic Skill Missions

### 1. Heart Awareness (`heart_awareness`)

**Goal:** Learn to recognize and value hearts in inventory

**Environment:**
- Empty 15×15 map
- Agent spawns with 1 heart in inventory
- No chest or other objects

**Reward:**
- Inventory hearts only (reward weight per agent)

**Key Learning:**
- Hearts have value
- Inventory awareness
- Basic observation

**Difficulty Buckets:**
- Map size: 10×10 to 30×30
- Episode length: 500-1500 steps
- Energy capacity: 50-200

---

### 2. Chest Discovery (`chest_discovery`)

**Goal:** Learn to locate and interact with chests

**Environment:**
- 15×15 map with one chest (starts empty)
- Agent spawns with 1 heart
- Chest accepts deposits

**Reward:**
- 0.5× inventory hearts
- 1.0× chest deposit events

**Key Learning:**
- Spatial navigation to chest
- Vibe changing (default → deposit)
- Chest interaction mechanics

**Difficulty Buckets:**
- Map size: 10×10 to 30×30
- Episode length: 500-1500 steps
- Move energy cost: 1-4

---

### 3. Chest 101 (`chest101`)

**Goal:** Deposit hearts in chest for interest growth over time

**Environment:**
- 15×15 map with one chest
- Agent spawns with 1 heart
- Hearts in chest gain interest: `H(t+1) = H(t) × (1 + rate)`

**Reward:**
- Inventory hearts + chest hearts (shared across agents)

**Key Learning:**
- Urgency to deposit early (compound interest)
- Chest hearts are shared reward
- Time-based optimization

**Difficulty Buckets:**
- Interest rates: 0.001-0.02 per step
- Chest distance: 5-40 tiles from spawn
- Number of chests: 1-3

**Why This Matters:**
Teaches agents that depositing hearts early maximizes cumulative reward through compounding.

---

### 4. Resource Awareness (`resource_awareness`)

**Goal:** Recognize that resources exist and have value

**Environment:**
- 15×15 empty map
- Agent spawns with 2 carbon + 2 oxygen
- No buildings

**Reward:**
- 0.1× carbon inventory
- 0.1× oxygen inventory

**Key Learning:**
- Resource types (carbon, oxygen)
- Multiple inventory items
- Resource valuation

**Difficulty Buckets:**
- Initial resources: 0-20 per type
- Reward weights: 0.1-1.0

---

### 5. Resource to Heart (`resource_to_heart`)

**Goal:** Convert 4 resources into heart at assembler, then deposit in chest

**Environment:**
- 15×15 map with assembler + chest
- Agent spawns with: 20 carbon, 20 oxygen, 5 germanium, 30 silicon
- Simplified assembler (heart_cost=5 instead of 10)

**Resources Needed:**
- 10 carbon (2× heart_cost)
- 10 oxygen (2× heart_cost)
- 2-5 germanium (heart_cost/2 with vibe synergy)
- 25 silicon (5× heart_cost)
- 10 energy (2× heart_cost)

**Reward:**
- 0.5× inventory hearts
- 1.0× chest hearts

**Key Learning:**
- Multi-resource crafting
- Assembler navigation
- Vibe coordination (need heart vibes for recipe)
- Recipe requirements

**Difficulty Buckets:**
- Assembler distance: 5-40 tiles
- Heart cost: 5-10
- Number of assemblers: 1-3

---

### 6. Harvest One Resource (`harvest_one_resource`)

**Goal:** Extract missing resource from chest to complete recipe

**Environment:**
- 15×15 map with chest + assembler
- Agent spawns with 3 of 4 resources (missing carbon)
- Chest contains 20 carbon

**Reward:**
- 1.0× inventory hearts

**Key Learning:**
- Chest withdrawal (negative vibe transfers)
- Resource gaps and dependencies
- Multi-step planning: chest → assembler → craft

**Difficulty Buckets:**
- Which resource is missing: carbon/oxygen/germanium/silicon
- Number of resources in chest: 1-4
- Chest resource amounts: 10-50

---

### 7. Extractor Discovery (`extractor_discovery`)

**Goal:** Find and use an extractor to obtain resources

**Environment:**
- 15×15 map with carbon extractor + assembler + chest
- Agent spawns with 3 of 4 resources (missing carbon)
- Must mine carbon from extractor

**Reward:**
- 0.5× inventory hearts
- 1.0× chest hearts

**Key Learning:**
- Extractor location and usage
- Mining mechanics
- Extractor cooldowns
- Extractor efficiency/max_uses

**Difficulty Buckets:**
- Extractor efficiency: 50-200%
- Extractor max uses: 100-1000
- Production rate: 0.1-2.0

---

### 8. Extractor Usage (`extractor_usage`)

**Goal:** Combine chest extraction with extractor mining

**Environment:**
- 20×20 map with chest + carbon extractor + assembler
- Agent spawns with no resources
- Chest contains: 20 oxygen, 5 germanium, 30 silicon
- Must mine carbon from extractor

**Reward:**
- 0.5× inventory hearts
- 1.0× chest hearts

**Key Learning:**
- Multi-source resource gathering
- Mixed extraction strategies (chest + extractor)
- Route planning optimization
- Capacity management

**Difficulty Buckets:**
- Map size: 15×15 to 30×30
- Number of extractors per type: 1-4
- Resources split: 0-4 in chest, rest from extractors

---

## Curriculum Structure

### Bucketed Curriculum

Each atomic skill samples from multiple difficulty buckets:

**Spatial Complexity:**
- Map width: 10, 15, 20, 30
- Map height: 10, 15, 20, 30
- Wall count: 0-20 (future)

**Temporal Constraints:**
- Episode length: 500, 750, 1000, 1500 steps
- Move energy cost: 1, 2, 4
- Energy capacity: 50, 100, 200

**Resource Availability:**
- Initial carbon/oxygen/germanium/silicon: 0, 5, 10, 20
- Extractor efficiency: 50-200%
- Extractor max uses: 100-1000

**Reward Shaping:**
- Heart reward weight: 0.5, 1.0, 2.0
- Chest deposit bonus: 0-0.5
- Assembly bonus: 0-1.0

**Multi-Agent (if num_cogs > 1):**
- Agent count: 1, 2, num_cogs
- Assembler simultaneous required: 1-3

### Learning Progress Algorithm

- **Type:** Bidirectional learning progress with EMA
- **Exploration bonus:** 0.15 (high for diverse skills)
- **Active tasks:** 2000 (large pool)
- **Memory tasks:** 3000
- **Slice axes:** 6 (multiple dimensions)

**How it works:**
1. Tracks fast and slow moving averages of task success
2. Prioritizes tasks with largest learning signal (fast-slow divergence)
3. Automatically discovers curriculum path through skill clusters
4. Evicts tasks with low learning progress

---

## Standard CVC Evaluations

### Evaluation Frequency

**Epoch interval:** 10 epochs (configurable via `EvaluatorConfig.epoch_interval=10`)

### Evaluation Missions (13 Standard CVC Missions)

1. **energy_starved** - Limited energy regeneration
2. **oxygen_bottleneck** - Oxygen extractors are scarce
3. **extractor_hub_30** - 30×30 hub layout
4. **extractor_hub_50** - 50×50 hub layout
5. **extractor_hub_70** - 70×70 hub layout
6. **extractor_hub_80** - 80×80 hub layout
7. **extractor_hub_100** - 100×100 hub layout
8. **collect_resources_classic** - Classic collection layout
9. **collect_resources_spread** - Spread-out resources
10. **collect_far** - Long-distance collection
11. **divide_and_conquer** - Multiple separated zones
12. **go_together** - Multi-agent coordination required
13. **single_use_swarm** - Extractors have max_uses=1

### Difficulty Levels (13 Canonical Difficulties)

#### Core Difficulties
- **story_mode** - Abundant resources (easiest)
- **standard** - Balanced baseline
- **hard** - Reduced efficiency/uses
- **brutal** - Extreme scarcity

#### Constraint Variants
- **single_use** - Extractors max_uses=1-5
- **speed_run** - 500 step limit (vs 1000 default)
- **energy_crisis** - No passive energy regen

#### Clipping Variants
- **clipped_oxygen** - Oxygen extractor starts clipped (needs decoder)
- **clipped_carbon** - Carbon extractor starts clipped (needs modulator)
- **clipped_germanium** - Germanium extractor starts clipped (needs resonator)
- **clipped_silicon** - Silicon extractor starts clipped (needs scrambler)
- **clipping_chaos** - Multiple resources clipped
- **hard_clipped_oxygen** - Clipped + harder constraints

### Evaluation Configurations

**Default (Fast):**
- 13 missions × 1 difficulty (standard) = **13 evaluations**

**Full Sweep (Scripted Agent Comparison):**
- 13 missions × 13 difficulties = **169 evaluations**

---

## Training Configurations

### Basic Training Functions

#### `train_basic_skills(num_cogs=1)`
Trains on awareness and interaction skills:
- Heart Awareness
- Chest Discovery
- Chest 101

**Best for:** Initial learning, understanding environment

#### `train_resource_skills(num_cogs=1)`
Trains on resource management:
- Resource Awareness
- Resource to Heart
- Harvest One Resource

**Best for:** Crafting, inventory management

#### `train_extraction_skills(num_cogs=1)`
Trains on extractor usage:
- Extractor Discovery
- Extractor Usage

**Best for:** Mining, resource gathering strategies

#### `train_all_atomic_skills(num_cogs=1)`
Trains on all 8 atomic skills with standard difficulty eval.

**Best for:** Complete skill acquisition

#### `train_full_difficulty_sweep(num_cogs=4)`
Trains on all 8 atomic skills with full difficulty sweep (13 difficulties).

**Best for:** Direct comparison with scripted baseline agents

---

## Usage Examples

### Quick Start - Single Agent

```bash
# Train on all atomic skills (standard difficulty eval)
uv run ./tools/run.py experiments.recipes.cvc.atomicskills.train_all_atomic_skills num_cogs=1
```

### Multi-Agent Training

```bash
# Train with 4 agents (optimal for scripted baseline)
uv run ./tools/run.py experiments.recipes.cvc.atomicskills.train_all_atomic_skills \
    num_cogs=4
```

### Custom Difficulty Sweep

```bash
# Evaluate on specific difficulties
uv run ./tools/run.py experiments.recipes.cvc.atomicskills.train_all_atomic_skills \
    num_cogs=4 \
    eval_difficulties='["standard","hard","energy_crisis"]'
```

### Full Evaluation Suite (169 evals)

```bash
# Complete difficulty sweep matching scripted agent testing
uv run ./tools/run.py experiments.recipes.cvc.atomicskills.train_full_difficulty_sweep \
    num_cogs=4

# Alternative: pass CANONICAL_DIFFICULTY_ORDER
uv run ./tools/run.py experiments.recipes.cvc.atomicskills.train_all_atomic_skills \
    num_cogs=4 \
    eval_difficulties=CANONICAL_DIFFICULTY_ORDER
```

### Debug Single Skill

```bash
# Train on single skill only
uv run ./tools/run.py experiments.recipes.cvc.atomicskills.train_single_skill \
    mission_cls=ResourceToHeartMission \
    num_cogs=1
```

### Custom Curriculum

```python
from experiments.recipes.cvc.atomicskills import (
    make_curriculum,
    HeartAwarenessMission,
    Chest101Mission,
    train,
)

# Custom curriculum with specific skills
custom_curriculum = make_curriculum(
    num_cogs=2,
    skill_missions=[HeartAwarenessMission, Chest101Mission],
    enable_detailed_slice_logging=True,
)

# Train with custom curriculum
tool = train(
    num_cogs=2,
    curriculum=custom_curriculum,
    use_standard_cvc_evals=True,
    eval_difficulties=["standard", "hard"],
)
```

---

## Skill Progression Design

### Phase 1: Awareness (Missions 1-2)
**Goal:** Basic environment understanding
- Recognize inventory items
- Locate objects
- Basic interaction

**Success Criteria:** Agent can navigate and interact

### Phase 2: Core Mechanics (Missions 3-5)
**Goal:** Understand core CVC systems
- Chest interest mechanics
- Resource types
- Basic crafting

**Success Criteria:** Agent can craft hearts and deposit

### Phase 3: Resource Gathering (Missions 6-8)
**Goal:** Multi-source resource acquisition
- Chest extraction
- Extractor mining
- Mixed strategies

**Success Criteria:** Agent can gather all required resources

### Phase 4: Optimization (Curriculum Buckets)
**Goal:** Efficiency and generalization
- Spatial complexity
- Time pressure
- Resource scarcity
- Multi-agent coordination

**Success Criteria:** Performance on standard CVC missions

---

## Comparison with Scripted Baseline

### Scripted Agent Performance (Reference)

**Overall Success Rate:** 40.9% (425/1,040 tests)

**By Agent Count:**
- 1 agent: 37.7%
- 2 agents: 41.5%
- **4 agents: 48.8%** ← Optimal
- 8 agents: 35.4%

**By Difficulty:**
- standard: 63.5%
- energy_crisis: 59.6%
- clipped_silicon: 57.7%
- hard: 55.8%
- clipped_oxygen: 50.0%
- single_use: 39.4%
- clipped_germanium: 1.9% (broken)
- clipped_carbon: 0.0% (broken)

### Evaluation Methodology

The atomic skills curriculum uses **identical evaluation conditions** as the scripted baseline:
- Same 13 missions
- Same 13 difficulty levels
- Same evaluation frequency (every 10 epochs)
- Same agent counts (1, 2, 4, 8)

This enables direct performance comparison and tracking improvement over training.

---

## Advanced Features

### Interest Rate Configuration

```python
# Configure chest interest rates in curriculum buckets
tasks.add_bucket("chest.interest_rate", [0.001, 0.005, 0.01, 0.02])
```

Interest compounds each step:
- 0.001 = +0.1% per step → ~10% after 100 steps
- 0.01 = +1% per step → ~170% after 100 steps

### Coordination Requirements

```python
# Require multiple agents at assembler simultaneously
tasks.add_bucket("assembler.simultaneous_required", [1, 2, 3])
```

Forces agents to coordinate timing for crafting.

### Resource Scarcity

```python
# Limit extractor uses
tasks.add_bucket("extractor.max_capacity", [1, 3, 5, 10])
tasks.add_bucket("num_extractors_per_type", [1, 2, 4])
```

Creates exploration pressure and resource competition.

---

## Future Skill Extensions

### Navigation Skills
- **obstacle_navigation** - Walls and barriers
- **multi_room** - Separate rooms for different stations
- **home_navigation** - Return to base assembler in complex maps

### Timing & Efficiency
- **opportunity_cost** - When to stop collecting and deposit
- **energy_management** - Limited energy, efficient paths
- **time_pressure** - Episode ending soon, prioritization

### Coordination Skills
- **parallel_collection** - Simultaneous resource gathering
- **sequential_handoff** - Agent A collects → Agent B assembles
- **synchronized_assembly** - Must arrive at assembler together
- **role_specialization** - Different agents for different tasks

### Recipe Complexity
- **recipe_scanning** - Multiple recipes, identify achievable
- **multi_stage_recipes** - Resource → Component → Heart
- **recipe_substitution** - Multiple paths to same goal

---

## Monitoring and Debugging

### Key Metrics to Track

**Learning Progress:**
- `lp/mean_learning_progress` - Average learning signal
- `lp/num_tracked_tasks` - Active task pool size
- `lp/mean_task_success_rate` - Overall success rate

**Slice Distribution:**
- `slice/num_unique_values_*` - Coverage of each dimension
- `slice/mean_success_by_*` - Performance across slices

**Evaluation:**
- Success rate by mission
- Success rate by difficulty
- Reward per mission
- Episode length utilization

### Common Issues

**Low learning progress:**
- Tasks too easy or too hard
- Increase exploration_bonus
- Adjust bucket ranges

**Poor generalization:**
- Not enough task diversity
- Increase num_active_tasks
- Add more bucket dimensions

**Evaluation failures:**
- Skills not transferring to standard missions
- May need additional intermediate skills
- Check bucket coverage matches eval difficulty

---

## File Structure

```
experiments/recipes/cvc/
├── atomicskills.py          # Main recipe implementation
├── ATOMIC_SKILLS.md         # This documentation
└── core.py                  # Standard CVC training (comparison baseline)
```

---

## References

- **Scripted Agent Evaluation:** `experiments/SCRIPTED_AGENT_EVALUATION.md`
- **CVC Missions:** `packages/cogames/src/cogames/cogs_vs_clips/evals/eval_missions.py`
- **Difficulty Variants:** `packages/cogames/src/cogames/cogs_vs_clips/evals/difficulty_variants.py`
- **Learning Progress Algorithm:** `metta/cogworks/curriculum/learning_progress_algorithm.py`
- **Curriculum System:** `metta/cogworks/curriculum/`

---

## Contributing

To add new atomic skills:

1. Create a new mission class extending `AtomicSkillMission`
2. Implement `make_env()` to configure the environment
3. Add to `ATOMIC_SKILL_MISSIONS` list
4. Document the skill in this file
5. Test with `train_single_skill(YourMission)`

Example:
```python
class YourNewSkill(AtomicSkillMission):
    name: str = "your_skill"
    description: str = "What this teaches"

    def make_env(self) -> MettaGridConfig:
        # Configure environment
        pass
```

---

**Last Updated:** 2025-11-06
**Recipe Version:** 1.0
**Evaluation Baseline:** Scripted agents (40.9% success)


