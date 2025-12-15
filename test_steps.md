# LLM Policy Testing Plan

# QWEN TESTS

## Goal
Test if Qwen (local model) can perform well on cogames missions, then scale up to longer runs on harder missions.

## New Feature: debug_summary_interval
Added `debug_summary_interval` parameter that generates LLM summaries every N steps and writes to file `llm_debug_summary_agent{id}.log`. This helps debug long runs without reading through all actions.

- Set to 0 to disable (default)
- Set to 100 to get summaries every 100 steps
- Uses the same LLM model as the agent

## Test Sequence

### Test 1: Qwen baseline (1 agent, 200 steps, hello_world)
```bash
uv run cogames play -m hello_world -c 1 -s 200 -p "class=llm-ollama,kw.model=qwen3-coder:30b,kw.context_window_size=20,kw.summary_interval=5" --render none 2>&1 | tee qwen_hello_world_1agent_200steps.log
```
**Success criteria:**
- [ ] Agent doesn't hallucinate actions (valid JSON output)
- [ ] Agent collects at least 2 different resources
- [ ] Reward > 0.3

---

### Test 2: Qwen 2 agents (if Test 1 passes)
```bash
uv run cogames play -m hello_world -c 2 -s 200 -p "class=llm-ollama,kw.model=qwen3-coder:30b,kw.context_window_size=20,kw.summary_interval=5" --render none 2>&1 | tee qwen_hello_world_2agents_200steps.log
```
**Success criteria:**
- [ ] Both agents produce valid actions
- [ ] Combined reward > 0.5
- [ ] No context confusion between agents

---

### Test 3: Claude hello_world extended (1 agent, 300 steps)
```bash
uv run cogames play -m hello_world -c 1 -s 300 -p "class=llm-ollama,kw.model=qwen3-coder:30b,kw.context_window_size=20,kw.summary_interval=5,kw.debug_summary_interval=100" --render none 2>&1 | tee claude_hello_world_1agent_300steps.log
```
**Success criteria:**
- [ ] Agent crafts at least 1 heart
- [ ] Reward > 1.0

---

### Test 4: Machina 1 long run with debug summaries (if Test 3 passes)
```bash
uv run cogames play -m machina_1 -c 2 -s 1000 -p "class=llm-ollama,kw.model=qwen3-coder:30b,kw.context_window_size=20,kw.summary_interval=10,kw.debug_summary_interval=100" --render none 2>&1 | tee claude_machina1_2agents_1000steps.log
```
**Success criteria:**
- [ ] At least 1 agent collects all 4 resource types
- [ ] Reward > 0 (any heart crafted)
- [ ] Debug summaries written to `llm_debug_summary_agent0.log` and `llm_debug_summary_agent1.log`

---

## Results Log

| Test | Date | Reward | Cost | Notes |
|------|------|--------|------|-------|
| 1    | 12/12 | 0.17  | $0   | PARTIAL FAIL - Valid JSON, but only found germanium. Got stuck in loops. Position hallucination ("At 5,5" constantly). Tried invalid action (move_southwest). |
| 2    | 12/12 | 0.34 (0.17+0.17) | $0 | FAIL - Both agents valid JSON, but both only found germanium. Agent 1 got stuck repeatedly trying to "move into" extractor. Combined reward < 0.5. |
| 3    | 12/12 | 0.17  | $0   | FAIL - Qwen only found germanium (2-4 units). Never found carbon/oxygen/silicon. Debug summaries work! Agent explored 130 tiles but got stuck in loops. |
| 4    | 12/12 | 0.00  | $0   | FAIL - 1000 steps, 2 agents, machina_1. Agents explored 231 tiles but only collected germanium (4 units). Never found other extractors. Energy ran critically low (1-63). Debug summaries generated successfully. |

## Notes
- Qwen runs locally (free)
- Claude Sonnet costs ~$0.025/step/agent
- hello_world recipe: 1 carbon, 1 oxygen, 1 germanium, 1 silicon
- machina_1 recipe: 10 carbon, 10 oxygen, 2 germanium, 30 silicon
- Debug summaries: Every `debug_summary_interval` steps, writes to `llm_debug_summary_agent{id}.log`


# CLAUDE TESTS

## Goal
Get agents to craft hearts on hello_world, then machina_1. Start with diagnostic test to understand what Claude does well/poorly with current tools.

## Test Sequence

### Test C1: Claude baseline diagnostic (2 agents, 100 steps, hello_world)
```bash
uv run cogames play -m hello_world -c 2 -s 100 -p "class=llm-anthropic,kw.model=claude-sonnet-4-5,kw.context_window_size=20,kw.summary_interval=5" --render none 2>&1 | tee claude_hello_world_2agents_100steps.log
```
**Estimated cost:** ~$5.00

**What we're measuring:**
- [ ] Does each agent find multiple extractor types?
- [ ] Does the agent understand how to collect resources (step onto extractor)?
- [ ] Does the agent avoid getting stuck in loops?
- [ ] Does the agent manage energy properly?
- [ ] Does the agent understand the crafting goal?

**Analysis questions:**
1. How many unique extractors did each agent discover?
2. What resources were collected?
3. Did agents waste steps (loops, backtracking)?
4. Did agents get close to assembler with resources?

---

## Results Log

| Test | Date | Reward | Cost | Notes |
|------|------|--------|------|-------|
| C1   | 12/12 | 1.30 (0.35 + 0.95) | $5.07 | SUCCESS! Much better than Qwen. Agent 0: carbon(2), silicon(15). Agent 1: germanium(2), oxygen(10), silicon(15). Found 3 of 4 extractor types! |
| C2   | 12/12 | **12.87** (5.95 + 6.92) | $10.79 | **SUCCESS!** Both agents collected resources. Agent 1 had C=2, Ge=2, Si=15 (missing only O). High rewards suggest hearts were crafted! |

---

### Test C2: Claude extended run (2 agents, 200 steps, hello_world)
```bash
uv run cogames play -m hello_world -c 2 -s 200 -p "class=llm-anthropic,kw.model=claude-sonnet-4-5,kw.context_window_size=20,kw.summary_interval=5" --render none 2>&1 | tee claude_hello_world_2agents_200steps.log
```
**Estimated cost:** ~$10.00

**What we're measuring:**
- [ ] Can agents find all 4 extractor types with more time?
- [ ] Do agents craft at least 1 heart?
- [ ] Do agents deposit hearts in chest?

**Success criteria:**
- At least 1 agent collects all 4 resources (carbon, oxygen, germanium, silicon)
- Reward > 2.0 (indicating heart crafted/deposited)

---

---

### Test C3: Persistent extractor memory + agent inventory tracking (2 agents, 200 steps, hello_world)

**New features to implement:**
1. **Persistent extractor memory with visit counts**
   - Track: "oxygen_extractor at 15N8E (visited 3 times, collected 6 oxygen)"
   - Include in every prompt so agent knows where to return

2. **Other agents' last seen inventory**
   - Track: "Agent 1 last seen at 5N3E with: carbon=2, silicon=15"
   - Helps coordination - agent knows what others have collected

```bash
uv run cogames play -m hello_world -c 2 -s 200 -p "class=llm-anthropic,kw.model=claude-sonnet-4-5,kw.context_window_size=20,kw.summary_interval=5" --render none 2>&1 | tee claude_hello_world_2agents_200steps_c3.log
```
**Estimated cost:** ~$10.00

**What we're measuring:**
- [ ] Do agents return to known extractors instead of re-exploring?
- [ ] Do agents use knowledge of other agents' inventories?
- [ ] Faster resource collection than C2?

**Success criteria:**
- Reward > 15 (improvement over C2's 12.87)
- OR faster heart crafting (hearts deposited before step 150)

**C3 Results:**
| Test | Reward | Cost   | Notes             |
|------|--------|--------|-------------------|
| C2   | 12.87  | $10.79 | Baseline          |
| C3   | 1.12   | $10.76 | REGRESSION - wall navigation issue |

**Analysis:** The new features (extractor memory, other agents' inventories) ARE working - agents can see `"Agent 1 at 10N8E with germanium=2, oxygen=10"`. However, performance dropped because both agents got stuck trying to navigate around walls to reach visible extractors. The bottleneck is **pathfinding**, not memory.

---

### Test C4: Verify AssemblerDrawsFromChestsVariant detection

**Goal:** Confirm we can detect whether a mission has the AssemblerDrawsFromChestsVariant active (assembler can pull resources from nearby chests).

**Test commands:**
```bash
# Test 1: hello_world (should NOT have variant)
uv run cogames play -m hello_world -c 1 -s 1 -p "class=llm-ollama,kw.model=qwen3-coder:30b" --render none 2>&1 | tee has_assembler_draws_from_chest_variant_hello_world_false.log

# Test 2: machina_1.open_world_with_chests (should HAVE variant)
uv run cogames play -m machina_1.open_world_with_chests -c 1 -s 1 -p "class=llm-ollama,kw.model=qwen3-coder:30b" --render none 2>&1 | tee has_assembler_draws_from_chest_variant_machina1_true.log

# Check results
grep "AssemblerDrawsFromChestsVariant" has_assembler_draws_from_chest_variant_*.log
```

**Expected output:**
- `has_assembler_draws_from_chest_variant_hello_world_false.log`: `[DEBUG AssemblerDrawsFromChestsVariant] NOT ACTIVE - chest_search_distance=0`
- `has_assembler_draws_from_chest_variant_machina1_true.log`: `[DEBUG AssemblerDrawsFromChestsVariant] ACTIVE - chest_search_distance=2`

**Success criteria:**
- [x] hello_world shows NOT ACTIVE
- [x] open_world_with_chests shows ACTIVE with chest_search_distance > 0

---

### Test C5: BFS Pathfinding hints 

**New feature:** BFS pathfinding within visible 11x11 grid. For each important visible object (extractors, assembler, chest, charger), calculates the optimal first move to reach it, avoiding walls.

**Prompt output format:**
```
=== PATHFINDING HINTS ===
These are the BEST FIRST MOVES to reach important objects (avoiding walls):

  assembler (3N2E, 5 tiles) -> move_east
  carbon_extractor (2S1W, 3 tiles) -> move_south
  silicon_extractor (4N, 4 tiles) -> NO PATH (blocked)

IMPORTANT: Follow these hints to navigate around walls efficiently!
```

```bash
uv run cogames play -m hello_world -c 2 -s 200 -p "class=llm-anthropic,kw.model=claude-sonnet-4-5,kw.context_window_size=20,kw.summary_interval=5" --render none 2>&1 | tee claude_hello_world_2agents_200steps_c5.log
```
**Estimated cost:** ~$10.00

**What we're measuring:**
- [ ] Do agents follow pathfinding hints to navigate around walls?
- [ ] Do agents reach extractors faster than C3?
- [ ] Reward improvement over C3 (1.12)?

**Success criteria:**
- Reward > 5.0 (improvement over C3's 1.12)
- At least 1 agent collects 3+ resource types

**C5 Results:**
| Test | Reward | Cost | Notes |
|------|--------|------|-------|
| C5 | 2.27 (0.35 + 1.92) | $11.37 | Pathfinding WORKS! Agent 1 collected ALL 4 resources, was 2 tiles from assembler when episode ended. Map had only 1 carbon extractor (bad RNG). Agent 0 explored 129 tiles but never found carbon. |

**C5 Analysis:**
- Pathfinding hints are being followed ("Following pathfinding hint to move_south")
- Agent 1 had: carbon=2, oxygen=10, germanium=2, silicon=15 (ALL resources!)
- Episode ended before Agent 1 could craft
- Map comparison shows C2 had 3 carbon extractors vs C5's 1 (RNG difference)

---

### Test C6: Extended run to verify pathfinding (2 agents, 300 steps, hello_world)

**Hypothesis:** C5's Agent 1 was about to craft a heart when the episode ended. With 300 steps instead of 200, agents should have enough time to complete the crafting cycle despite map RNG.

```bash
uv run cogames play -m hello_world -c 2 -s 300 -p "class=llm-anthropic,kw.model=claude-sonnet-4-5,kw.context_window_size=20,kw.summary_interval=5" --render none 2>&1 | tee claude_hello_world_2agents_300steps_c6.log
```
**Estimated cost:** ~$17.00

**What we're measuring:**
- [ ] Do agents craft at least 1 heart?
- [ ] Do agents deposit hearts in chest?
- [ ] Reward comparable to C2 (12.87)?

**Success criteria:**
- Reward > 8.0 (accounting for map RNG)
- At least 1 heart crafted and deposited

**C6 Results:**
| Test | Reward | Cost | Notes |
|------|--------|------|-------|
| C6 | **33.84** (16.92 + 16.92) | $18.08 | **SUCCESS!** 2.6x better than C2! Agent 1 crafted hearts, deposited 6 hearts in chest. Both agents got equal rewards. Pathfinding confirmed working! |

**C6 Analysis:**
- Agent 1: "Excellent! Crafted a heart" → "Heart deposited (chest now has 6)"
- Both agents achieved 16.92 reward each (perfectly balanced)
- Extra 100 steps (vs C5) allowed multiple crafting cycles
- Agent 1 was going back for MORE resources when episode ended

**Hypothesis CONFIRMED:**
1. Pathfinding hints work - agents navigate efficiently around walls
2. C5 just needed more time - 200 steps wasn't enough for full cycle
3. 300 steps allows multiple heart crafting and depositing cycles

---

## Results Log Update

| Test | Date | Reward | Cost | Notes |
|------|------|--------|------|-------|
| C3   | 12/12 | 1.12 (0.17 + 0.95) | $10.76 | REGRESSION - Features work but wall navigation blocked both agents. Agent 0: carbon=2 (missing O/Ge/Si). Agent 1: O=10, Ge=2, Si=15 (missing 1 carbon). Both got stuck on walls trying to reach visible extractors. |
| C5   | 12/14 | 2.27 (0.35 + 1.92) | $11.37 | Pathfinding works! Agent 1 had all resources, 2 tiles from assembler at end. Bad map RNG (only 1 carbon extractor). |
| C6   | 12/14 | **33.84** (16.92 + 16.92) | $18.08 | **SUCCESS!** 2.6x better than C2! 6 hearts deposited. Pathfinding + extra time = major improvement. |

---

### Test C6.5: Quick machina_1 diagnostic (2 agents, 50 steps)

**Goal:** Verify pathfinding works on the larger 200x200 map before committing to expensive long runs.

```bash
uv run cogames play -m machina_1 -c 2 -s 50 -p "class=llm-anthropic,kw.model=claude-sonnet-4-5,kw.context_window_size=20,kw.summary_interval=5" --render none 2>&1 | tee claude_machina1_2agents_50steps_c65.log
```
**Estimated cost:** ~$3.00

**What we're measuring:**
- [ ] Do pathfinding hints generate correctly on larger map?
- [ ] Do agents find at least 1 extractor type?
- [ ] Do agents follow pathfinding hints (check logs for "Following pathfinding hint")?
- [ ] Any obvious issues with energy on larger map?

**Success criteria:**
- Agents produce valid actions (no crashes)
- At least 1 resource collected
- Pathfinding hints visible in prompts

**C6.5 Results:**
| Test | Reward | Cost | Notes |
|------|--------|------|-------|
| C6.5 |        |      |       |

---

## C7: machina_1 Tests

**Goal:** Get at least 1 heart crafted on machina_1 (much harder than hello_world)

**Challenge comparison:**
| Factor | hello_world | machina_1 |
|--------|-------------|-----------|
| Carbon needed | 1 | 10 |
| Oxygen needed | 1 | 10 |
| Germanium needed | 1 | 2 |
| Silicon needed | 1 | **30** |
| Total resources | 4 | 52 (13x more) |
| Map size | ~50x50 | 200x200 (16x larger) |

**Success metrics (since full heart is difficult):**
- [ ] Total resources collected by all agents
- [ ] Number of unique extractor types found (4 = all)
- [ ] Did any agent collect significant silicon (need 30)?
- [ ] Energy management on larger map
- [ ] Any heart crafted?

---

### Test C7a: Diagnostic run (500 steps, 2 agents)

**Goal:** Understand machina_1 difficulty, measure resource collection rate.

```bash
uv run cogames play -m machina_1 -c 2 -s 500 -p "class=llm-anthropic,kw.model=claude-sonnet-4-5,kw.context_window_size=20,kw.summary_interval=5" --render none 2>&1 | tee claude_machina1_2agents_500steps_c7a.log
```
**Estimated cost:** ~$30.00

**What we're measuring:**
- How many resources can 2 agents collect in 500 steps?
- Do agents find all 4 extractor types?
- What's the resource collection rate per step?

---

### Test C7b: More agents (500 steps, 4 agents)

**Goal:** Test if parallelizing with more agents speeds up resource collection.

```bash
uv run cogames play -m machina_1 -c 4 -s 500 -p "class=llm-anthropic,kw.model=claude-sonnet-4-5,kw.context_window_size=20,kw.summary_interval=5" --render none 2>&1 | tee claude_machina1_4agents_500steps_c7b.log
```
**Estimated cost:** ~$60.00

**What we're measuring:**
- Does 4 agents collect 2x resources vs 2 agents?
- Do agents coordinate or compete for same extractors?
- Any agent reach assembler with partial resources?

---

### Test C7c: Longer run (1000 steps, 2 agents)

**Goal:** Test if longer runs help with the larger 200x200 map.

```bash
uv run cogames play -m machina_1 -c 2 -s 1000 -p "class=llm-anthropic,kw.model=claude-sonnet-4-5,kw.context_window_size=20,kw.summary_interval=5" --render none 2>&1 | tee claude_machina1_2agents_1000steps_c7c.log
```
**Estimated cost:** ~$60.00

**What we're measuring:**
- Can 2 agents explore enough of 200x200 map in 1000 steps?
- Do agents make multiple trips to extractors?
- Any heart crafted with extended time?

---

### C7 Results Log

| Test | Date | Reward | Cost | Resources Collected | Notes |
|------|------|--------|------|---------------------|-------|
| C7a  |      |        |      |                     |       |
| C7b  |      |        |      |                     |       |
| C7c  |      |        |      |                     |       |

---

## Next Steps (based on C1 results)
- If exploration is weak → Add spatial memory map or pathfinding hints
- If gets stuck in loops → Add backtracking prevention
- If finds resources but doesn't craft → Add goal-oriented state machine
- If energy issues → Add better charger-seeking logic

