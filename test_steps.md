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
| C1   |      |        |      |       |

## Next Steps (based on C1 results)
- If exploration is weak → Add spatial memory map or pathfinding hints
- If gets stuck in loops → Add backtracking prevention
- If finds resources but doesn't craft → Add goal-oriented state machine
- If energy issues → Add better charger-seeking logic

