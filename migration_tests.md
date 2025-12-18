# LLM Agent Migration Tests

These tests verify that the migration from `mettagrid.policy` to `packages/llm_agent` didn't break functionality.

**Compare results with original logs from `test_steps.md`.**

---

## Quick Smoke Tests

### Test M1: Policy Discovery (no API calls)
```bash
uv run python -c "
from mettagrid.policy.loader import discover_and_register_policies
from mettagrid.policy.policy_registry import get_policy_registry

discover_and_register_policies()
registry = get_policy_registry()

llm_policies = {k: v for k, v in registry.items() if 'llm' in k.lower()}
print('LLM policies discovered:')
for name, path in sorted(llm_policies.items()):
    print(f'  {name}: {path}')
"
```

**Expected output:**
```
LLM policies discovered:
  llm: llm_agent.policy.llm_policy.LLMMultiAgentPolicy
  llm-anthropic: llm_agent.policy.llm_policy.LLMClaudeMultiAgentPolicy
  llm-ollama: llm_agent.policy.llm_policy.LLMOllamaMultiAgentPolicy
  llm-openai: llm_agent.policy.llm_policy.LLMGPTMultiAgentPolicy
```

- [x] All 4 policies discovered from `llm_agent.policy` (not `mettagrid.policy`)

---

## Ollama Tests

### Test M2: Ollama Basic (10 steps)
```bash
uv run cogames play -m hello_world -c 1 -s 10 -p "class=llm-ollama,kw.model=qwen2.5:7b" --render none 2>&1 | tee ollama_hello_world_1agent_10steps_post_migration.log
```

**Success criteria:**
- [ ] Agent produces valid actions (no crashes)
- [ ] No import errors in logs

---

## Claude Tests (compare with pre-migration baselines)

### Test M3: Claude C1 Equivalent (2 agents, 100 steps)
**Baseline:** C1 achieved reward 1.30, cost $5.07

```bash
uv run cogames play -m hello_world -c 2 -s 100 -p "class=llm-anthropic,kw.model=claude-sonnet-4-5,kw.context_window_size=20,kw.summary_interval=5" --render none 2>&1 | tee claude_hello_world_2agents_100steps_post_migration.log
```

**Success criteria:**
- [ ] Reward > 0.5 (baseline was 1.30)
- [ ] Both agents produce valid actions
- [ ] No import errors

**Compare with:** `claude_hello_world_2agents_100steps.log`

---

### Test M4: Claude C2 Equivalent (2 agents, 200 steps)
**Baseline:** C2 achieved reward 12.87, cost $10.79

```bash
uv run cogames play -m hello_world -c 2 -s 200 -p "class=llm-anthropic,kw.model=claude-sonnet-4-5,kw.context_window_size=20,kw.summary_interval=5" --render none 2>&1 | tee claude_hello_world_2agents_200steps_post_migration.log
```

**Success criteria:**
- [ ] Reward > 5.0 (baseline was 12.87)
- [ ] Hearts crafted

**Compare with:** `claude_hello_world_2agents_200steps.log`

---

### Test M5: Claude C6 Equivalent (2 agents, 300 steps) - Full Verification
**Baseline:** C6 achieved reward 33.84, cost $18.08

```bash
uv run cogames play -m hello_world -c 2 -s 300 -p "class=llm-anthropic,kw.model=claude-sonnet-4-5,kw.context_window_size=20,kw.summary_interval=5" --render none 2>&1 | tee claude_hello_world_2agents_300steps_post_migration.log
```

**Success criteria:**
- [ ] Reward > 10.0 (baseline was 33.84)
- [ ] Hearts crafted and deposited in chest
- [ ] Pathfinding hints appear in prompts

**Compare with:** `claude_hello_world_2agents_300steps_c6.log`

---

## OpenAI Tests

### Test M6: OpenAI Basic (2 agents, 100 steps)
```bash
uv run cogames play -m hello_world -c 2 -s 100 -p "class=llm-openai,kw.model=gpt-4o-mini,kw.context_window_size=20,kw.summary_interval=5" --render none 2>&1 | tee openai_hello_world_2agents_100steps_post_migration.log
```

**Success criteria:**
- [ ] Both agents produce valid actions
- [ ] Reward > 0
- [ ] No import errors

---

## Feature Verification

### Test M7: Debug Summary Interval
```bash
uv run cogames play -m hello_world -c 1 -s 50 -p "class=llm-ollama,kw.model=qwen2.5:7b,kw.debug_summary_interval=25" --render none 2>&1 | tee debug_summary_test_post_migration.log
```

**Success criteria:**
- [ ] File `llm_debug_summary_agent0.log` created
- [ ] Contains summaries at steps 25 and 50

---

## Results Log

| Test | Date | Reward | Cost | Status | Notes |
|------|------|--------|------|--------|-------|
| M1   |      | N/A    | $0   |        | Policy discovery |
| M2   |      |        | $0   |        | Ollama smoke test |
| M3   |      |        |      |        | Compare with C1 (1.30) |
| M4   |      |        |      |        | Compare with C2 (12.87) |
| M5   |      |        |      |        | Compare with C6 (33.84) |
| M6   |      |        |      |        | OpenAI test |
| M7   |      |        | $0   |        | Debug summary feature |

---

## Log Comparison Commands

After running tests, compare with pre-migration logs:

```bash
# Compare rewards
grep "Total reward" claude_hello_world_2agents_100steps.log claude_hello_world_2agents_100steps_post_migration.log

# Compare API calls
grep "LLM API USAGE" claude_hello_world_2agents_100steps.log claude_hello_world_2agents_100steps_post_migration.log

# Check for import errors
grep -i "import\|error\|exception" *_post_migration.log
```
