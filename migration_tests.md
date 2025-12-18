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

---

# Phase 2: Clean Code Refactoring

After migration tests pass, refactor the code following clean code principles.

## Current State

| File | Lines | Issues |
|------|-------|--------|
| `llm_policy.py` | ~1900 | Too large, multiple responsibilities |
| `prompt_builder.py` | ~900 | Acceptable size |
| `observation_debugger.py` | ~250 | Good |

## Refactoring Tasks

### R1: Extract Model Configuration (`model_config.py`)

Move from `llm_policy.py`:
- [ ] `MODEL_CONTEXT_WINDOWS` dict
- [ ] `TOKENS_PER_STEP_BASE`, `TOKENS_PER_CONVERSATION_TURN` constants
- [ ] `get_model_context_window()` function
- [ ] `estimate_required_context()` function
- [ ] `validate_model_context()` function
- [ ] OpenAI/Anthropic pricing dicts

**Test after:**
```bash
uv run python -c "from llm_agent.model_config import MODEL_CONTEXT_WINDOWS, calculate_llm_cost; print('OK')"
```

---

### R2: Extract Cost Tracker (`cost_tracker.py`)

Create `CostTracker` class to replace class-level state:
- [ ] `total_calls`, `total_input_tokens`, `total_output_tokens`, `total_cost`
- [ ] `calculate_llm_cost()` function
- [ ] `get_cost_summary()` method
- [ ] `_print_cost_summary_on_exit()` function

**Test after:**
```bash
uv run python -c "from llm_agent.cost_tracker import CostTracker; t = CostTracker(); print('OK')"
```

---

### R3: Extract Provider Abstraction (`providers/`)

Create provider abstraction to eliminate repeated if/elif branches:

```
providers/
├── __init__.py
├── base.py           # Abstract LLMProvider class
├── openai.py         # OpenAIProvider
├── anthropic.py      # AnthropicProvider
└── ollama.py         # OllamaProvider
```

Each provider implements:
- [ ] `chat(messages, temperature, max_tokens) -> (response, usage)`
- [ ] `select_model() -> str` (interactive model selection)
- [ ] `get_available_models() -> list[str]`

**Test after:**
```bash
uv run python -c "from llm_agent.providers import OpenAIProvider, AnthropicProvider, OllamaProvider; print('OK')"
```

---

### R4: Extract Exploration Tracker (`exploration_tracker.py`)

Move from `LLMAgentPolicy`:
- [ ] `_global_x`, `_global_y` position tracking
- [ ] `_current_window_positions`, `_all_visited_positions`
- [ ] `_discovered_objects` dict
- [ ] `_extractor_stats` dict
- [ ] `_other_agents_info` dict
- [ ] `_extract_discovered_objects()` method
- [ ] `_get_discovered_objects_text()` method

**Test after:**
```bash
uv run python -c "from llm_agent.exploration_tracker import ExplorationTracker; t = ExplorationTracker(); print('OK')"
```

---

### R5: Extract Action Parser (`action_parser.py`)

Move from `LLMAgentPolicy`:
- [ ] `_parse_action()` method -> `ActionParser.parse()`
- [ ] Action validation logic
- [ ] JSON extraction from LLM response

**Test after:**
```bash
uv run python -c "from llm_agent.action_parser import ActionParser; print('OK')"
```

---

### R6: Remove Duplicate Code

Fix duplicated `pos_to_dir()` function (appears 4 times):
- [ ] Create single `pos_to_dir()` in utils module
- [ ] Replace all 4 occurrences with import

**Verify:**
```bash
grep -n "def pos_to_dir" packages/llm_agent/src/llm_agent/**/*.py
# Should return only 1 result after fix
```

---

### R7: Simplify Core Policy

After extractions, `LLMAgentPolicy` should only contain:
- [ ] Provider initialization
- [ ] `step()` method (simplified)
- [ ] Conversation history management
- [ ] `_add_to_messages()`, `_get_messages_for_api()`

**Target:** < 500 lines

---

## Post-Refactoring Tests

After all refactoring, re-run migration tests to verify no regressions:

```bash
# Quick smoke test
uv run cogames play -m hello_world -c 1 -s 10 -p "class=llm-ollama,kw.model=qwen2.5:7b" --render none

# Full verification
uv run cogames play -m hello_world -c 2 -s 300 -p "class=llm-anthropic,kw.model=claude-sonnet-4-5,kw.context_window_size=20,kw.summary_interval=5" --render none 2>&1 | tee claude_post_refactor_verification.log
```

---

## Refactoring Progress

| Task | Status | Notes |
|------|--------|-------|
| R1: model_config.py | [x] Complete | MODEL_CONTEXT_WINDOWS, pricing, validation |
| R2: cost_tracker.py | [x] Complete | Singleton CostTracker class |
| R3: providers.py | [x] Complete | Ollama/OpenAI/Anthropic model selection |
| R4: exploration_tracker.py | [x] Complete | Position, discovery, other agents tracking |
| R5: action_parser.py | [x] Complete | LLM response parsing |
| R6: Remove duplicates | [x] Complete | pos_to_dir consolidated in utils.py |
| R7: Simplify policy | [~] Partial | 1114 lines (from ~1900, 41% reduction). <500 requires step() refactoring |
| Post-refactor tests | [ ] Pending | |

### Created Files
- `packages/llm_agent/src/llm_agent/model_config.py`
- `packages/llm_agent/src/llm_agent/cost_tracker.py`
- `packages/llm_agent/src/llm_agent/providers.py`
- `packages/llm_agent/src/llm_agent/exploration_tracker.py`
- `packages/llm_agent/src/llm_agent/action_parser.py`
- `packages/llm_agent/src/llm_agent/utils.py`





- [ ] validate model_config.py -> add gpt 5.2 and test
- [ ] validate providers.py
- [ ] validate exploration_tracker.py
- [ ] validate action_parser
- [ ] validate utils
- [ ] llm_policy.py
  - [ ] initiating models is still with a lot of code instead of using clean code
- [ ] use cogames/scripted_agent/utils.py to enhance my code
- [ ] make sure no unused functions go in

- [x] cost_tracker.py remove unused code
