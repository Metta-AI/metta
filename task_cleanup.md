# LLM Agent Code Cleanup Plan

## Overview

This document outlines the cleanup and reorganization needed to move LLM policy code from `packages/mettagrid/python/src/mettagrid/policy/` into a new dedicated package `packages/llm_agent/`.

## Current Location

The code currently lives in `packages/mettagrid/python/src/mettagrid/policy/`:

| File | Lines | Description |
|------|-------|-------------|
| `llm_policy.py` | 1910 | Main LLM policy classes |
| `llm_prompt_builder.py` | 894 | Dynamic prompt building |
| `observation_debugger.py` | 248 | Debug visualization |
| `prompts/*.md` | 5 files | Prompt templates |
| `test_llm_prompt_builder.py` | 598 | Tests |

## Policy Discovery Integration

**Important:** The policy system auto-discovers policies by scanning specific packages. Currently in `mettagrid/policy/loader.py:213`:

```python
def discover_and_register_policies(*packages: str) -> None:
    for package_name in ["mettagrid.policy", "metta.agent.policy", "cogames.policy", *packages]:
        _walk_and_import_package(package_name)
```

**Required change:** Add `"llm_agent.policy"` to this list so `cogames play -p llm-openai` continues to work.

---

## Clean Code Issues to Address

### 1. Single Responsibility Principle (SRP) Violations

**`llm_policy.py` mixes too many concerns:**

- Cost calculation logic (lines 20-62, 65-94)
- Model context validation (lines 102-202)
- Ollama/OpenAI/Anthropic model selection UI (lines 227-447)
- LLMAgentPolicy class (~1250 lines) combines:
  - API client management
  - Conversation history tracking
  - Position/exploration tracking
  - Strategic hint generation
  - Action parsing
  - Debug summary generation

### 2. Code Duplication

- `pos_to_dir()` function duplicated 4 times (lines 689, 894, 1084, 1119)
- Provider-specific API calls repeat similar patterns (lines 1385-1574)

### 3. Missing Package Infrastructure

- No `__init__.py` with proper exports
- No `pyproject.toml` for the package
- Dependencies not declared (openai, anthropic SDKs)
- No README for the package

---

## Recommended Package Structure

```
packages/llm_agent/
├── pyproject.toml
├── README.md
├── python/
│   └── src/
│       └── llm_agent/
│           ├── __init__.py
│           ├── policy/                   # Policy module (for discovery)
│           │   ├── __init__.py           # Exports all policy classes
│           │   ├── llm_policy.py         # Core LLMAgentPolicy, LLMMultiAgentPolicy
│           │   └── prompt_builder.py     # LLMPromptBuilder class
│           ├── prompts/                  # Markdown templates
│           │   ├── basic_info.md
│           │   ├── dynamic_prompt.md
│           │   ├── exploration_history.md
│           │   ├── full_prompt.md
│           │   └── pathfinding_hints.md
│           ├── providers/
│           │   ├── __init__.py
│           │   ├── base.py               # Abstract provider interface
│           │   ├── openai.py             # OpenAI-specific client
│           │   ├── anthropic.py          # Claude-specific client
│           │   └── ollama.py             # Ollama-specific client
│           ├── model_config.py           # Context windows, pricing, validation
│           ├── cost_tracker.py           # Cost tracking and summaries
│           ├── exploration_tracker.py    # Position, history, discovered objects
│           ├── action_parser.py          # Response parsing logic
│           └── observation_debugger.py   # Debug visualization
└── tests/
    ├── __init__.py
    ├── test_prompt_builder.py
    ├── test_action_parser.py
    └── test_cost_tracker.py
```

**Note:** The `policy/` subdirectory is required so `llm_agent.policy` can be added to the discovery list in `mettagrid/policy/loader.py`.

---

## Specific Refactoring Tasks

### Task 1: Create Package Infrastructure

- [ ] Create `packages/llm_agent/` directory structure
- [ ] Create `pyproject.toml` with dependencies (openai, anthropic)
- [ ] Create `README.md` with usage documentation
- [ ] Create `__init__.py` with public exports

### Task 2: Extract Model Configuration (`model_config.py`)

Move from `llm_policy.py`:
- `MODEL_CONTEXT_WINDOWS` dict
- `TOKENS_PER_STEP_BASE`, `TOKENS_PER_CONVERSATION_TURN` constants
- `get_model_context_window()` function
- `estimate_required_context()` function
- `validate_model_context()` function
- OpenAI/Anthropic pricing dicts from `calculate_llm_cost()`

### Task 3: Extract Cost Tracker (`cost_tracker.py`)

Create a `CostTracker` class to replace class-level state:
- `total_calls`, `total_input_tokens`, `total_output_tokens`, `total_cost`
- `calculate_llm_cost()` function
- `get_cost_summary()` method
- `_print_cost_summary_on_exit()` function

### Task 4: Create Provider Abstraction (`providers/`)

**`providers/base.py`:**
```python
class LLMProvider(ABC):
    @abstractmethod
    def chat(self, messages: list[dict], temperature: float, max_tokens: int) -> tuple[str, Usage]

    @abstractmethod
    def select_model(self) -> str
```

**`providers/openai.py`:**
- `OpenAIProvider` class
- `get_openai_models()` function
- `select_openai_model()` function

**`providers/anthropic.py`:**
- `AnthropicProvider` class
- `get_anthropic_models()` function
- `select_anthropic_model()` function

**`providers/ollama.py`:**
- `OllamaProvider` class
- `check_ollama_available()` function
- `list_ollama_models()` function
- `ensure_ollama_model()` function

### Task 5: Extract Exploration Tracker (`exploration_tracker.py`)

Move from `LLMAgentPolicy`:
- `_global_x`, `_global_y` position tracking
- `_current_window_positions`, `_all_visited_positions`
- `_discovered_objects` dict
- `_extractor_stats` dict
- `_other_agents_info` dict
- `_current_direction`, `_steps_in_direction`
- `_extract_discovered_objects()` method
- `_extract_other_agents_info()` method
- `_update_extractor_collection()` method
- `_get_discovered_objects_text()` method
- `_get_other_agents_text()` method
- Consolidate duplicated `pos_to_dir()` into single utility

### Task 6: Extract Action Parser (`action_parser.py`)

Move from `LLMAgentPolicy`:
- `_parse_action()` method -> `ActionParser.parse()` class

### Task 7: Extract Strategic Hints

Move from `LLMAgentPolicy`:
- `_get_strategic_hints()` method
- `_get_heart_recipe()` method
- `_get_visible_extractors()` method

### Task 8: Clean Up Core Policy (`policy.py`)

After extractions, `LLMAgentPolicy` should only contain:
- Provider initialization
- `step()` method (simplified)
- Conversation history management
- `_add_to_messages()`, `_get_messages_for_api()`

### Task 9: Update Policy Discovery

**Critical for `cogames play -p llm-openai` to work!**

In `packages/mettagrid/python/src/mettagrid/policy/loader.py`, update:

```python
# Before
for package_name in ["mettagrid.policy", "metta.agent.policy", "cogames.policy", *packages]:

# After
for package_name in ["mettagrid.policy", "metta.agent.policy", "cogames.policy", "llm_agent.policy", *packages]:
```

### Task 10: Update Imports

Update consumers in:
- `packages/cogames/src/cogames/cli/policy.py` (if any direct imports)
- Remove old files from `packages/mettagrid/python/src/mettagrid/policy/`:
  - `llm_policy.py`
  - `llm_prompt_builder.py`
  - `observation_debugger.py`
  - `prompts/` directory

### Task 11: Move and Update Tests

- Move `test_llm_prompt_builder.py` to new location
- Add tests for new extracted modules
- Update import paths

---

## Implementation Order

1. **Phase 1: Infrastructure** (Task 1)
   - Create package structure and build files
   - Set up `pyproject.toml` with dependencies

2. **Phase 2: Extract Utilities** (Tasks 2, 3, 6)
   - Model config, cost tracker, action parser
   - These have no dependencies on each other

3. **Phase 3: Provider Abstraction** (Task 4)
   - Requires model_config
   - Can be tested independently

4. **Phase 4: Exploration Tracker** (Task 5)
   - Extract position/discovery tracking

5. **Phase 5: Core Policy Cleanup** (Tasks 7, 8)
   - Simplify main policy class
   - Wire up extracted components

6. **Phase 6: Integration** (Tasks 9, 10, 11)
   - **Update policy discovery in `loader.py`** (critical!)
   - Remove old files from mettagrid
   - Move and update tests
   - Verify `cogames play -p llm-openai` works

---

## Success Criteria

### Code Quality
- [ ] All LLM agent code lives in `packages/llm_agent/`
- [ ] `llm_agent.policy` added to discovery list in `loader.py`
- [ ] Old files removed from `packages/mettagrid/python/src/mettagrid/policy/`
- [ ] No file exceeds 300 lines (except maybe policy.py at ~500)
- [ ] Each module has single responsibility
- [ ] No duplicated code
- [ ] All unit tests pass

### Package Installation
- [ ] Package can be installed independently
- [ ] `uv pip install -e packages/llm_agent` works

### Functional Tests (from test_steps.md)

Run these to verify refactoring didn't break functionality:

**Test 1: Ollama basic (smoke test)**
```bash
uv run cogames play -m hello_world -c 1 -s 10 -p "class=llm-ollama,kw.model=qwen2.5:7b" --render none
```
- [ ] Agent produces valid actions (no crashes)
- [ ] Policy discovery finds `llm-ollama`

**Test 2: Claude hello_world (100 steps)**
```bash
uv run cogames play -m hello_world -c 2 -s 100 -p "class=llm-anthropic,kw.model=claude-sonnet-4-5,kw.context_window_size=20,kw.summary_interval=5" --render none
```
- [ ] Both agents produce valid actions
- [ ] Reward > 0 (some resources collected)
- [ ] Baseline: C1 achieved reward 1.30

**Test 3: OpenAI hello_world (100 steps)**
```bash
uv run cogames play -m hello_world -c 2 -s 100 -p "class=llm-openai,kw.model=gpt-4o-mini,kw.context_window_size=20,kw.summary_interval=5" --render none
```
- [ ] Both agents produce valid actions
- [ ] Reward > 0

**Test 4: Full verification (300 steps, matches C6)**
```bash
uv run cogames play -m hello_world -c 2 -s 300 -p "class=llm-anthropic,kw.model=claude-sonnet-4-5,kw.context_window_size=20,kw.summary_interval=5" --render none
```
- [ ] Reward > 10 (C6 achieved 33.84)
- [ ] Hearts are crafted and deposited

### Feature Verification
- [ ] Pathfinding hints appear in prompts ("=== PATHFINDING HINTS ===")
- [ ] Exploration history summaries appear ("[HISTORY Agent 0]")
- [ ] Cost tracking prints at end ("LLM API USAGE SUMMARY")
- [ ] `kw.debug_summary_interval=100` writes to `llm_debug_summary_agent0.log`
