# LLM Policy for MettaGrid - Task Tracker

## Project Context

**Goal:** Create an LLM-based policy to play MettaGrid that conforms to the AgentPolicy API, enabling GPT, Claude, and local LLMs (via Ollama) to control agents.

**Motivation:** First step toward fine-tuning a LORA using observation/action pairs collected from LLM gameplay.

**Branch:** `lschiavini/llm-policy`

**PR:** https://github.com/Metta-AI/metta/pull/4090

---

## Completed Work

### Phase 1: Core Infrastructure ✅

- [x] Implement `LLMAgentPolicy` conforming to `AgentPolicy` API
- [x] Implement `LLMMultiAgentPolicy` for multi-agent support
- [x] Add OpenAI GPT provider support (`llm-openai`)
- [x] Add Anthropic Claude provider support (`llm-anthropic`)
- [x] Add Ollama local LLM support (`llm-ollama`)
- [x] Model selection UI when model not specified
- [x] API key validation with helpful error messages
- [x] Cost tracking for paid APIs (OpenAI, Anthropic)
- [x] Cost summary printed on exit

### Phase 2: Prompt Engineering ✅

- [x] Dynamic prompt system with context window management
- [x] `context_window_size` parameter - resend basic info every N steps
- [x] Prompt templates in markdown files (`prompts/*.md`)
- [x] Basic info prompt with game rules, recipes, object types
- [x] Observable prompt with inventory, adjacent tiles, visible objects
- [x] Strategic hints based on current state (energy warnings, resource needs)
- [x] Directional awareness (blocked vs clear tiles)

### Phase 3: Memory Systems ✅

- [x] `summary_interval` - generates history summaries every N steps
- [x] `debug_summary_interval` - writes LLM debug summaries to file for long runs
- [x] Global position tracking from origin
- [x] Persistent extractor memory with visit counts and collection stats
- [x] Other agents' position and inventory tracking
- [x] Exploration direction tracking with direction-change suggestions

### Phase 4: Testing & Tuning 🔄

- [x] Qwen baseline tests (local, free)
- [x] Claude baseline tests (paid)
- [x] Claude extended runs with hearts crafted
- [x] Debug summary feature verification
- [ ] Pathfinding hints (C5 - future)

---

## Key Results

| Test | Provider | Agents | Steps | Reward | Notes |
|------|----------|--------|-------|--------|-------|
| Qwen T1 | Ollama | 1 | 200 | 0.17 | Only found germanium, loops |
| Qwen T4 | Ollama | 2 | 1000 | 0.00 | Explored 231 tiles, no hearts |
| Claude C1 | Anthropic | 2 | 100 | 1.30 | Found 3/4 extractors |
| **Claude C2** | **Anthropic** | **2** | **200** | **12.87** | **SUCCESS - Hearts crafted!** |
| Claude C3 | Anthropic | 2 | 200 | 1.12 | Memory works, wall navigation issue |

---

## Current Bottleneck

**Pathfinding:** Agents can see and remember extractors but get stuck navigating around walls. The memory features work correctly, but agents can't efficiently path to targets.

**Proposed Solution (C5):** Add pathfinding hints like "To reach silicon_extractor, go: E→E→N→E→E"

---

## Files Modified

**Main implementation:**
- `packages/mettagrid/python/src/mettagrid/policy/llm_policy.py` - Core policy classes
- `packages/mettagrid/python/src/mettagrid/policy/llm_prompt_builder.py` - Dynamic prompts
- `packages/mettagrid/python/src/mettagrid/policy/prompts/*.md` - Prompt templates

**Tests:**
- `packages/mettagrid/tests/policy/test_llm_prompt_builder.py`

---

## Usage

```bash
# Claude (recommended for best results)
uv run cogames play -m hello_world -c 2 -s 200 -p "class=llm-anthropic,kw.model=claude-sonnet-4-5,kw.context_window_size=20,kw.summary_interval=5"

# GPT
uv run cogames play -m hello_world -c 1 -s 100 -p "class=llm-openai,kw.model=gpt-4o"

# Local Ollama (free)
uv run cogames play -m hello_world -c 1 -s 100 -p "class=llm-ollama,kw.model=llama3.2"
```

---

## Current Testing

See [test_steps.md](./test_steps.md) for detailed test plan and results.

---

## Next Steps

1. **Decision point:** Continue prompt tuning (C5 pathfinding) or move to LORA fine-tuning phase?
2. If continuing prompts: Implement pathfinding hints to address wall navigation
3. If moving to LORA: Set up observation/action pair collection from Claude runs
