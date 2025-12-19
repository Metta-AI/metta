# LLM Agent

LLM-based agents for MettaGrid environments. Supports OpenAI GPT, Anthropic Claude, and local Ollama models.

## Usage

```bash
# OpenAI GPT
cogames play -m hello_world -p "class=llm-openai"

# Anthropic Claude
cogames play -m hello_world -p "class=llm-anthropic"

# Local Ollama
cogames play -m hello_world -p "class=llm-ollama"
```

## Configuration

Pass configuration via `kw.*` parameters:

```bash
cogames play -m hello_world -p "class=llm-anthropic,kw.model=claude-sonnet-4-5,kw.context_window_size=20,kw.summary_interval=5"
```

### Parameters

| Parameter             | Default          | Description                                                         |
| --------------------- | ---------------- | ------------------------------------------------------------------- |
| `model`               | Provider default | Model name (e.g., `gpt-4o-mini`, `claude-sonnet-4-5`, `qwen2.5:7b`) |
| `temperature`         | 0.7              | Sampling temperature                                                |
| `context_window_size` | 20               | Steps before resending basic game info                              |
| `summary_interval`    | 5                | Steps between history summaries                                     |
| `verbose`             | false            | Enable verbose output (shows prompts sent to LLM)                   |

## Environment Variables

- `OPENAI_API_KEY` - Required for OpenAI models
- `ANTHROPIC_API_KEY` - Required for Anthropic models
- Ollama models run locally (no API key needed)

## Features

- Dynamic prompt building with context window management
- BFS pathfinding hints for navigation
- Exploration history tracking
- Multi-agent coordination support
- Cost tracking for API usage
