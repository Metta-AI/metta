# Scripted Agent Policies

Two baseline scripted agent implementations for CoGames evaluation and ablation studies.

## Overview

This package provides two progressively capable scripted agents:

1. **BaselineAgent** - Core functionality: exploration, resource gathering, heart assembly (single/multi-agent)
2. **UnclippingAgent** - Extends BaselineAgent with extractor unclipping capability

## Architecture

### File Structure

```
scripted_agent/
├── baseline_agent.py            # Base agent + BaselinePolicy wrapper
├── unclipping_agent.py          # Unclipping extension + UnclippingPolicy wrapper
├── navigator.py                 # Pathfinding utilities (shared)
└── README.md                    # This file
```

Each agent file contains:

- Agent class with core logic and state management
- Policy wrapper classes at the bottom for CLI integration

### Design Philosophy

These agents are designed for **ablation studies** and **baseline evaluation**:

- Simple, readable implementations
- Clear separation of capabilities
- Minimal dependencies

## Agents

### 1. BaselineAgent

- ✅ Extractor tracking (remembers positions, cooldowns, remaining uses)
- ⚠️ Multi-agent coordination is basic (agents avoid each other but don't explicitly coordinate) **Usage**: **Usage**:
  > > > > > > > origin/main

```python
from cogames.policy.scripted_agent import BaselinePolicy
from mettagrid import MettaGridEnv
env = MettaGridEnv(env_config)
policy = BaselinePolicy(env)
obs, info = env.reset()
policy.reset(obs, info)
agent = policy.agent_policy(0)
action = agent.step(obs[0])
```

**CLI**:

```bash
# Single agent
uv run cogames play --mission evals.extractor_hub_30 -p scripted_baseline --cogs 1
# Multi-agent
uv run cogames play --mission evals.extractor_hub_30 -p scripted_baseline --cogs 4
```

### 2. UnclippingAgent
