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

**Purpose**: Minimal working agent for single/multi-agent missions

**Capabilities**:

- ✅ Visual discovery (explores to find stations and extractors)
- ✅ Resource gathering (navigates to extractors, handles cooldowns)
- ✅ Heart assembly (deposits resources at assembler)
- ✅ Heart delivery (brings hearts to chest)
- ✅ Energy management (recharges when low)
- ✅ Extractor tracking (remembers positions, cooldowns, remaining uses)
- ✅ Agent occupancy avoidance (multi-agent collision avoidance via pathfinding)

**Limitations**:

- ❌ No unclipping support (can't handle clipped extractors)
- ⚠️ Multi-agent coordination is basic (agents avoid each other but don't explicitly coordinate)

**Usage**:

```python
from cogames.policy.scripted_agent.baseline_agent import BaselinePolicy
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

**Purpose**: Handle missions with clipped extractors

**Extends BaselineAgent with**:

- ✅ Clipped extractor detection
- ✅ Unclip item crafting
- ✅ Extractor restoration
- ✅ Resource deficit management (ensures enough resources for both unclipping and hearts)

**Unclip Item Mapping**: | Clipped Resource | Unclip Item | Crafted From | Glyph |
|-----------------|-------------|--------------|-------| | Oxygen | decoder | carbon | gear | | Carbon | modulator |
oxygen | gear | | Germanium | resonator | silicon | gear | | Silicon | scrambler | germanium | gear |

**Workflow**:

1. Detects clipped extractor blocking progress
2. Gathers craft resource (e.g., carbon for decoder)
3. Changes glyph to "gear"
4. Crafts unclip item at assembler
5. Navigates to clipped extractor
6. Uses item to unclip
7. Resumes normal gathering

**Usage**:

```python
from cogames.policy.scripted_agent.unclipping_agent import UnclippingPolicy

policy = UnclippingPolicy(env)
# ... same as BaselinePolicy
```

**CLI**:

```bash
# Test with clipped oxygen (single agent)
uv run cogames play --mission evals.extractor_hub_30 -p scripted_unclipping --variant clipped_oxygen --cogs 1

# Test with clipped oxygen (multi-agent)
uv run cogames play --mission evals.extractor_hub_30 -p scripted_unclipping --variant clipped_oxygen --cogs 4
```

## Shared Components

### Phase System

All agents use a phase-based state machine:

```python
class Phase(Enum):
    GATHER = "gather"          # Collecting resources
    ASSEMBLE = "assemble"      # Crafting heart at assembler
    DELIVER = "deliver"        # Bringing heart to chest
    RECHARGE = "recharge"      # Restoring energy
    CRAFT_UNCLIP = "craft_unclip"  # UnclippingAgent only
    UNCLIP = "unclip"          # UnclippingAgent only
```

### Navigation

Shared `navigator.py` module provides:

- **BFS pathfinding** with occupancy grid
- **Greedy fallback** when path blocked
- **Adjacent positioning** for station interactions
- **Agent occupancy avoidance** for multi-agent scenarios

### Observation Parsing

Agents parse egocentric observations (11×11 grid) to detect:

- Stations (assembler, chest, charger, extractors)
- Other agents
- Walls and obstacles
- Agent state (resources, energy, inventory)

### Extractor Tracking

```python
@dataclass
class ExtractorInfo:
    position: tuple[int, int]
    resource_type: str  # "carbon", "oxygen", "germanium", "silicon"
    remaining_uses: int
    clipped: bool       # For UnclippingAgent
```

## Testing

### Quick Tests

#### BaselineAgent (Non-Clipping)

```bash
# Default difficulty (single agent)
uv run cogames play --mission evals.extractor_hub_30 -p scripted_baseline --cogs 1 --steps 1000

# Story mode (easy)
uv run cogames play --mission evals.extractor_hub_30 -p scripted_baseline --variant story_mode --cogs 1 --steps 1000

# Standard difficulty
uv run cogames play --mission evals.extractor_hub_30 -p scripted_baseline --variant standard --cogs 1 --steps 1500

# Hard difficulty
uv run cogames play --mission evals.extractor_hub_30 -p scripted_baseline --variant hard --cogs 1 --steps 2000

# Multi-agent (2, 4, 8 agents)
uv run cogames play --mission evals.extractor_hub_30 -p scripted_baseline --cogs 2 --steps 1500
uv run cogames play --mission evals.extractor_hub_30 -p scripted_baseline --cogs 4 --steps 2000
uv run cogames play --mission evals.extractor_hub_30 -p scripted_baseline --cogs 8 --steps 2500

# Different mission
uv run cogames play --mission evals.oxygen_bottleneck -p scripted_baseline --cogs 1 --steps 1000
```

#### UnclippingAgent (Clipping Variants)

```bash
# Clipped oxygen (single agent)
uv run cogames play --mission evals.extractor_hub_30 -p scripted_unclipping --variant clipped_oxygen --cogs 1 --steps 2000

# Clipped carbon
uv run cogames play --mission evals.extractor_hub_30 -p scripted_unclipping --variant clipped_carbon --cogs 1 --steps 2000

# Clipped germanium
uv run cogames play --mission evals.extractor_hub_30 -p scripted_unclipping --variant clipped_germanium --cogs 1 --steps 2000

# Clipped silicon
uv run cogames play --mission evals.extractor_hub_30 -p scripted_unclipping --variant clipped_silicon --cogs 1 --steps 2000

# Hard + clipped oxygen (combined difficulty)
uv run cogames play --mission evals.extractor_hub_30 -p scripted_unclipping --variant hard_clipped_oxygen --cogs 1 --steps 3000

# Clipping chaos (random clipping)
uv run cogames play --mission evals.extractor_hub_30 -p scripted_unclipping --variant clipping_chaos --cogs 1 --steps 2000

# Multi-agent with clipping
uv run cogames play --mission evals.extractor_hub_30 -p scripted_unclipping --variant clipped_oxygen --cogs 2 --steps 2000
uv run cogames play --mission evals.extractor_hub_30 -p scripted_unclipping --variant clipped_oxygen --cogs 4 --steps 2500
```

### Comprehensive Evaluation

```bash
# Run full evaluation suite
uv run python packages/cogames/scripts/run_evaluation.py

# Evaluate specific agent
uv run python packages/cogames/scripts/run_evaluation.py --agent simple
uv run python packages/cogames/scripts/run_evaluation.py --agent unclipping
```

## Evaluation Results

See `experiments/SCRIPTED_AGENT_EVALUATION.md` for comprehensive evaluation results across all missions and difficulty
variants.

**Summary**:

- **BaselineAgent**: 33.8% success rate across 1-8 agents, best for non-clipped missions
- **UnclippingAgent**: 38.6% success rate, best overall performance, handles clipping well

## Extending

### Adding New Agent Capabilities

To create a new agent variant:

1. **Create new file** (e.g., `my_agent.py`)
2. **Extend base class**:

```python
from .baseline_agent import BaselineAgent, SimpleAgentState

class MyAgent(BaselineAgent):
    def _update_phase(self, s: SimpleAgentState) -> None:
        # Add custom phase logic
        super()._update_phase(s)

    def _execute_phase(self, s: SimpleAgentState) -> int:
        # Add custom phase execution
        return super()._execute_phase(s)
```

3. **Add policy wrapper** at bottom of file:

```python
class MyAgentPolicy:
    """Per-agent policy wrapper."""
    def __init__(self, impl: MyAgent, agent_id: int):
        self._impl = impl
        self._agent_id = agent_id

    def step(self, obs) -> int:
        return self._impl.step(self._agent_id, obs)

class MyPolicy:
    """Policy wrapper for MyAgent."""
    def __init__(self, simulation=None):
        self._simulation = simulation
        self._impl = None
        self._agent_policies = {}

    def reset(self, obs, info):
        # Initialize impl from simulation
        pass

    def agent_policy(self, agent_id: int):
        # Return per-agent policy
        pass
```

4. **Register in `__init__.py`**:

```python
from cogames.policy.scripted_agent.my_agent import MyPolicy

__all__ = [..., "MyPolicy"]
```

### Resource Management

Agents track deficits and gather in priority order:

1. Germanium (5 needed, highest priority)
2. Silicon (50 needed)
3. Carbon (20 needed)
4. Oxygen (20 needed)

UnclippingAgent adds special logic:

- Ensures enough craft resource for both unclipping AND hearts
- Prevents resource deficits when crafting decoders

## Future Work

- [ ] Dynamic heart recipe detection
- [ ] Charger clipping strategies
- [ ] Clip spread handling
- [ ] Learned extractor efficiency
- [ ] Advanced multi-agent coordination (task assignment, resource reservation)
- [ ] Frontier-based exploration improvements
