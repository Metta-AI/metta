# Scripted Agent Policies

Three baseline scripted agent implementations for CoGames evaluation and ablation studies.

## Overview

This package provides three progressively capable scripted agents:

1. **SimpleBaselineAgent** - Core functionality: exploration, resource gathering, heart assembly
2. **UnclippingAgent** - Extends SimpleBaselineAgent with extractor unclipping capability
3. **CoordinatingAgent** - Extends UnclippingAgent with multi-agent coordination (has all capabilities)

## Architecture

### File Structure

```
scripted_agent/
├── simple_baseline_agent.py    # Base agent + SimpleBaselinePolicy wrapper
├── unclipping_agent.py          # Unclipping extension + UnclippingPolicy wrapper
├── coordinating_agent.py        # Coordination extension + CoordinatingPolicy wrapper
├── navigator.py                 # Pathfinding utilities (shared)
└── README.md                    # This file
```

Each agent file contains:
- Agent class with core logic and state management
- Policy wrapper classes at the bottom for CLI integration

### Design Philosophy

These agents are designed for **ablation studies** and **baseline evaluation**:
- Simple, readable implementations
- No hyperparameter tuning
- Clear separation of capabilities
- Minimal dependencies

## Agents

### 1. SimpleBaselineAgent

**Purpose**: Minimal working agent for single-agent missions

**Capabilities**:
- ✅ Visual discovery (explores to find stations and extractors)
- ✅ Resource gathering (navigates to extractors, handles cooldowns)
- ✅ Heart assembly (deposits resources at assembler)
- ✅ Heart delivery (brings hearts to chest)
- ✅ Energy management (recharges when low)
- ✅ Extractor tracking (remembers positions, cooldowns, remaining uses)

**Limitations**:
- ❌ No unclipping support (can't handle clipped extractors)
- ❌ No multi-agent coordination (will collide with other agents)
- ⚠️ Single-agent only (for multi-agent, use CoordinatingAgent)

**Usage**:
```python
from cogames.policy.scripted_agent import SimpleBaselinePolicy
from mettagrid import MettaGridEnv

env = MettaGridEnv(env_config)
policy = SimpleBaselinePolicy(env)

obs, info = env.reset()
policy.reset(obs, info)

agent = policy.agent_policy(0)
action = agent.step(obs[0])
```

**CLI**:
```bash
uv run cogames play --mission evals.extractor_hub_30 -p simple_baseline --cogs 1
```

### 2. UnclippingAgent

**Purpose**: Handle missions with clipped extractors

**Extends SimpleBaselineAgent with**:
- ✅ Clipped extractor detection
- ✅ Unclip item crafting
- ✅ Extractor restoration
- ✅ Resource deficit management (ensures enough resources for both unclipping and hearts)

**Unclip Item Mapping**:
| Clipped Resource | Unclip Item | Crafted From | Glyph |
|-----------------|-------------|--------------|-------|
| Oxygen | decoder | carbon | gear |
| Carbon | modulator | oxygen | gear |
| Germanium | resonator | silicon | gear |
| Silicon | scrambler | germanium | gear |

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
from cogames.policy.scripted_agent import UnclippingPolicy

policy = UnclippingPolicy(env)
# ... same as SimpleBaselinePolicy
```

**CLI**:
```bash
# Test with clipped oxygen
uv run cogames play --mission evals.extractor_hub_30 -p unclipping --variant clipped_oxygen --cogs 1
```

### 3. CoordinatingAgent

**Purpose**: Multi-agent coordination with full capabilities

**Extends UnclippingAgent with**:
- ✅ Core gathering/assembly (from SimpleBaselineAgent)
- ✅ Unclipping capability (from UnclippingAgent)
- ✅ Smart mouth selection (agents spread around stations)
- ✅ Free mouth detection (avoids occupied spots)
- ✅ Commitment to selected mouths (prevents oscillation)

**Coordination Strategy**:
- When within 2 cells of assembler or extractor, picks a specific "mouth" (adjacent cell)
- Checks observations for other agents at potential mouths
- Commits to chosen mouth to avoid flip-flopping
- Agents naturally distribute around stations

**Usage**:
```python
from cogames.policy.scripted_agent import CoordinatingPolicy

policy = CoordinatingPolicy(env)
# ... same as above
```

**CLI**:
```bash
# Test with multiple agents
uv run cogames play --mission evals.extractor_hub_30 -p coordinating --cogs 4
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

#### SimpleBaselineAgent (Non-Clipping)
```bash
# Default difficulty
uv run cogames play --mission evals.extractor_hub_30 -p simple_baseline --cogs 1 --steps 1000

# Story mode (easy)
uv run cogames play --mission evals.extractor_hub_30 -p simple_baseline --variant story_mode --cogs 1 --steps 1000

# Standard difficulty
uv run cogames play --mission evals.extractor_hub_30 -p simple_baseline --variant standard --cogs 1 --steps 1500

# Hard difficulty
uv run cogames play --mission evals.extractor_hub_30 -p simple_baseline --variant hard --cogs 1 --steps 2000

# Brutal difficulty
uv run cogames play --mission evals.extractor_hub_30 -p simple_baseline --variant brutal --cogs 1 --steps 3000

# Different mission
uv run cogames play --mission evals.oxygen_bottleneck -p simple_baseline --cogs 1 --steps 1000
```

#### UnclippingAgent (Clipping Variants)
```bash
# Clipped oxygen
uv run cogames play --mission evals.extractor_hub_30 -p unclipping --variant clipped_oxygen --cogs 1 --steps 2000

# Clipped carbon
uv run cogames play --mission evals.extractor_hub_30 -p unclipping --variant clipped_carbon --cogs 1 --steps 2000

# Clipped germanium
uv run cogames play --mission evals.extractor_hub_30 -p unclipping --variant clipped_germanium --cogs 1 --steps 2000

# Clipped silicon
uv run cogames play --mission evals.extractor_hub_30 -p unclipping --variant clipped_silicon --cogs 1 --steps 2000

# Hard + clipped oxygen (combined difficulty)
uv run cogames play --mission evals.extractor_hub_30 -p unclipping --variant hard_clipped_oxygen --cogs 1 --steps 3000

# Clipping chaos (random clipping)
uv run cogames play --mission evals.extractor_hub_30 -p unclipping --variant clipping_chaos --cogs 1 --steps 2000
```

#### CoordinatingAgent (Multi-Agent)
```bash
# 2 agents
uv run cogames play --mission evals.extractor_hub_30 -p coordinating --cogs 2 --steps 1500

# 4 agents
uv run cogames play --mission evals.extractor_hub_30 -p coordinating --cogs 4 --steps 2000

# 8 agents
uv run cogames play --mission evals.extractor_hub_30 -p coordinating --cogs 8 --steps 2500

# With difficulty variant
uv run cogames play --mission evals.extractor_hub_30 -p coordinating --variant hard --cogs 4 --steps 2500

# With clipping variant (CoordinatingAgent has unclipping capability!)
uv run cogames play --mission evals.extractor_hub_30 -p coordinating --variant clipped_oxygen --cogs 2 --steps 2000

# Hard + clipping + coordination
uv run cogames play --mission evals.extractor_hub_30 -p coordinating --variant hard_clipped_oxygen --cogs 4 --steps 3000
```

### Comprehensive Evaluation
```bash
# Run full evaluation suite
uv run python -u packages/cogames/scripts/evaluate_scripted_agent.py full

# Training facility only
uv run python -u packages/cogames/scripts/evaluate_scripted_agent.py training-facility
```

## Evaluation Results

Performance benchmarks across difficulty variants:

### SimpleBaselineAgent (Non-Clipping Missions)
| Mission | Story Mode | Standard | Hard | Brutal |
|---------|------------|----------|------|--------|
| extractor_hub_30 | TBD | TBD | TBD | TBD |
| oxygen_bottleneck | TBD | TBD | TBD | TBD |
| silicon_limited | TBD | TBD | TBD | TBD |

### UnclippingAgent (Clipping Missions)
| Mission | Clipped Oxygen | Clipped Carbon | Clipped Germanium | Clipped Silicon |
|---------|----------------|----------------|-------------------|-----------------|
| extractor_hub_30 | 11-13 hearts | TBD | TBD | TBD |
| oxygen_bottleneck | TBD | TBD | TBD | TBD |

### CoordinatingAgent (Multi-Agent)
| Mission | 1 COG | 2 COGs | 4 COGs | 8 COGs |
|---------|-------|--------|--------|--------|
| extractor_hub_30 | TBD | TBD | TBD | TBD |
| oxygen_bottleneck | TBD | TBD | TBD | TBD |

*TBD: To be determined through comprehensive evaluation*

## Extending

### Adding New Agent Capabilities

To create a new agent variant:

1. **Create new file** (e.g., `my_agent.py`)
2. **Extend base class**:
```python
from .simple_baseline_agent import SimpleBaselineAgent, SimpleAgentState

class MyAgent(SimpleBaselineAgent):
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
    def __init__(self, env=None, device=None):
        self._env = env
        self._impl = MyAgent(env) if env is not None else None
        self._agent_policies = {}

    def reset(self, obs, info):
        # Initialize impl from info if needed
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

5. **Add to CLI** in `cogames/policy/utils.py`:
```python
_POLICY_CLASS_SHORTHAND = {
    ...
    "my_agent": "cogames.policy.scripted_agent.my_agent.MyPolicy",
}
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
- [ ] Advanced coordination (task assignment)
- [ ] Frontier-based exploration
