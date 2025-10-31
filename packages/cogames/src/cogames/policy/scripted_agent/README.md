# Scripted Agent Policy

A sophisticated scripted agent implementation for CoGames that uses visual discovery, phase-based control, and adaptive strategies.

## Architecture

### Core Components

- **`agent.py`**: Main policy implementation with state management and action selection
- **`phase_controller.py`**: Finite state machine for managing agent phases and transitions
- **`navigator.py`**: Pathfinding utilities (BFS, A*, greedy fallback)
- **`hyperparameters.py`**: Hyperparameter dataclass definition
- **`hyperparameter_presets.py`**: Curated presets for different strategies

### Key Features

1. **Visual Discovery**: Agent discovers stations and extractors through observations (no global knowledge)
2. **Phase-Based Control**: FSM with phases like GATHER_GERMANIUM, ASSEMBLE_HEART, RECHARGE, etc.
3. **Extractor Memory**: Tracks discovered extractors, cooldowns, depletion, and efficiency
4. **Adaptive Exploration**: Exploration duration scales with map size
5. **Energy Management**: Dynamic recharge thresholds based on map size
6. **Clipping Support**: Handles clipped extractors by crafting unclip items

## Usage

```python
from cogames.policy.scripted_agent import ScriptedAgentPolicy, HYPERPARAMETER_PRESETS
from mettagrid import MettaGridEnv

# Create environment
env = MettaGridEnv(env_config)

# Create policy with preset
policy = ScriptedAgentPolicy(env, hyperparams=HYPERPARAMETER_PRESETS['balanced'])

# Run episode
obs, info = env.reset()
policy.reset(obs, info)
agent_policy = policy.agent_policy(0)

for step in range(max_steps):
    action = agent_policy.step(obs[0])
    obs, rewards, dones, truncated, info = env.step([action])
    if dones[0]:
        break
```

## Hyperparameter Presets

The agent includes 10 curated presets optimized for different scenarios:

| Preset | Strategy | Best For |
|--------|----------|----------|
| `explorer_short` | Early exploration (50 steps) | Small maps, quick missions |
| `explorer_long` | Extended exploration (200 steps) | Large maps, complex layouts |
| `greedy_conservative` | Minimal exploration, early recharge | Energy-constrained missions |
| `greedy_aggressive` | Minimal exploration, late recharge | Resource-rich missions |
| `balanced` | Moderate exploration & recharge | General purpose |
| `efficiency_learner` | Tracks extractor efficiency | Multi-extractor missions |
| `sequential_simple` | Fixed gathering order | Predictable missions |
| `patient_waiter` | Waits for cooldowns | High-cooldown extractors |
| `quick_rotator` | Rotates between extractors | Low-cooldown extractors |
| `depletion_aware` | Avoids low-resource extractors | Limited-use missions |

## Phase System

The agent operates in distinct phases managed by a finite state machine:

### Resource Gathering Phases
- `GATHER_GERMANIUM`: Collect germanium (5 required)
- `GATHER_SILICON`: Collect silicon (50 required)
- `GATHER_CARBON`: Collect carbon (20 required)
- `GATHER_OXYGEN`: Collect oxygen (20 required)

### Production Phases
- `ASSEMBLE_HEART`: Craft heart at assembler
- `DEPOSIT_HEART`: Deposit heart at chest

### Support Phases
- `RECHARGE`: Restore energy at charger
- `EXPLORE`: Discover new areas and stations

### Clipping Phases
- `CRAFT_DECODER`: Craft unclip item (decoder/modulator/resonator/scrambler)
- `UNCLIP_STATION`: Use unclip item to access clipped extractor

## Clipping Support

The agent handles clipped extractors through a multi-step process:

1. **Detection**: Observes `clipped` feature on extractors
2. **Alternative Resource**: Gathers the correct resource for crafting
3. **Craft Unclip Item**: Creates the appropriate unclip item
4. **Unclip**: Uses item to make extractor available
5. **Resume**: Continues gathering the originally clipped resource

### Unclip Item Mapping

| Clipped Resource | Unclip Item | Crafted From |
|-----------------|-------------|--------------|
| Oxygen | decoder | carbon |
| Carbon | modulator | oxygen |
| Germanium | resonator | silicon |
| Silicon | scrambler | germanium |

## Extractor Memory

The agent maintains detailed information about discovered extractors:

```python
@dataclass
class ExtractorInfo:
    position: Tuple[int, int]
    resource_type: str
    station_name: str
    last_used_step: int
    total_harvests: int
    total_output: int
    uses_remaining_fraction: float
    observed_cooldown_remaining: int
    observed_converting: bool
    is_clipped: bool
    permanently_depleted: bool
    learned_cooldown: Optional[int]
```

## Navigation

The navigator provides multiple pathfinding strategies:

- **BFS**: Breadth-first search for guaranteed shortest path
- **A***: Heuristic search for faster pathfinding on large maps
- **Greedy Fallback**: Direct movement when path blocked
- **Optimistic Mode**: Treats unknown cells as passable

## Customization

### Creating Custom Hyperparameters

```python
from cogames.policy.scripted_agent import Hyperparameters

custom_params = Hyperparameters(
    strategy_type="explorer_first",
    exploration_phase_steps=150,
    min_energy_for_silicon=60,
    recharge_start_small=70,
    recharge_stop_small=95,
    wait_if_cooldown_leq=5,
    depletion_threshold=0.3,
)

policy = ScriptedAgentPolicy(env, hyperparams=custom_params)
```

### Adding New Presets

Edit `hyperparameter_presets.py`:

```python
MY_PRESET = Hyperparameters(
    strategy_type="explorer_first",
    # ... your parameters
)

HYPERPARAMETER_PRESETS["my_preset"] = MY_PRESET
```

## Testing

Run tests for the scripted agent:

```bash
uv run pytest packages/cogames/tests/scripted_agent/
```

## Performance

The scripted agent achieves:
- **100% success** on basic training facility missions
- **90%+ success** on medium difficulty missions
- **Handles clipping** with 100% success on clipped missions
- **Adaptive to map size** through dynamic exploration and recharge

## Future Improvements

- [ ] Multi-agent coordination
- [ ] Dynamic recipe detection from environment
- [ ] Clip spreading support (clip_spread_rate > 0)
- [] Charger clipping strategies
- [ ] Learned extractor efficiency persistence
- [ ] Frontier exploration optimization

