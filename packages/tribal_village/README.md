# Tribal Village Environment

Multi-agent RL environment built in Nim with PufferLib integration. Features 60 agents across 12 teams competing for resources while fighting off hostile tumors.

<img width="2742" height="1628" alt="image" src="https://github.com/user-attachments/assets/a5992e9d-abdd-4d8b-ab83-efabd90e2bd5" />

## Quick Start

**Setup**
```bash
# Install Nim and Nimble
curl https://nim-lang.org/choosenim/init.sh -sSf | sh
# Install dependencies
nimble install
```

**Standalone Game**
```bash
nim r -d:release tribal_village.nim
```

**PufferLib Training**
```bash
nimble buildLib  # Builds in danger mode for maximum performance
python -c "from tribal_village_env import TribalVillageEnv; env = TribalVillageEnv()"
```

## Configuration

The Python environment accepts a config dictionary to customize the Nim simulation:

```python
config = {
    'max_steps': 1000,          # Episode length
    'ore_per_battery': 1,       # Ore needed to craft battery
    'batteries_per_heart': 1,   # Batteries needed for heart at altar
    'enable_combat': True,      # Enable tumor spawning and combat
    'tumor_spawn_rate': 0.1,   # Tumor spawn frequency (lower = slower spawns)
    'tumor_damage': 1,         # Damage tumors deal to agents
    'heart_reward': 1.0,        # Reward for heart crafting
    'ore_reward': 0.1,          # Reward for mining ore
    'battery_reward': 0.8,      # Reward for crafting batteries
    'wood_reward': 0.0,         # Reward for chopping wood
    'water_reward': 0.0,        # Reward for collecting water
    'wheat_reward': 0.0,        # Reward for harvesting wheat
    'spear_reward': 0.0,        # Reward for crafting spears
    'armor_reward': 0.0,        # Reward for crafting armor
    'food_reward': 0.0,         # Reward for crafting bread
    'cloth_reward': 0.0,        # Reward for crafting lanterns
    'tumor_kill_reward': 0.0,  # Reward for clearing tumors
    'survival_penalty': 0.0,    # Penalty per step (negative)
    'death_penalty': 0.0        # Penalty for agent death (negative)
}
env = TribalVillageEnv(config=config)
```

## Game Overview

**Map**: 192x108 grid with procedural terrain (rivers, wheat fields, tree groves)
**Agents**: 60 agents in 12 teams of 5, each with specialized AI roles
**Resources**: ore, batteries, water, wheat, wood, spear, lantern, armor, bread
**Threats**: Autonomous tumors that spawn and expand across the map

### Core Gameplay Loop
1. **Gather** resources (mine ore, harvest wheat, chop wood, collect water)
2. **Craft** items using specialized buildings (forge spears, weave lanterns, etc.)
3. **Cooperate** within teams and compete across teams
4. **Defend** against tumors using crafted spears

## Controls

**Agent Selection**: Click agents to view inventory overlay in top-left
**Movement**: Arrow keys/WASD for cardinal, QEZC for diagonal
**Actions**: U (use/craft), P (special action)
**Global**: Space (pause), +/- (speed), Mouse (pan/zoom)

## Technical Details

### Observation Space
21 layers, 11x11 grid per agent:
- **Layer 0**: Team-aware agent presence (1=team0, 2=team1, 3=team2, 255=Tumor)
- **Layers 1-9**: Agent orientation + inventories (ore, battery, water, wheat, wood, spear, lantern, armor)
- **Layers 10-18**: Buildings (walls, mines, converters, altars) + status
- **Layers 19-20**: Environmental effects + bread inventory

### Action Space
Multi-discrete `[move_direction, action_type]`:
- **Movement**: 8 directions (N/S/E/W + diagonals)
- **Actions**: Move, attack, use/craft, give items, plant lanterns

### Architecture
- **Nim backend**: High-performance simulation and rendering
- **Python wrapper**: PufferLib-compatible interface for all 60 agents
- **Zero-copy communication**: Direct pointer passing for efficiency
- **Web ready**: Emscripten support for WASM deployment

## Build

- Native shared library for Python: `nimble buildLib`
- Native desktop viewer: `nimble run`
- WebAssembly demo (requires Emscripten on PATH): `nimble wasm`
  - Outputs to `build/web/tribal_village.html`; serve via `python -m http.server 8000`

### PufferLib Rendering

- Python bindings default to `render_mode="rgb_array"` and stream full-map RGB frames via Nim.
- Adjust `render_scale` in the env config (default 4) to control output resolution.
- Set `render_mode="ansi"` for lightweight terminal output.

## Files

**Core**: `tribal_village.nim` (main), `src/environment.nim` (simulation), `src/ai.nim` (built-in agents)
**Rendering**: `src/renderer.nim`, `src/ui.nim`, `src/controls.nim`
**Integration**: `src/tribal_village_interface.nim` (C interface), `tribal_village_env/` (Python wrapper)
**Build**: `build_lib.sh`, `tribal_village.nimble`

## Dependencies

**Nim**: 2.2.4+ with boxy, windy, vmath, chroma packages
**Python**: 3.8+ with gymnasium, numpy, pufferlib
**System**: OpenGL for rendering
