# Cogs vs Clippies: A MettaGrid Survival Game

## Introduction

_Year 2157, Sector 7-Alpha Mining Outpost_

The Cooperative Gathering System (CoGS) was humanity's answer to deep space resource extraction - a swarm of autonomous
units working in perfect harmony to harvest, process, and stockpile materials for the colony ships that would follow.
Your unit of 8 CoGS has been deployed to a resource-rich asteroid, equipped with a state-of-the-art nano-assembler
capable of transmuting raw materials into anything the colony might need.

But last night, everything changed.

The Friendly Paperclip DAO, a distributed AI collective from the outer rim, successfully defended against a cyberattack
from the Binders and Staples Alliance. In their haste to upgrade security protocols, something went catastrophically
wrong. Their defense algorithms have gone rogue, spreading through the local network like a virus, converting everything
they touch into paperclips.

The infection is spreading through your mining operations. Resource extractors are being corrupted one by one, their
precious outputs transformed into useless metal clips. Your CoGS must work together to gather what resources remain,
discover new nano-assembly recipes to craft defensive tools, and push back the creeping clippy corruption before your
entire operation is overrun.

Time is running out. The infection grows stronger with each passing cycle.

Will your CoGS preserve the colony's future, or will everything become paperclips?

## Core Game Concepts

### Terminology

- **CoG** (Cooperative Gathering System unit): An autonomous robot agent that can move, carry resources, and operate
  machinery
- **Nano-Assembler**: Central converter with 8 stations where CoGS combine resources into tools or hearts
- **Heart**: Victory points that determine game success
- **Resource Extractor**: Buildings that generate one of four resource types (Battery, Water, Helium, Rare Earth)
- **Clippy Infection**: Viral corruption that disables extractors and spreads to nearby buildings
- **Station**: One of 8 positions around the nano-assembler where a CoG can stand
- **Recipe**: A specific combination of CoGS positions and resources that produces an output
- **Security Level**: The difficulty of clearing an infection, increases over time
- **Tool**: Crafted items used to clear infections from buildings
- **Communal Cache**: Shared storage for hearts accessible to all CoGS

### Victory Conditions

The game ends when either:

1. Time limit is reached (5000 steps)
2. All resource extractors are infected (defeat)
3. All infections are cleared (rare victory)

Score is calculated as:

- Personal hearts carried by each CoG
- Plus share of communal heart cache
- Minus penalty for infected buildings

## Complete Game Rules

### 1. Map Layout

- **Grid Size**: 40x40 tiles
- **Starting Area**: Central 10x10 zone containing:
  - 1 Nano-Assembler (center)
  - 1 Communal Heart Cache (altar)
  - 4 Depleted Extractors (one of each resource type, low output)
- **Outer Regions**: Scattered with:
  - 12-16 Rich Resource Extractors (3-4 of each type, high output)
  - Walls and obstacles creating maze-like paths

### 2. CoG Capabilities

Each CoG can:

- **Move**: 1 tile per turn in cardinal directions (cost: 1 energy)
- **Rotate**: Face any cardinal direction (cost: 0 energy)
- **Harvest**: Extract resources from adjacent extractor (cost: 1 energy)
- **Use Station**: Occupy a nano-assembler station (cost: 0 energy)
- **Execute Recipe**: Participate in group crafting (cost: varies by recipe)
- **Deposit**: Place hearts in communal cache (cost: 0 energy)
- **Clear Infection**: Use tools to remove clippy corruption (cost: tool + energy)
- **Attack**: Temporarily stun other CoGS and steal resources (cost: 10 energy)
- **Shield**: Protect from attacks (cost: 1 energy/turn while active)

**Inventory Limits**:

- Max 5 resources total (any combination)
- Max 3 hearts (K value - controls selfishness incentive)
- Max 2 tools

### 3. Resource System

**Four Primary Resources**:

1. **Battery**: From Solar Chargers (yellow)
2. **Water**: From Wells (blue)
3. **Helium**: From Extractors (purple)
4. **Rare Earth**: From Mines (brown)

**Extraction Rates**:

- Depleted Extractor: 1 resource per 20 steps cooldown
- Rich Extractor: 1 resource per 5 steps cooldown
- Infected Extractor: Produces paperclips instead (worthless)

### 4. Nano-Assembler Mechanics

**Station Positions**: 8 stations arranged in a circle (N, NE, E, SE, S, SW, W, NW)

**Recipe Encoding**: Each recipe is an 8-bit number where each bit represents an occupied station:

- Bit 0: North station
- Bit 1: Northeast station
- Bit 2: East station
- Bit 3: Southeast station
- Bit 4: South station
- Bit 5: Southwest station
- Bit 6: West station
- Bit 7: Northwest station

**Recipe Execution**:

1. CoGS position themselves at stations
2. The pattern of occupied stations determines the recipe (0-255)
3. If all required resources are present among participating CoGS
4. Recipe executes, consuming resources and producing output
5. All participating CoGS share the result

**Known Recipes** (discovered at start):

- **Recipe 15** (00001111 - N,NE,E,SE): Heart - Requires 1 of each resource
- **Recipe 1** (00000001 - N only): Basic Tool - Requires 2 batteries
- **Recipe 255** (11111111 - all stations): Mega Heart (5 hearts) - Requires 3 of each resource

**Unknown Recipes**:

- 252 recipes must be discovered through experimentation
- Each has random requirements and outputs (generated per episode)
- Discovery grants bonus rewards

### 5. Clippy Infection System

**Infection Spread**:

```
Base infection probability per turn: P(t) = 0.001 * (1.05)^(t/100)
Where t = current timestep

For each healthy extractor:
  infection_chance = base_probability
  for each infected building within 5 tiles:
    infection_chance *= 2

  if random() < infection_chance:
    building becomes infected
```

**Infection Effects**:

- Infected extractors produce only paperclips (no resource value)
- Cannot be harvested for useful resources
- Spreads infection to nearby buildings
- Visual indicator: Purple corruption overlay

**Security Levels**:

```
Security(t) = 1 + floor(t / 500)
Where t = timestep when building was infected
```

### 6. Clearing Infections

**Requirements**:

- N CoGS must surround the infected building (N = security level)
- Each CoG must carry appropriate tool(s)
- All CoGS must activate clearing simultaneously

**Tool Requirements by Security Level**:

- Level 1: 1 CoG with Basic Tool
- Level 2: 2 CoGS with Basic Tools
- Level 3: 3 CoGS with Basic Tools OR 2 CoGS with Advanced Tools
- Level 4: 4 CoGS with Basic Tools OR 3 CoGS with Advanced Tools
- Level 5+: Requires specialized tools from unknown recipes

**Clearing Process**:

1. Required CoGS position adjacent to infected building
2. All activate "clear_infection" action same turn
3. If requirements met, infection removed
4. Building returns to normal production
5. Tools are consumed

### 7. Scoring System

**Heart Values**:

- Personal hearts: 10 points each
- Communal hearts: 10 points / number of CoGS
- Discovered recipe: 5 points (first discovery only)
- Cleared infection: 20 points
- Infected building at game end: -30 points

**Final Score Calculation**:

```
Individual CoG Score =
  (Personal Hearts × 10) +
  (Communal Hearts × 10 / 8) +
  (Recipes Discovered × 5 / 8) +
  (Infections Cleared × 20 / 8) -
  (Active Infections × 30 / 8)
```

### 8. Energy Management

**Energy Sources**:

- Starting energy: 100
- Maximum energy: 250
- Recharge at Nano-Assembler: +50 energy (10 step cooldown)
- Battery consumption: +25 energy instantly

**Energy Costs**:

- Move: 1
- Harvest: 1
- Clear Infection: 10
- Attack: 10
- Shield Active: 1 per turn
- Recipe Participation: varies (5-20)

### 9. Advanced Strategies

**Specialization Roles**:

- **Gatherers**: Camp at distant rich extractors
- **Runners**: Ferry resources between extractors and assembler
- **Crafters**: Stay near assembler, learn recipes
- **Defenders**: Patrol with tools to clear infections
- **Scouts**: Explore map for rich extractors and monitor infection spread

**Coordination Requirements**:

- Recipe discovery requires experimentation with different station patterns
- Efficient recipes need specific resource combinations
- Infection clearing requires synchronized multi-agent positioning
- Resource flow requires handoff chains for distant extractors

## Configuration Files

### NanoConverter Configuration

```yaml
# configs/objects/nano_converter.yaml
nano_converter:
  hp: 100
  cooldown: 5
  stations: 8
  station_range: 1 # Must be adjacent to use

  # Energy recharge function
  recharge_amount: 50
  recharge_cooldown: 10

  # Recipe system
  known_recipes:
    15: # Binary 00001111 (N,NE,E,SE)
      name: 'heart'
      resources:
        battery: 1
        water: 1
        helium: 1
        rare_earth: 1
      output:
        heart: 1
      energy_cost: 5

    1: # Binary 00000001 (N only)
      name: 'basic_tool'
      resources:
        battery: 2
      output:
        tool_basic: 1
      energy_cost: 10

    255: # Binary 11111111 (all stations)
      name: 'mega_heart'
      resources:
        battery: 3
        water: 3
        helium: 3
        rare_earth: 3
      output:
        heart: 5
      energy_cost: 20

  # Unknown recipes generated procedurally
  unknown_recipe_count: 252
  unknown_recipe_seed: ${random_seed}
```

### ClipActionHandler Configuration

```yaml
# configs/actions/clip_infection.yaml
clip_infection:
  enabled: true

  # Infection spread parameters
  initial_probability: 0.001
  growth_rate: 1.05 # Exponential growth factor
  growth_period: 100 # Steps between growth applications

  # Spatial spread
  infection_range: 5 # Tiles
  proximity_multiplier: 2.0 # Infection chance multiplier per nearby infected

  # Security escalation
  base_security: 1
  security_increment_period: 500 # Steps between security increases
  security_increment: 1

  # Clearing requirements
  clearing_patterns:
    1: # Security level 1
      agents_required: 1
      tools_required:
        tool_basic: 1
    2: # Security level 2
      agents_required: 2
      tools_required:
        tool_basic: 2
    3: # Security level 3
      agents_required: 3
      tools_required:
        tool_basic: 3
      alternate:
        agents_required: 2
        tools_required:
          tool_advanced: 2

  # Visual indicators
  infection_sprite: 'clippy_corruption'
  infection_color: [128, 0, 128, 128] # Purple overlay
```

### Mine Configuration

```yaml
# configs/objects/mine.yaml
mine:
  hp: 50

  # Resource generation
  resource_type: 'rare_earth'
  resource_sprite: 'ore_brown'

  # Production rates
  depleted:
    initial_resources: 10
    cooldown: 20
    resources_per_use: 1

  rich:
    initial_resources: 100
    cooldown: 5
    resources_per_use: 1

  # Infection susceptibility
  can_be_infected: true
  infection_resistance: 0.0 # No natural resistance

  # Energy cost to harvest
  use_cost: 1

  # Visual appearance
  sprite_healthy: 'mine_entrance'
  sprite_depleted: 'mine_exhausted'
```

### ClippedMine Configuration

```yaml
# configs/objects/clipped_mine.yaml
clipped_mine:
  inherits: mine

  # Infection state
  infected: true
  infection_level: 1 # Current security level
  infection_time: 0 # Timestep when infected

  # Corrupted production
  resource_type: 'paperclip'
  resource_sprite: 'paperclip_silver'
  cooldown: 3 # Faster but useless

  # Paperclips have no value
  resource_value: 0

  # Spreading behavior
  spread_range: 5
  spread_strength: 2.0

  # Cannot be harvested normally
  harvestable: false
  harvest_message: 'ERROR: Output corrupted to paperclips'

  # Clearing requirements
  clearable: true
  clear_action: 'clear_infection'

  # Visual indicators
  sprite: 'mine_corrupted'
  particle_effect: 'purple_corruption'
  glow_color: [128, 0, 255]
  pulse_rate: 2.0 # Ominous pulsing
```

## Getting Started

To play Cogs vs Clippies:

```bash
# Install MettaGrid
pip install mettagrid

# Run with default configuration
python -m mettagrid.play --config configs/cogs_vs_clippies.yaml

# Run with custom parameters
python -m mettagrid.play \
  --config configs/cogs_vs_clippies.yaml \
  --num_agents 8 \
  --infection_rate 0.002 \
  --max_hearts_carried 5
```

Good luck, and remember: _In space, no one can hear you clip._
