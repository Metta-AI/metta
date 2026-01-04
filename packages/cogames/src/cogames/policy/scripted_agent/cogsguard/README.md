# CoGsGuard Scripted Agent

A role-based multi-agent policy for the Cogs vs Clips arena game.

## Game Rules

### Overview
CoGsGuard is a team-based resource management game where the **Cogs** team competes against the **Clips** team. The Cogs team uses this scripted policy while Clips can be controlled by another policy or bot.

### Resources
- **Elements**: carbon, oxygen, germanium, silicon (gathered from extractors)
- **Energy**: Required for movement (auto-regenerates near aligned structures)
- **Hearts**: Required for align/scramble actions
- **Influence**: Required for aligning supply depots
- **HP**: Health points

### Key Structures

| Structure | Owner | Function |
|-----------|-------|----------|
| **Main Nexus** | Cogs | Energy AOE regeneration, resource deposits, heart withdrawal |
| **Supply Depot (Charger)** | Clips (initially) | Can be scrambled (→neutral) then aligned (→cogs) |
| **Gear Stations** | Cogs | Dispense role-specific gear (costs commons resources) |
| **Extractors** | Neutral | Gather element resources (in map corners) |

### Gear System
Agents must acquire role-specific gear from gear stations before executing their role. Gear costs are paid from the **cogs commons** inventory:

| Gear | Cost | Bonus |
|------|------|-------|
| Miner | 3 carbon, 1 oxygen, 1 germanium, 1 silicon | +40 cargo capacity |
| Scout | 1 carbon, 1 oxygen, 1 germanium, 3 silicon | +100 energy, +400 HP |
| Aligner | 3 carbon, 1 oxygen, 1 germanium, 1 silicon | +20 influence capacity |
| Scrambler | 1 carbon, 3 oxygen, 1 germanium, 1 silicon | +200 HP |

### Supply Depot Mechanics
- **Align**: Convert neutral depot to cogs-aligned (requires aligner gear + 1 influence + 1 heart)
- **Scramble**: Remove depot's alignment (requires scrambler gear + 1 heart)
- Aligned depots provide energy AOE to their team

## Agent Strategy

### Role Assignment
Agents are assigned roles in round-robin order:
- Agent 0, 4, 8: **Miner**
- Agent 1, 5, 9: **Scout**
- Agent 2, 6: **Aligner**
- Agent 3, 7: **Scrambler**

### Phase System
Each agent follows a two-phase lifecycle:

```
┌─────────────┐     Get Gear     ┌──────────────┐
│  GET_GEAR   │ ───────────────► │ EXECUTE_ROLE │
└─────────────┘                  └──────────────┘
```

1. **GET_GEAR**: Find and bump the role-specific gear station
2. **EXECUTE_ROLE**: Perform role-specific behavior

### Role Behaviors

#### 🔨 Miner
```
1. Find nearest extractor (carbon/oxygen/germanium/silicon chests)
2. Navigate to extractor and extract resources
3. When cargo full (40 capacity), return to supply depot to deposit
4. Repeat
```

#### 🔭 Scout
```
1. Explore the map systematically (high energy allows long-range scouting)
2. Discover structures and resources for team knowledge
3. Patrol map edges to maximize coverage
```

#### 🔗 Aligner
```
1. Get influence from nexus AOE (stand nearby)
2. Get hearts from nexus (bump to withdraw from commons)
3. Find neutral supply depots (after scrambler has neutralized them)
4. Bump depot to align it to cogs
5. Repeat
```

#### 🌀 Scrambler
```
1. Get hearts from nexus or chest
2. Find clips-aligned supply depots (chargers)
3. Bump depot to scramble (remove alignment → neutral)
4. Repeat
```

### Exploration Strategy
Agents explore systematically by cycling through cardinal directions:
```
East (8 steps) → South (8 steps) → West (8 steps) → North (8 steps) → repeat
```
Starting direction is East (where gear stations are typically located in hub maps).

### Resource Flow
```
Extractors ──► Miners ──► Commons ──► Gear Stations ──► Agents
                              │
                              └──► Hearts ──► Aligners/Scramblers
```

## Known Limitations

1. **Aligner Timing**: Aligners often take too long to find their gear stations. By then, the commons may be depleted of resources needed for aligner gear.

2. **No Communication**: Agents don't share discovered locations. Each agent must independently explore to find structures.

3. **Random Station Placement**: Gear stations are randomly placed around the hub perimeter, making exploration outcomes variable.

## Usage

```bash
# Run with the cogsguard policy
./tools/run.py recipes.experiment.cogsguard.play policy_uri=metta://policy/cogsguard

# With limited timesteps and log rendering
./tools/run.py recipes.experiment.cogsguard.play policy_uri=metta://policy/cogsguard render=log max_steps=500
```

## File Structure

```
cogsguard/
├── __init__.py      # Exports CogsguardPolicy
├── policy.py        # Base agent logic, phase system, navigation
├── types.py         # State definitions (CogsguardAgentState, Role, Phase)
├── miner.py         # Miner role implementation
├── scout.py         # Scout role implementation
├── aligner.py       # Aligner role implementation
├── scrambler.py     # Scrambler role implementation
└── README.md        # This file
```

## Debug Mode

Set `DEBUG = True` in `policy.py` to enable detailed logging:
```python
DEBUG = True  # Enable debug logging
```

This will print agent decisions, discoveries, and phase transitions.

