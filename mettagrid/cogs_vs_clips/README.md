# Cogs vs Clips
*A Cooperative Resource Management Game*

## Game Summary

**Players:** 1-8 cooperative AI agents
**Game Time:** Variable (until time limit or infection spreads)
**Objective:** Manufacture the most hearts before the paperclip infection consumes the asteroid

## Story

The year is 2157. Your team of Cooperative Gathering Systems (CoGS) has been deployed to asteroid Machina IV on a critical resource extraction mission. Your goal: harvest valuable minerals and manufacture Holon Enhanced Agent Replication Transducers (hearts) using the advanced nano-assembler facility.

But danger lurks in the void. The Friendly Paperclip DAO, fresh from repelling yet another cyber attack by the Staples and Binders Alliance, suffered a catastrophic alignment failure in their security AI. The rogue system has one directive: convert everything to paperclips. Their mining operation on Machina IV has been transformed into a paperclip factory, and the infection is spreading patch by patch across the asteroid.

Time is running out. Can your team gather enough resources and manufacture enough hearts before the paperclip plague consumes everything?

## What's in the Box

### Game Board
- Asteroid Machina IV grid layout
- Central nano-assembler with 8 surrounding stations (N, NE, E, SE, S, SW, W, NW)
- Resource patches: Rare Earth Minerals, Trapped Helium, Mercury
- Solar stations for battery generation
- Storage facilities: Hearts chest and 4 resource depots

### Game Pieces
- 8 CoGS figures (customizable display glyphs)
- Resource tokens: Rare Earth Minerals, Trapped Helium, Mercury, Batteries
- Item tokens: Hearts, Nano Disruptors, Magnetic Resonators, Quantum Modulators, Lasers
- Paperclip infection markers
- Turn counter

## How to Play

### Setup
1. Place the game board in the center of the table
2. Position CoGS figures at starting locations
3. Distribute initial resource tokens on patches
4. Place the nano-assembler in the center
5. Set storage facilities around the board perimeter
6. Begin turn counter

### Game Turn Structure

#### 1. Movement Phase
- Each CoGS may move one space in any cardinal direction (N, E, S, W)
- CoGS cannot move onto patches or buildings
- CoGS can move through empty spaces and around other CoGS

#### 2. Activation Phase
- CoGS may attempt to move onto patches/buildings to activate them
- Activation occurs, but the CoGS remains in their current position
- Multiple CoGS can activate simultaneously if positioned correctly

#### 3. Communication Phase
- CoGS may change their display glyph (choose from 255 options)
- All CoGS can observe glyphs and inventories of others in their visual field

#### 4. Infection Spread Phase
- Roll dice to determine if paperclip infection spreads
- Clipped patches become unusable and begin deteriorating
- Infection typically spreads to adjacent patches

### Winning and Losing

**Victory Conditions:**
- Manufacture the target number of hearts before time runs out
- Successfully contain the paperclip infection

**Defeat Conditions:**
- Time limit reached without achieving heart production goal
- Paperclip infection consumes all resource patches
- All CoGS become unable to function

## Core Game Mechanics

### Resource Management
- **Carrying Capacity:** Each CoGS can hold 50 resources + 1 heart maximum
- **Resource Types:** Rare Earth Minerals, Trapped Helium, Mercury, Batteries
- **Storage:** Use directional storage depots to manage team inventory

### Manufacturing System
The nano-assembler creates items based on CoGS positioning and available resources:

**Hearts (Victory Points)**
- Requires: 4 CoGS at N, E, S, W stations + 10 of each resource type
- Most valuable item for winning the game

**Tools for Exploitation**
- **Nano Disruptor:** 1 CoGS at N + 5 Rare Earth + 3 Batteries
- **Magnetic Resonator:** 1 CoGS at E + 5 Trapped Helium + 2 Batteries
- **Quantum Modulator:** 1 CoGS at W + 3 Mercury + 5 Batteries
- **Laser:** 1 CoGS at S + 2 Rare Earth + 3 Helium + 2 Mercury

### Paperclip Counter-Operations
- **Clipped Patches:** Infected patches have security vulnerabilities
- **Exploitation:** Surround clipped patches with CoGS carrying correct tools
- **Activation:** One CoGS activates the patch to exploit the vulnerability
- **Result:** Varies (patch restoration, resource recovery, infection containment)

### Communication Strategy
- Use 255 display glyphs to coordinate team actions
- Visual field allows inventory and status monitoring
- Critical for coordinating complex manufacturing recipes

## Strategic Tips

1. **Early Game:** Focus on resource gathering and establishing supply chains
2. **Mid Game:** Balance manufacturing tools for exploitation with heart production
3. **Late Game:** Race against infection spread while maximizing heart output
4. **Team Coordination:** Use glyphs to signal roles, needs, and planned actions
5. **Resource Management:** Utilize storage depots to exceed individual carrying limits
6. **Infection Response:** Manufacture exploitation tools before patches get clipped

## Game Rules

### Objective
Manufacture as many hearts as possible before time runs out or the paperclip infection consumes all resources.

### Movement and Activation Mechanics

#### Basic Movement
- **CoGS Movement**: CoGS can move in 4 directions (north, east, west, south)
- **Free Movement**: CoGS move normally to empty spaces
- **Blocked Movement**: CoGS cannot move onto patches/buildings - they remain solid obstacles

#### Move-to-Use System (`move_and_activate` action)
- **Unified Action**: Single action that handles both movement and object interaction
- **Smart Behavior**:
  - Empty space → Agent moves there
  - Object/Building → Agent activates it while staying in current position
  - Wall/Obstacle → Action fails
- **Seamless Integration**: Eliminates need for separate move and activate actions
- **Resource Management**: Automatically handles resource transfers with converters

#### Activation Results
- **Resource Patches**: Automatically collect resources and handle cooldowns
- **Converters**: Transfer input resources and collect output items
- **Nano-Assembler**: Execute recipes based on positioned agents
- **Storage Facilities**: Deposit/withdraw based on approach direction
- **Combat**: Can still use separate attack action for aggressive interactions

### Communication and Visual Information
- **Display Glyph**: Each CoGS can change their current display to any of 255 available glyphs
- **Visual Field**: CoGS can see the glyphs and inventory contents of other CoGS within their visual field
- **Purpose**: Enables visual communication and coordination between team members

### Paperclip Infection System

#### Algorithmic Spread Model
- **Exponential Growth**: P(infection) = base_rate × growth_rate^(time/period)
- **Proximity Boost**: 2x infection rate for patches adjacent to existing clipped areas
- **Deterministic Timing**: Uses seeded RNG for reproducible infection patterns
- **Target Selection**: Prioritizes rich resource patches over depleted ones

#### Infection Mechanics
- **Resource Conversion**: Infected patches stop producing original resources
- **Consumption Rate**: Clipped patches deteriorate over time, eventually becoming unusable
- **Visual Indicator**: Infected patches display paperclip symbols and change color
- **Spread Pattern**: Typically forms contiguous infected regions

#### Security Exploitation System
- **Tool Requirements**: Different clipped patches require specific tool combinations
- **Positioning**: CoGS must surround infected patches with required tools
- **Security Levels**: Higher-tier infections need more sophisticated tool combinations
- **Exploitation Results**: Can restore patches, recover resources, or slow infection spread
- **Energy Costs**: Exploitation attempts consume energy from participating CoGS

### Nano-Assembler System
- **Location**: Central manufacturing facility with 8 surrounding stations
- **Recipe Activation**: Recipes are coded by which stations are occupied by CoGS
- **Operation**: Activating the nano-assembler attempts to execute the current recipe, but the CoGS remains in their station position

### Nano-Assembler Recipe System

The nano-assembler uses an 8-station pattern matching system where CoGS positions encode recipes as binary patterns (0-255 possible recipes).

#### Station Layout
```
   N  NE
NW  🔧  E
W   🔧  SE
   SW  S
```

#### Recipe Encoding
- **8-bit Pattern**: Each station position represents one bit (N=0, NE=1, E=2, SE=3, S=4, SW=5, W=6, NW=7)
- **Binary Value**: Recipe ID = sum of 2^position for occupied stations
- **Resource Pooling**: All participating CoGS contribute their resources to recipe
- **Energy Sharing**: Energy costs distributed among positioned CoGS

#### Key Recipes

**Heart Recipe (Pattern: 85 = 01010101₂)**
- **Positions**: N, E, S, W stations (alternating pattern)
- **Ingredients**: 10 rare earth + 10 helium + 10 mercury (pooled from all participants)
- **Energy Cost**: Distributed among 4 positioned CoGS
- **Output**: Heart delivered to activating CoGS

**Tool Recipes (Single Station)**
- **Nano Disruptor (Pattern: 1)**: N station, 5 rare earth + 3 batteries
- **Magnetic Resonator (Pattern: 4)**: E station, 5 helium + 2 batteries
- **Quantum Modulator (Pattern: 64)**: W station, 3 mercury + 5 batteries
- **Laser (Pattern: 16)**: S station, 2 rare earth + 3 helium + 2 mercury

#### Recipe Discovery System
- **First Discovery Bonus**: Extra rewards for executing new recipe patterns
- **Pattern Learning**: CoGS can experiment with different station combinations
- **Resource Efficiency**: Failed recipes consume minimal resources

### Inventory Limitations
- **Heart Capacity**: Each CoGS can carry maximum 1 heart
- **Resource Capacity**: Each CoGS can carry maximum 50 resources total

### Storage Facilities

#### Hearts Chest
- **Deposit**: CoGS holding a heart can deposit it by activating the hearts chest from the right side
- **Withdraw**: CoGS can withdraw a heart by activating the hearts chest from the left side
- **Interaction**: CoGS remains in their current position when interacting with the chest
- **Purpose**: Allows team coordination and heart storage beyond individual carrying capacity

#### Resource Storage Depots
- **Rare Earth Minerals Depot**: Works like hearts chest (activate from right = deposit, left = withdraw)
- **Trapped Helium Depot**: Works like hearts chest (activate from right = deposit, left = withdraw)
- **Mercury Depot**: Works like hearts chest (activate from right = deposit, left = withdraw)
- **Battery Depot**: Works like hearts chest (activate from right = deposit, left = withdraw)
- **Interaction**: CoGS remains in their current position when interacting with depots
- **Purpose**: Allows team resource management beyond individual 50-resource carrying capacity

## Objects

### CoGS (Cooperative Gathering Systems)
- **Agent Class**: Primary controllable entities with position and orientation
- **Inventory System**: 50 total resource capacity + 1 heart maximum
- **Stats Tracking**: Automatic metrics collection for all actions and resources
- **Glyph Display**: 0-255 visual communication values
- **Freeze Mechanics**: Can be temporarily disabled via attack system

### Resource Extractors

#### Depleted Patches (Type IDs 21-24)
- **Low Yield**: Minimal resource production per activation
- **Resource Types**: Rare earth minerals, trapped helium, mercury, batteries
- **Cooldown**: Time delay between extractions
- **Vulnerability**: Can become infected and converted to paperclip production

#### Rich Patches (Type IDs 31-34)
- **High Yield**: Maximum resource production per activation
- **Priority Targets**: Preferred by infection algorithm
- **Resource Types**: Same as depleted but higher quantities
- **Strategic Value**: Critical for efficient heart production

#### Infected Patches (Type IDs 41-44)
- **Converted Production**: Produces paperclips instead of original resources
- **Security Vulnerabilities**: Exploitable with correct tool combinations
- **Deterioration**: Gradually becomes completely unusable
- **Spread Source**: Infection radiates to adjacent patches

### Manufacturing Facilities

#### Nano-Assembler (Type ID 20)
- **8-Station Layout**: N, NE, E, SE, S, SW, W, NW positions around central hub
- **Binary Recipe Encoding**: 256 possible recipes based on occupied stations
- **Resource Pooling**: Collects ingredients from all positioned CoGS
- **Energy Distribution**: Shares costs among recipe participants

#### Communal Cache (Type ID 8)
- **Heart Storage**: Unlimited capacity for manufactured hearts
- **Directional Interaction**: Approach direction determines deposit/withdraw
- **Team Coordination**: Enables storage beyond individual carrying limits
- **Strategic Buffer**: Critical for managing production vs. infection race

### Hidden Recipe Converters (Type IDs 50-52)
- **Implementation Detail**: Internal objects for recipe execution logic
- **Not Visible**: Don't appear on game board but handle recipe processing
- **Resource Transformation**: Convert raw materials to finished products
- **Event Integration**: Trigger automatic production cycles

## Actions

- **MoveAndActivate**: Combined movement and interaction (primary action for gameplay)
  - Direction parameter (North, East, West, South)
  - Moves to empty spaces, activates objects when blocked
- **Move**: Traditional movement-only action (for scenarios requiring pure movement)
- **Attack**: Combat action for aggressive interactions
- **Rotate**: Change facing direction
- **ChangeGlyph**: Select any of 255 available display glyphs for visual communication

### MoveAndActivate Behavior
- **Empty Space**: Agent moves to the target position
- **Resource Patches**: Collect resources while staying in current position
- **Converters**: Deposit input resources and collect outputs while staying in current position
- **Nano-Assembler**: Participate in recipe execution while staying in current position
- **Storage Facilities**: Deposit/withdraw items based on approach direction
- **Walls/Obstacles**: Action fails, agent remains in current position
- **Clipped Patches**: Exploit security vulnerabilities (with proper tools and positioning)

---

## Technical Details

*This section contains precise implementation details based on the actual mettagrid C++ codebase.*

### Architecture Overview

#### Core Engine
- **C++ Core:** High-performance grid environment with Python bindings via Pybind11
- **Configuration System:** Pydantic models in Python with C++ reflection
- **Event System:** Scheduled events for conversions, cooldowns, and infection spread
- **Builder Pattern:** Game-specific configurations built in Python, executed in C++

#### Grid System (`grid.hpp`)
- **Grid Class:** Central game world manager with multiple layers (AgentLayer, ObjectLayer)
- **Coordinates:** `PackedCoordinate` for memory-efficient position storage
- **Dimensions:** 40×40 grid for Cogs vs Clippies (configurable)
- **Layers:** Separate tracking for agents and objects to prevent conflicts

### GameObject Hierarchy

#### Base Classes
- **GridObject:** Base class for all game entities with position, type_id, and lifecycle
- **ActionHandler:** Base for all actions with resource requirements and consumption patterns

#### Agent Implementation (`objects/agent.hpp`)
- **Inventory System:** Per-resource limits with automatic reward computation
- **Orientation:** 4-directional facing for movement and interaction
- **Stats Tracking:** Hierarchical metrics (e.g., "action.attack.success")
- **Frozen Mechanism:** Temporary disabling via attack system
- **Visitation Grid:** Tracks agent movement patterns for analysis

#### Converter Objects (`objects/converter.hpp`)
- **Resource Transformation:** Input/output mappings with conversion ratios
- **Event-Driven Production:** Automatic cycles when resources available
- **Cooldown Management:** Time-based restrictions on usage
- **Capacity Limits:** Maximum output per production cycle

### Action System Implementation

#### Core Actions
- **Move:** Grid navigation with collision detection
- **MoveAndActivate:** Combined movement and object interaction (implements move-to-use mechanic)
- **Rotate:** Orientation changes (4 directions)
- **Attack:** Combat with energy cost and freeze duration
- **GetOutput/PutRecipeItems:** Resource transfer between agents and converters
- **ChangeGlyph:** Visual communication (0-255 glyph values)

#### Cogs vs Clippies Specific Actions

##### Nano Recipe System (`actions/nano_recipe.hpp`)
- **8-Station Pattern:** Binary encoding of agent positions around nano-assembler
- **Recipe Lookup:** 256 possible recipes (8-bit pattern matching)
- **Resource Pooling:** Collects ingredients from all participating agents
- **Energy Distribution:** Shares energy costs among recipe participants
- **Discovery Bonuses:** Rewards for first-time recipe execution

##### Clippy Infection (`actions/clippy_infection.hpp`)
- **Exponential Growth:** P(t) = base_rate × growth_rate^(t/period)
- **Proximity Boost:** Higher infection rate near existing clipped patches
- **Security Levels:** Tool requirements for clearing infections
- **Algorithmic Spread:** Deterministic with configurable parameters

### Object Type System

#### Predefined Types (from `builder/cogs_vs_clippies.py`)
- **Nano-assembler (20):** Central crafting station with 8 surrounding positions
- **Resource Extractors:**
  - Depleted (21-24): Low-yield sources
  - Rich (31-34): High-yield sources
  - Infected (41-44): Clipped patches with vulnerabilities
- **Communal Cache (8):** Heart storage facility
- **Recipe Converters (50-52):** Hidden objects for recipe execution logic

### Configuration System

#### Python Configuration (`mettagrid_config.py`)
```python
@dataclass
class CogsVsClippiesConfig(MettaGridConfig):
    grid_size: tuple[int, int] = (40, 40)
    max_agents: int = 8
    resource_types: dict[str, int] = {
        "rare_earth": 0,
        "trapped_helium": 1,
        "mercury": 2,
        "battery": 3,
        "heart": 4
    }
    infection_params: dict = {
        "base_rate": 0.01,
        "growth_rate": 1.1,
        "period": 100.0,
        "proximity_boost": 2.0
    }
```

#### C++ Integration
- **Shared Pointers:** Automatic memory management for game objects
- **Configuration Validation:** Type-safe parameter checking at initialization
- **Dynamic Object Creation:** Objects instantiated from configuration dictionaries

### Observation System

#### Token-Based Observations
- **Feature-Value Pairs:** Efficient encoding of object properties
- **Configurable Windows:** 3×3 to 15×15 observation ranges
- **Global Information:** Completion percentage, last action/reward
- **Multi-Layer:** Separate observations for agents and objects

#### Agent Visibility
- **Visual Field:** Configurable radius for observing other agents
- **Inventory Transparency:** All agents can see others' resource counts
- **Glyph Communication:** 8-bit display values for coordination

### Event System

#### EventManager (`grid.hpp`)
- **Scheduled Events:** Time-based triggers for conversions and infections
- **Priority Queuing:** Deterministic execution order for concurrent events
- **Automatic Cleanup:** Event removal after execution

#### Converter Production Cycles
- **Input Monitoring:** Automatic triggering when resources available
- **Output Delivery:** Results placed in converter for agent collection
- **Cooldown Enforcement:** Time delays between production cycles

### Performance Characteristics

#### Computational Complexity
- **Movement:** O(1) position updates with collision checking
- **Nano Recipe:** O(8) for station pattern matching and resource pooling
- **Infection Spread:** O(adjacent_patches) per infected location
- **Observation:** O(window_size²) per agent per step

#### Memory Usage
- **Grid Storage:** Sparse representation for empty cells
- **Agent State:** Fixed-size structures with inventory arrays
- **Event Queue:** Dynamic sizing based on active converters and infections

### Implementation Status

#### Completed Components
- Core C++ engine with Python bindings
- Configuration system and builder patterns
- Basic action handlers (move, rotate, attack, etc.)
- Agent and converter object implementations
- Stats tracking and reward systems

#### Partially Implemented
- Nano recipe action handler (defined but not integrated)
- Clippy infection action handler (defined but not integrated)
- Map generation (ASCII builder exists but needs symbol mapping)

#### Integration Requirements
- Action handler registration in `mettagrid_c.cpp`
- Symbol-to-object mapping for map builder
- Event system integration for infection spread
- Recipe execution triggering via nano-assembler activation

### Development Workflow

#### Configuration-Driven Development
1. Define game mechanics in Python configuration
2. Create builder functions for object placement
3. Implement action handlers in C++ (if needed)
4. Register new actions in main environment
5. Test via Python training scripts

#### No C++ Changes Required For:
- New object types (via configuration)
- Resource balancing and rewards
- Map layouts and object placement
- Training curricula and evaluation scenarios