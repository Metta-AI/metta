# Tribal Terrain System

## Features Implemented

### 1. River System
- Rivers spawn from map edges and flow through the terrain
- Rivers fork into two branches partway through
- 4 tiles wide for significant map impact
- **NOW PASSABLE** - Agents can traverse water (could add movement penalties later)

### 2. Clustered Wheat Fields
- 4-5 wheat field clusters spawn on the map
- Each cluster contains 5-20 wheat tiles
- Fields preferentially spawn near water sources
- Creates resource-rich zones for strategic competition

### 3. Tree Groves
- 4-5 tree groves spawn on the map
- Each grove contains 5-20 trees
- More organic, natural-looking distribution
- Provides visual variety and potential cover

### 4. House Integration
- Houses with altars spawn as complete structures
- Each house has walls, entrances, and a central altar
- 2-3 houses per map for variety

## Usage

### Run the terrain test:
```bash
nim c -r -d:release test_terrain_clusters.nim
```

### Run MettaScope with the new terrain:
```bash
cd mettascope2
nim r -d:release src/mettascope
```

## Map Legend
- `~` = Water (passable)
- `.` = Wheat field 
- `T` = Tree
- `#` = Wall
- `A` = Agent
- `g` = Generator
- `c` = Converter
- `a` = Altar

## Future Enhancements
- Movement speed penalties on different terrain
- Seasonal changes (frozen rivers, harvested wheat)
- Bridges as strategic chokepoints
- Different tree types with varying properties
- Terrain-based bonuses (defense in forests, resources from wheat)