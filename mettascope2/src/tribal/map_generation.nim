import tribal_game
## Map generation and initialization
## Handles world setup, building placement, and entity spawning

import std/[random, tables], vmath, chroma
import environment_core

# Global village color management
var agentVillageColors*: seq[Color] = @[]
var altarColors*: Table[IVec2, Color] = initTable[IVec2, Color]()

proc randomEmptyPos*(r: var Rand, env: Environment): IVec2 =
  ## Find an empty position in the environment
  for i in 0 ..< 100:
    let pos = ivec2(r.rand(0 ..< MapWidth), r.rand(0 ..< MapHeight))
    if env.isEmpty(pos):
      result = pos
      return
  # If we can't find empty position after 100 tries, return a border position
  result = ivec2(MapBorder, MapBorder)

proc findEmptyPositionsAround*(env: Environment, center: IVec2, radius: int): seq[IVec2] =
  ## Find empty positions around a center point
  result = @[]
  for dx in -radius..radius:
    for dy in -radius..radius:
      if abs(dx) + abs(dy) <= radius and (dx != 0 or dy != 0):
        let pos = center + ivec2(dx, dy)
        if env.isEmpty(pos):
          result.add(pos)

proc getHouseCorners*(mapWidth, mapHeight, mapBorder: int): array[4, IVec2] =
  ## Get the four corner positions for houses
  let margin = mapBorder + 3
  result = [
    ivec2(margin, margin),                                    # Top-left
    ivec2(mapWidth - margin - 5, margin),                     # Top-right (5 is house width)
    ivec2(margin, mapHeight - margin - 5),                    # Bottom-left (5 is house height)
    ivec2(mapWidth - margin - 5, mapHeight - margin - 5)      # Bottom-right
  ]

proc generateVillageColor*(villageId: int): Color =
  ## Generate a unique color for a village
  let hue = (villageId.float * 137.5) mod 360.0 / 360.0
  return color(hue, 0.7, 0.8, 1.0)

proc initMapGeneration*(env: Environment, seed: int = 2024) =
  ## Initialize the environment with terrain, buildings, and entities
  var r = initRand(seed)
  
  # Initialize terrain
  env.terrain.initTerrain(MapWidth, MapHeight, MapBorder, seed)
  
  # Clear and prepare village colors arrays
  agentVillageColors.setLen(MapRoomObjectsAgents)
  altarColors.clear()
  
  # Place border walls if configured
  if MapBorder > 0:
    for x in 0 ..< MapWidth:
      for j in 0 ..< MapBorder:
        env.add(Thing(kind: Wall, pos: ivec2(x, j)))
        env.add(Thing(kind: Wall, pos: ivec2(x, MapHeight - j - 1)))
    for y in 0 ..< MapHeight:
      for j in 0 ..< MapBorder:
        env.add(Thing(kind: Wall, pos: ivec2(j, y)))
        env.add(Thing(kind: Wall, pos: ivec2(MapWidth - j - 1, y)))
  
  # Create placement grid
  var placementGrid: PlacementGrid
  for x in 0 ..< MapWidth:
    for y in 0 ..< MapHeight:
      if env.grid[x][y] != nil:
        placementGrid[x][y] = true
  
  # Spawn houses with their altars, walls, corner buildings, and agents
  var totalAgentsSpawned = 0
  let housePositions = getHouseCorners(MapWidth, MapHeight, MapBorder)
  let housesToSpawn = min(MapRoomObjectsHouses, 4)
  
  for villageId in 0 ..< housesToSpawn:
    let house = createHouse()
    let housePos = housePositions[villageId]
    
    # Generate unique color for this village
    let villageColor = generateVillageColor(villageId)
    
    # Get all house elements
    let elements = house.getHouseElements(housePos)
    
    # Add the altar
    env.add(Thing(
      kind: Altar,
      pos: elements.center,
      hearts: MapObjectAltarInitialHearts,
      cooldown: 0
    ))
    altarColors[elements.center] = villageColor
    
    # Add walls
    for wallPos in elements.walls:
      env.add(Thing(kind: Wall, pos: wallPos))
    
    # Add corner buildings
    env.add(Thing(kind: Armory, pos: elements.armory, cooldown: 0))
    env.add(Thing(kind: Forge, pos: elements.forge, cooldown: 0))
    env.add(Thing(kind: ClayOven, pos: elements.clayOven, cooldown: 0))
    env.add(Thing(kind: WeavingLoom, pos: elements.weavingLoom, cooldown: 0))
    
    # Spawn agents around this house
    let emptySpots = env.findEmptyPositionsAround(elements.center, 5)
    var agentsForThisHouse = 0
    let maxAgentsPerHouse = if villageId < housesToSpawn - 1:
      MapAgentsPerHouse
    else:
      # Last house gets any remaining agents
      MapRoomObjectsAgents - totalAgentsSpawned
    
    for spawnPos in emptySpots:
      if agentsForThisHouse < maxAgentsPerHouse and totalAgentsSpawned < MapRoomObjectsAgents:
        let agentId = totalAgentsSpawned
        env.add(Thing(
          kind: Agent,
          agentId: agentId,
          pos: spawnPos,
          orientation: Orientation(r.rand(0..7)),
          homeAltar: elements.center,  # Link agent to their home altar
          inventoryOre: 0,
          inventoryBattery: 0,
          inventoryWater: 0,
          inventoryWheat: 0,
          inventoryWood: 0,
          inventorySpear: 0,
          reward: 0.0
        ))
        
        # Store the village color for this agent
        agentVillageColors[agentId] = villageColor
        
        inc agentsForThisHouse
        inc totalAgentsSpawned
        
        if totalAgentsSpawned >= MapRoomObjectsAgents:
          break
  
  # Spawn remaining agents randomly if needed
  let neutralColor = color(0.5, 0.5, 0.5, 1.0)  # Gray for unaffiliated
  while totalAgentsSpawned < MapRoomObjectsAgents:
    let agentId = totalAgentsSpawned
    let pos = r.randomEmptyPos(env)
    env.add(Thing(
      kind: Agent,
      agentId: agentId,
      pos: pos,
      orientation: Orientation(r.rand(0..7)),
      homeAltar: ivec2(-1, -1),  # No home altar
      inventoryOre: 0,
      inventoryBattery: 0,
      inventoryWater: 0,
      inventoryWheat: 0,
      inventoryWood: 0,
      inventorySpear: 0,
      reward: 0.0
    ))
    agentVillageColors[agentId] = neutralColor
    inc totalAgentsSpawned
  
  # Spawn temples for clippys (place in less occupied areas)
  let templeCount = 2
  for i in 0 ..< templeCount:
    let temple = createTemple()
    # Try to place temples away from houses
    let mapCenterX = MapWidth div 2
    let mapCenterY = MapHeight div 2
    let offsetX = if i == 0: -15 else: 15
    let templatePos = ivec2(mapCenterX + offsetX, mapCenterY)
    
    # Place temple if position is empty
    if env.isEmpty(templatePos):
      env.add(Thing(
        kind: Temple,
        pos: templatePos,
        cooldown: 0
      ))
      
      # Spawn initial clippys near temple
      let clippyCount = 3
      for j in 0 ..< clippyCount:
        let angle = (j.float / clippyCount.float) * 2 * 3.14159
        let spawnRadius = 3
        let clippyPos = templatePos + ivec2(
          (cos(angle) * spawnRadius.float).int,
          (sin(angle) * spawnRadius.float).int
        )
        
        if env.isEmpty(clippyPos):
          env.add(Thing(
            kind: Clippy,
            pos: clippyPos,
            homeTemple: templatePos,
            wanderRadius: 5,
            wanderAngle: angle,
            targetPos: ivec2(-1, -1),
            wanderStepsRemaining: 0
          ))
  
  # Spawn converters
  for i in 0 ..< MapRoomObjectsConverters:
    let pos = r.randomEmptyPos(env)
    env.add(Thing(
      kind: Converter,
      pos: pos,
      cooldown: 0
    ))
  
  # Spawn mines
  for i in 0 ..< MapRoomObjectsMines:
    let pos = r.randomEmptyPos(env)
    env.add(Thing(
      kind: Mine,
      pos: pos,
      resources: MapObjectMineInitialResources,
      cooldown: 0
    ))
  
  # Spawn additional random walls
  for i in 0 ..< MapRoomObjectsWalls:
    let pos = r.randomEmptyPos(env)
    env.add(Thing(kind: Wall, pos: pos))
  
  # Update all agent observations
  for agentId in 0 ..< MapAgents:
    env.updateObservations(agentId)