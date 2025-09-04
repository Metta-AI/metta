## Unified placement system for terrain and structures
## Handles placement priority: river → terrain features → structures → objects

import std/[random, math, algorithm], vmath, terrain

type
  PlacementPriority* = enum
    ## Order matters - higher priority items are placed first
    PriorityRiver = 0      # Rivers always first - they shape the map
    PriorityTerrain = 1    # Wheat fields and trees
    PriorityStructure = 2  # Houses and temples 
    PriorityObject = 3     # Mines, generators, walls
    PriorityAgent = 4      # Agents placed last
  
  Structure* = object
    ## Generic structure that can represent houses, temples, or any building
    width*, height*: int
    centerPos*: IVec2      # Center/important position within structure
    needsBuffer*: bool     # Whether to enforce empty space around it
    bufferSize*: int       # How much buffer space
    layout*: seq[seq[char]] # Optional layout grid for complex structures
  
  PlacementGrid* = ptr array[100, array[50, pointer]]
  
  PlacementResult* = object
    success*: bool
    position*: IVec2
    message*: string

# Constants for standard structures
const
  DefaultHouseLayout* = @[
    @['#', '#', '.', '#', '#'],  # Top row with north entrance
    @['#', ' ', ' ', ' ', '#'],  # Second row  
    @['.', ' ', 'a', ' ', '.'],  # Middle row with altar and E/W entrances
    @['#', ' ', ' ', ' ', '#'],  # Fourth row
    @['#', '#', '.', '#', '#']   # Bottom row with south entrance
  ]

proc createStructure*(width, height: int, centerX, centerY: int, 
                     needsBuffer = false, bufferSize = 0): Structure =
  ## Create a generic structure
  result.width = width
  result.height = height  
  result.centerPos = ivec2(centerX.int32, centerY.int32)
  result.needsBuffer = needsBuffer
  result.bufferSize = bufferSize

proc createHouseStructure*(): Structure =
  ## Create a standard house structure
  result = createStructure(5, 5, 2, 2)
  result.layout = DefaultHouseLayout

proc createTempleStructure*(): Structure =
  ## Create a temple structure (simpler, with buffer zone)
  result = createStructure(3, 3, 1, 1, needsBuffer = true, bufferSize = 2)

# ============ Core placement logic ============

proc checkBounds(x, y, width, height, mapWidth, mapHeight: int): bool =
  ## Check if a rectangle fits within map bounds
  x >= 0 and y >= 0 and x + width <= mapWidth and y + height <= mapHeight

proc canPlaceAt*(grid: PlacementGrid, terrain: ptr TerrainGrid, 
                pos: IVec2, structure: Structure, 
                mapWidth, mapHeight: int, 
                allowWater = false): bool =
  ## Universal placement check for any structure
  
  # Check basic bounds
  if not checkBounds(pos.x, pos.y, structure.width, structure.height, mapWidth, mapHeight):
    return false
  
  # Check main structure area
  for dy in 0 ..< structure.height:
    for dx in 0 ..< structure.width:
      let gridX = pos.x + dx
      let gridY = pos.y + dy
      
      # Check for existing objects
      if not isNil(grid[gridX][gridY]):
        return false
      
      # Check terrain (unless water is explicitly allowed)
      if not allowWater and terrain[gridX][gridY] == Water:
        return false
  
  # Check buffer zone if required
  if structure.needsBuffer:
    for dy in -structure.bufferSize .. structure.height + structure.bufferSize - 1:
      for dx in -structure.bufferSize .. structure.width + structure.bufferSize - 1:
        let checkX = pos.x + dx
        let checkY = pos.y + dy
        
        # Skip checking the structure itself
        if dx >= 0 and dx < structure.width and dy >= 0 and dy < structure.height:
          continue
          
        # Check bounds for buffer
        if checkX >= 0 and checkX < mapWidth and checkY >= 0 and checkY < mapHeight:
          if not isNil(grid[checkX][checkY]):
            return false  # Something too close
  
  return true

proc findPlacement*(grid: PlacementGrid, terrain: ptr TerrainGrid,
                   structure: Structure, mapWidth, mapHeight, mapBorder: int,
                   r: var Rand, maxAttempts = 100, preferCorners = false): PlacementResult =
  ## Find a suitable location for any structure
  ## If preferCorners is true, tries corner locations first (for houses)
  
  # Calculate search bounds
  let minX = mapBorder + (if structure.needsBuffer: structure.bufferSize else: 0)
  let maxX = mapWidth - mapBorder - structure.width - (if structure.needsBuffer: structure.bufferSize else: 0)
  let minY = mapBorder + (if structure.needsBuffer: structure.bufferSize else: 0)
  let maxY = mapHeight - mapBorder - structure.height - (if structure.needsBuffer: structure.bufferSize else: 0)
  
  if maxX <= minX or maxY <= minY:
    return PlacementResult(success: false, message: "Map too small for structure")
  
  # If preferCorners, try corner regions first
  if preferCorners:
    # Define corner regions (25% of map from each corner)
    let cornerSize = min(mapWidth, mapHeight) div 4
    
    # Define the 4 corner regions with some randomness
    var cornerRegions: seq[tuple[minX, maxX, minY, maxY: int]] = @[]
    
    # Top-left corner
    cornerRegions.add((minX, min(minX + cornerSize, maxX), 
                       minY, min(minY + cornerSize, maxY)))
    # Top-right corner
    cornerRegions.add((max(maxX - cornerSize, minX), maxX,
                       minY, min(minY + cornerSize, maxY)))
    # Bottom-left corner
    cornerRegions.add((minX, min(minX + cornerSize, maxX),
                       max(maxY - cornerSize, minY), maxY))
    # Bottom-right corner
    cornerRegions.add((max(maxX - cornerSize, minX), maxX,
                       max(maxY - cornerSize, minY), maxY))
    
    # Shuffle corner order for variety
    for i in countdown(cornerRegions.len - 1, 1):
      let j = r.rand(0 .. i)
      swap(cornerRegions[i], cornerRegions[j])
    
    # Try each corner region
    for region in cornerRegions:
      for attempt in 0 ..< maxAttempts div 4:  # Fewer attempts per corner
        let x = r.rand(region.minX ..< region.maxX)
        let y = r.rand(region.minY ..< region.maxY)
        let pos = ivec2(x.int32, y.int32)
        
        if canPlaceAt(grid, terrain, pos, structure, mapWidth, mapHeight):
          return PlacementResult(success: true, position: pos)
  
  # Try random placement (original behavior)
  for attempt in 0 ..< maxAttempts:
    let x = r.rand(minX ..< maxX)
    let y = r.rand(minY ..< maxY)
    let pos = ivec2(x.int32, y.int32)
    
    if canPlaceAt(grid, terrain, pos, structure, mapWidth, mapHeight):
      return PlacementResult(success: true, position: pos)
  
  # Fall back to systematic search
  for y in minY ..< maxY:
    for x in minX ..< maxX:
      let pos = ivec2(x.int32, y.int32)
      if canPlaceAt(grid, terrain, pos, structure, mapWidth, mapHeight):
        return PlacementResult(success: true, position: pos)
  
  return PlacementResult(success: false, message: "No valid location found")

# ============ Terrain placement ============

proc placeRiver*(terrain: var TerrainGrid, mapWidth, mapHeight, mapBorder: int, 
                r: var Rand): seq[IVec2] =
  ## Generate a river and return its path
  ## This is always placed first as it shapes the entire map
  const riverWidth = 4
  
  result = @[]
  
  # Start on the left edge
  var currentPos = ivec2(mapBorder.int32, 
                         r.rand(mapBorder + riverWidth .. mapHeight - mapBorder - riverWidth).int32)
  
  # Generate main river path
  var hasFork = false
  var forkPoint: IVec2
  var secondaryPath: seq[IVec2] = @[]
  
  while currentPos.x >= mapBorder and currentPos.x < mapWidth - mapBorder and
        currentPos.y >= mapBorder and currentPos.y < mapHeight - mapBorder:
    result.add(currentPos)
    
    # Possible fork
    if not hasFork and result.len > 20 and r.rand(1.0) < 0.4:
      hasFork = true
      forkPoint = currentPos
      
      var secondaryDirection = ivec2(1, r.sample(@[-1, 1]).int32)
      var secondaryPos = forkPoint
      
      for i in 0 ..< 30:
        secondaryPos.x += 1
        secondaryPos.y += secondaryDirection.y
        if r.rand(1.0) < 0.2:
          secondaryPos.y += r.sample(@[-1, 0, 1]).int32
        
        if secondaryPos.x >= mapBorder and secondaryPos.x < mapWidth - mapBorder and
           secondaryPos.y >= mapBorder and secondaryPos.y < mapHeight - mapBorder:
          secondaryPath.add(secondaryPos)
        else:
          break
    
    # Move primarily right with meandering
    currentPos.x += 1
    if r.rand(1.0) < 0.3:
      currentPos.y += r.sample(@[-1, 0, 0, 1]).int32
  
  # Place water tiles for main river
  for pos in result:
    for dx in -riverWidth div 2 .. riverWidth div 2:
      for dy in -riverWidth div 2 .. riverWidth div 2:
        let waterX = pos.x + dx
        let waterY = pos.y + dy
        if waterX >= 0 and waterX < mapWidth and waterY >= 0 and waterY < mapHeight:
          terrain[waterX][waterY] = Water
  
  # Place water for secondary branch
  for pos in secondaryPath:
    for dx in -(riverWidth div 2 - 1) .. (riverWidth div 2 - 1):
      for dy in -(riverWidth div 2 - 1) .. (riverWidth div 2 - 1):
        let waterX = pos.x + dx
        let waterY = pos.y + dy
        if waterX >= 0 and waterX < mapWidth and waterY >= 0 and waterY < mapHeight:
          terrain[waterX][waterY] = Water
  
  # Add secondary path to result
  result.add(secondaryPath)

proc placeTerrainCluster*(terrain: var TerrainGrid, centerX, centerY, size: int,
                         terrainType: TerrainType, mapWidth, mapHeight: int, 
                         r: var Rand, density = 0.8) =
  ## Place a cluster of terrain (wheat or trees) around a point
  let radius = (size.float / 2.0).int
  
  for dx in -radius .. radius:
    for dy in -radius .. radius:
      let x = centerX + dx
      let y = centerY + dy
      
      if x >= 0 and x < mapWidth and y >= 0 and y < mapHeight:
        if terrain[x][y] == Empty:  # Don't overwrite water or existing features
          let dist = sqrt((dx * dx + dy * dy).float)
          if dist <= radius.float:
            let chance = density - (dist / radius.float) * 0.3
            if r.rand(1.0) < chance:
              terrain[x][y] = terrainType

proc placeWheatFields*(terrain: var TerrainGrid, mapWidth, mapHeight, mapBorder: int,
                      r: var Rand, numFields = 8) =
  ## Place wheat fields, preferring locations near water
  for i in 0 ..< numFields:
    var placed = false
    
    # Try to place near water
    for attempt in 0 ..< 20:
      let x = r.rand(mapBorder + 3 .. mapWidth - mapBorder - 3)
      let y = r.rand(mapBorder + 3 .. mapHeight - mapBorder - 3)
      
      # Check proximity to water
      var nearWater = false
      for dx in -5 .. 5:
        for dy in -5 .. 5:
          let checkX = x + dx
          let checkY = y + dy
          if checkX >= 0 and checkX < mapWidth and checkY >= 0 and checkY < mapHeight:
            if terrain[checkX][checkY] == Water:
              nearWater = true
              break
        if nearWater: break
      
      if nearWater or attempt > 10:
        let fieldSize = r.rand(5..20)
        placeTerrainCluster(terrain, x, y, fieldSize, Wheat, mapWidth, mapHeight, r, 0.9)
        placed = true
        break
    
    # Fallback placement
    if not placed:
      let x = r.rand(mapBorder + 3 .. mapWidth - mapBorder - 3)
      let y = r.rand(mapBorder + 3 .. mapHeight - mapBorder - 3)
      let fieldSize = r.rand(5..20)
      placeTerrainCluster(terrain, x, y, fieldSize, Wheat, mapWidth, mapHeight, r, 0.9)

proc placeTreeGroves*(terrain: var TerrainGrid, mapWidth, mapHeight, mapBorder: int,
                     r: var Rand, numGroves = 8) =
  ## Place tree groves across the map
  for i in 0 ..< numGroves:
    let x = r.rand(mapBorder + 3 .. mapWidth - mapBorder - 3)
    let y = r.rand(mapBorder + 3 .. mapHeight - mapBorder - 3)
    let groveSize = r.rand(5..20)
    placeTerrainCluster(terrain, x, y, groveSize, Tree, mapWidth, mapHeight, r, 0.7)

# ============ Structure element extraction ============

proc getStructureElements*(structure: Structure, topLeft: IVec2): tuple[
  center: IVec2,
  walls: seq[IVec2],
  entrances: seq[IVec2],
  special: seq[IVec2]  # For altars, spawn points, etc
] =
  ## Extract element positions from a structure with layout
  result.center = topLeft + structure.centerPos
  result.walls = @[]
  result.entrances = @[]
  result.special = @[]
  
  if structure.layout.len > 0:
    for y in 0 ..< structure.height:
      for x in 0 ..< structure.width:
        if y < structure.layout.len and x < structure.layout[y].len:
          let worldPos = topLeft + ivec2(x.int32, y.int32)
          case structure.layout[y][x]:
          of '#': result.walls.add(worldPos)
          of '.': result.entrances.add(worldPos)
          of 'a', 's', '*': result.special.add(worldPos)  # Various special tiles
          else: discard

# ============ Convenience functions for backward compatibility ============

proc findEmptyPosition*(grid: PlacementGrid, terrain: ptr TerrainGrid,
                       mapWidth, mapHeight, mapBorder: int, r: var Rand,
                       maxAttempts = 1000): IVec2 =
  ## Find any empty position on the map (for objects, agents, etc)
  for attempt in 0 ..< maxAttempts:
    let x = r.rand(mapBorder ..< mapWidth - mapBorder)
    let y = r.rand(mapBorder ..< mapHeight - mapBorder)
    
    if isNil(grid[x][y]) and terrain[x][y] != Water:
      return ivec2(x.int32, y.int32)
  
  # If we couldn't find anything randomly, give up
  # In production, you might want to do a systematic search
  return ivec2(-1, -1)

proc findEmptyPositionsAround*(grid: PlacementGrid, terrain: ptr TerrainGrid,
                              center: IVec2, radius: int,
                              mapWidth, mapHeight, mapBorder: int): seq[IVec2] =
  ## Find all empty positions around a center point
  result = @[]
  
  for dx in -radius .. radius:
    for dy in -radius .. radius:
      if dx == 0 and dy == 0:
        continue  # Skip center
      
      let x = center.x + dx
      let y = center.y + dy
      
      if x >= mapBorder and x < mapWidth - mapBorder and
         y >= mapBorder and y < mapHeight - mapBorder:
        if isNil(grid[x][y]) and terrain[x][y] != Water:
          result.add(ivec2(x.int32, y.int32))