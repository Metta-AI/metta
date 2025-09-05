## Unified terrain and placement system
## Handles terrain generation and structure placement with priority ordering

import std/[random, math, algorithm], vmath

type
  TerrainType* = enum
    Empty
    Water
    Wheat
    Tree

  TerrainGrid* = array[100, array[50, TerrainType]]  # 100x50 map size

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
    cornerUsed*: int  # Which corner was used (0-3), or -1 if not a corner

const
  DefaultHouseLayout* = @[
    @['A', '#', '.', '#', 'F'],  # Top row with Armory (A) top-left, Forge (F) top-right
    @['#', ' ', ' ', ' ', '#'],  # Second row
    @['.', ' ', 'a', ' ', '.'],  # Middle row with altar and E/W entrances
    @['#', ' ', ' ', ' ', '#'],  # Fourth row
    @['C', '#', '.', '#', 'W']   # Bottom row with Clay Oven (C) bottom-left, Weaving Loom (W) bottom-right
  ]

proc toIVec2*(x, y: int): IVec2 =
  ## Helper to create IVec2 from ints
  result.x = x.int32
  result.y = y.int32

proc checkBounds(x, y, width, height, mapWidth, mapHeight: int): bool =
  ## Check if a rectangle fits within map bounds
  x >= 0 and y >= 0 and x + width <= mapWidth and y + height <= mapHeight

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

proc generateRiver*(terrain: var TerrainGrid, mapWidth, mapHeight, mapBorder: int, r: var Rand) =
  ## Generate a river that flows from left to right across the map
  const riverWidth = 4
  
  var riverPath: seq[IVec2] = @[]
  
  # Always start on the left edge, randomly positioned vertically
  var currentPos = toIVec2(mapBorder, r.rand(mapBorder + riverWidth .. mapHeight - mapBorder - riverWidth))
  
  # Generate main river path
  var hasFork = false
  var forkPoint: IVec2
  var secondaryPath: seq[IVec2] = @[]
  
  while currentPos.x >= mapBorder and currentPos.x < mapWidth - mapBorder and
        currentPos.y >= mapBorder and currentPos.y < mapHeight - mapBorder:
    riverPath.add(currentPos)
    
    # Decide if we should fork (only once, after traveling some distance)
    if not hasFork and riverPath.len > 20 and r.rand(1.0) < 0.4:
      hasFork = true
      forkPoint = currentPos
      
      # Create secondary branch that also flows right but diverges up or down
      var secondaryDirection = toIVec2(1, r.sample(@[-1, 1]))  # Flow right and either up or down
      
      var secondaryPos = forkPoint
      for i in 0 ..< 30:  # Longer secondary branch for wider map
        secondaryPos.x += 1  # Always move right
        secondaryPos.y += secondaryDirection.y  # Move in chosen vertical direction
        # Add some randomness to secondary path
        if r.rand(1.0) < 0.2:
          secondaryPos.y += r.sample(@[-1, 0, 1]).int32
        
        if secondaryPos.x >= mapBorder and secondaryPos.x < mapWidth - mapBorder and
           secondaryPos.y >= mapBorder and secondaryPos.y < mapHeight - mapBorder:
          secondaryPath.add(secondaryPos)
        else:
          break
    
    # Move primarily right with some vertical meandering
    currentPos.x += 1  # Always move right
    if r.rand(1.0) < 0.3:
      # Add vertical movement for more natural meandering
      currentPos.y += r.sample(@[-1, 0, 0, 1]).int32  # Bias towards staying straight
  
  # Place water tiles for main river
  for pos in riverPath:
    for dx in -riverWidth div 2 .. riverWidth div 2:
      for dy in -riverWidth div 2 .. riverWidth div 2:
        let waterPos = pos + toIVec2(dx, dy)
        if waterPos.x >= 0 and waterPos.x < mapWidth and
           waterPos.y >= 0 and waterPos.y < mapHeight:
          terrain[waterPos.x][waterPos.y] = Water
  
  # Place water tiles for secondary branch
  for pos in secondaryPath:
    for dx in -(riverWidth div 2 - 1) .. (riverWidth div 2 - 1):
      for dy in -(riverWidth div 2 - 1) .. (riverWidth div 2 - 1):
        let waterPos = pos + toIVec2(dx, dy)
        if waterPos.x >= 0 and waterPos.x < mapWidth and
           waterPos.y >= 0 and waterPos.y < mapHeight:
          terrain[waterPos.x][waterPos.y] = Water

proc createWheatField*(terrain: var TerrainGrid, centerX, centerY: int, size: int, mapWidth, mapHeight: int, r: var Rand) =
  ## Create a wheat field cluster around a center point
  let radius = (size.float / 2.0).int
  for dx in -radius .. radius:
    for dy in -radius .. radius:
      let x = centerX + dx
      let y = centerY + dy
      if x >= 0 and x < mapWidth and y >= 0 and y < mapHeight:
        if terrain[x][y] == Empty:
          # Use distance from center to create more organic shape
          let dist = sqrt((dx * dx + dy * dy).float)
          if dist <= radius.float:
            let chance = 1.0 - (dist / radius.float) * 0.3  # Higher chance near center
            if r.rand(1.0) < chance:
              terrain[x][y] = Wheat

proc createTreeGrove*(terrain: var TerrainGrid, centerX, centerY: int, size: int, mapWidth, mapHeight: int, r: var Rand) =
  ## Create a tree grove cluster around a center point
  let radius = (size.float / 2.0).int
  for dx in -radius .. radius:
    for dy in -radius .. radius:
      let x = centerX + dx
      let y = centerY + dy
      if x >= 0 and x < mapWidth and y >= 0 and y < mapHeight:
        if terrain[x][y] == Empty:
          # Use distance from center to create more organic shape
          let dist = sqrt((dx * dx + dy * dy).float)
          if dist <= radius.float:
            let chance = 0.8 - (dist / radius.float) * 0.4  # Trees less dense than wheat
            if r.rand(1.0) < chance:
              terrain[x][y] = Tree

proc generateWheatFields*(terrain: var TerrainGrid, mapWidth, mapHeight, mapBorder: int, r: var Rand) =
  ## Generate 7-10 clustered wheat fields for 100x50 map
  let numFields = r.rand(7..10)
  
  for i in 0 ..< numFields:
    # Try to place near water if possible
    var placed = false
    for attempt in 0 ..< 20:
      let x = r.rand(mapBorder + 3 .. mapWidth - mapBorder - 3)
      let y = r.rand(mapBorder + 3 .. mapHeight - mapBorder - 3)
      
      # Check if near water
      var nearWater = false
      for dx in -5 .. 5:
        for dy in -5 .. 5:
          let checkX = x + dx
          let checkY = y + dy
          if checkX >= 0 and checkX < mapWidth and checkY >= 0 and checkY < mapHeight:
            if terrain[checkX][checkY] == Water:
              nearWater = true
              break
        if nearWater:
          break
      
      # Prefer locations near water, but accept any after some attempts
      if nearWater or attempt > 10:
        let fieldSize = r.rand(3..10)  # Each field has 5-20 wheat tiles
        createWheatField(terrain, x, y, fieldSize, mapWidth, mapHeight, r)
        placed = true
        break
    
    # Fallback: place anywhere if no good spot found
    if not placed:
      let x = r.rand(mapBorder + 3 .. mapWidth - mapBorder - 3)
      let y = r.rand(mapBorder + 3 .. mapHeight - mapBorder - 3)
      let fieldSize = r.rand(3..10)
      createWheatField(terrain, x, y, fieldSize, mapWidth, mapHeight, r)

proc generateTrees*(terrain: var TerrainGrid, mapWidth, mapHeight, mapBorder: int, r: var Rand) =
  ## Generate 7-10 tree groves for 100x50 map
  let numGroves = r.rand(7..10)
  
  for i in 0 ..< numGroves:
    let x = r.rand(mapBorder + 3 .. mapWidth - mapBorder - 3)
    let y = r.rand(mapBorder + 3 .. mapHeight - mapBorder - 3)
    let groveSize = r.rand(3..10)  # Each grove has 3-10 trees
    createTreeGrove(terrain, x, y, groveSize, mapWidth, mapHeight, r)

proc initTerrain*(terrain: var TerrainGrid, mapWidth, mapHeight, mapBorder: int, seed: int = 2024) =
  ## Initialize terrain with all features
  var r = initRand(seed)
  
  # Clear terrain to empty
  for x in 0 ..< mapWidth:
    for y in 0 ..< mapHeight:
      terrain[x][y] = Empty
  
  # Generate terrain features
  terrain.generateRiver(mapWidth, mapHeight, mapBorder, r)
  terrain.generateWheatFields(mapWidth, mapHeight, mapBorder, r)
  terrain.generateTrees(mapWidth, mapHeight, mapBorder, r)

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
                   r: var Rand, maxAttempts = 100, preferCorners = false, 
                   excludedCorners: seq[int] = @[]): PlacementResult =
  ## Find a suitable location for any structure
  ## If preferCorners is true, tries corner locations first (for houses)
  ## excludedCorners: list of corner indices (0-3) to skip
  
  # Calculate search bounds
  let minX = mapBorder + (if structure.needsBuffer: structure.bufferSize else: 0)
  let maxX = mapWidth - mapBorder - structure.width - (if structure.needsBuffer: structure.bufferSize else: 0)
  let minY = mapBorder + (if structure.needsBuffer: structure.bufferSize else: 0)
  let maxY = mapHeight - mapBorder - structure.height - (if structure.needsBuffer: structure.bufferSize else: 0)
  
  if maxX <= minX or maxY <= minY:
    return PlacementResult(success: false, message: "Map too small for structure", cornerUsed: -1)
  
  # If preferCorners, try corner regions first
  if preferCorners:
    # Define corner regions (25% of map from each corner)
    let cornerSize = min(mapWidth, mapHeight) div 4
    
    # Define the 4 corner regions with indices for tracking
    var cornerRegions: seq[tuple[id: int, minX, maxX, minY, maxY: int]] = @[]
    
    # Corner 0: Top-left
    cornerRegions.add((0, minX, min(minX + cornerSize, maxX), 
                       minY, min(minY + cornerSize, maxY)))
    # Corner 1: Top-right
    cornerRegions.add((1, max(maxX - cornerSize, minX), maxX,
                       minY, min(minY + cornerSize, maxY)))
    # Corner 2: Bottom-left
    cornerRegions.add((2, minX, min(minX + cornerSize, maxX),
                       max(maxY - cornerSize, minY), maxY))
    # Corner 3: Bottom-right
    cornerRegions.add((3, max(maxX - cornerSize, minX), maxX,
                       max(maxY - cornerSize, minY), maxY))
    
    # Filter out excluded corners
    var availableCorners: seq[tuple[id: int, minX, maxX, minY, maxY: int]] = @[]
    for corner in cornerRegions:
      if corner.id notin excludedCorners:
        availableCorners.add(corner)
    
    # Shuffle available corners for variety
    for i in countdown(availableCorners.len - 1, 1):
      let j = r.rand(0 .. i)
      swap(availableCorners[i], availableCorners[j])
    
    # Try each available corner region
    for region in availableCorners:
      for attempt in 0 ..< maxAttempts div 4:  # Fewer attempts per corner
        let x = r.rand(region.minX ..< region.maxX)
        let y = r.rand(region.minY ..< region.maxY)
        let pos = ivec2(x.int32, y.int32)
        
        if canPlaceAt(grid, terrain, pos, structure, mapWidth, mapHeight):
          return PlacementResult(success: true, position: pos, cornerUsed: region.id)
  
  # Try random placement (original behavior)
  for attempt in 0 ..< maxAttempts:
    let x = r.rand(minX ..< maxX)
    let y = r.rand(minY ..< maxY)
    let pos = ivec2(x.int32, y.int32)
    
    if canPlaceAt(grid, terrain, pos, structure, mapWidth, mapHeight):
      return PlacementResult(success: true, position: pos, cornerUsed: -1)
  
  # Fall back to systematic search
  for y in minY ..< maxY:
    for x in minX ..< maxX:
      let pos = ivec2(x.int32, y.int32)
      if canPlaceAt(grid, terrain, pos, structure, mapWidth, mapHeight):
        return PlacementResult(success: true, position: pos, cornerUsed: -1)
  
  return PlacementResult(success: false, message: "No valid location found", cornerUsed: -1)

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
          of 'a', 's', '*', 'A', 'F', 'C', 'W': result.special.add(worldPos)  # Various special tiles including corner buildings
          else: discard

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