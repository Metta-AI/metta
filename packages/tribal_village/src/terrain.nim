
import std/math, vmath
import rng_compat

type
  TerrainType* = enum
    Empty
    Water
    Wheat
    Tree

  # Backing storage for terrain and placement. Use generous bounds to allow larger maps.
  TerrainGrid* = array[256, array[256, TerrainType]]

  PlacementPriority* = enum
    PriorityRiver = 0      # Rivers always first - they shape the map
    PriorityTerrain = 1    # Wheat fields and trees
    PriorityStructure = 2  # Houses and spawners 
    PriorityObject = 3     # Mines, generators, walls
    PriorityAgent = 4      # Agents placed last
  
  Structure* = object
    width*, height*: int
    centerPos*: IVec2      # Center/important position within structure
    needsBuffer*: bool     # Whether to enforce empty space around it
    bufferSize*: int       # How much buffer space
    layout*: seq[seq[char]] # Optional layout grid for complex structures
  
  PlacementGrid* = ptr array[256, array[256, pointer]]
  
  PlacementResult* = object
    success*: bool
    position*: IVec2
    message*: string
    cornerUsed*: int  # Which corner was used (0-3), or -1 if not a corner

template randInclusive(r: var Rand, a, b: int): int = randIntInclusive(r, a, b)
template randExclusive(r: var Rand, a, b: int): int = randIntExclusive(r, a, b)
template randChance(r: var Rand, p: float): bool = randFloat(r) < p

const
  RiverWidth* = 6

proc inCornerReserve(x, y, mapWidth, mapHeight, mapBorder: int, reserve: int): bool =
  ## Returns true if the coordinate is within a reserved corner area
  let left = mapBorder
  let right = mapWidth - mapBorder
  let top = mapBorder
  let bottom = mapHeight - mapBorder
  let rx = reserve
  let ry = reserve
  let inTopLeft = (x >= left and x < left + rx) and (y >= top and y < top + ry)
  let inTopRight = (x >= right - rx and x < right) and (y >= top and y < top + ry)
  let inBottomLeft = (x >= left and x < left + rx) and (y >= bottom - ry and y < bottom)
  let inBottomRight = (x >= right - rx and x < right) and (y >= bottom - ry and y < bottom)
  inTopLeft or inTopRight or inBottomLeft or inBottomRight



proc toIVec2*(x, y: int): IVec2 =
  result.x = x.int32
  result.y = y.int32

proc checkBounds(x, y, width, height, mapWidth, mapHeight: int): bool =
  x >= 0 and y >= 0 and x + width <= mapWidth and y + height <= mapHeight

proc createStructure*(width, height: int, centerX, centerY: int, 
                     needsBuffer = false, bufferSize = 0): Structure =
  result.width = width
  result.height = height  
  result.centerPos = ivec2(centerX.int32, centerY.int32)
  result.needsBuffer = needsBuffer
  result.bufferSize = bufferSize



proc generateRiver*(terrain: var TerrainGrid, mapWidth, mapHeight, mapBorder: int, r: var Rand) =
  
  var riverPath: seq[IVec2] = @[]
  
  # Reserve corners for villages so river doesn't block them
  let reserve = max(8, min(mapWidth, mapHeight) div 10)
  
  # Start near left edge and centered vertically (avoid corner reserves)
  let centerY = mapHeight div 2
  let span = max(6, mapHeight div 6)
  var startMin = max(mapBorder + RiverWidth + reserve, centerY - span)
  var startMax = min(mapHeight - mapBorder - RiverWidth - reserve, centerY + span)
  if startMin > startMax: swap(startMin, startMax)
  var currentPos = toIVec2(mapBorder, randInclusive(r, startMin, startMax))
  
  var hasFork = false
  var forkPoint: IVec2
  var secondaryPath: seq[IVec2] = @[]
  
  while currentPos.x >= mapBorder and currentPos.x < mapWidth - mapBorder and
        currentPos.y >= mapBorder and currentPos.y < mapHeight - mapBorder:
    riverPath.add(currentPos)
    
    if not hasFork and riverPath.len > max(20, mapWidth div 8) and randChance(r, 0.5):
      hasFork = true
      forkPoint = currentPos
      
      # Choose direction toward nearest vertical edge so branch reaches top/bottom
      let towardTop = int(forkPoint.y) - mapBorder
      let towardBottom = (mapHeight - mapBorder) - int(forkPoint.y)
      let dirY = (if towardTop < towardBottom: -1 else: 1)
      var secondaryDirection = toIVec2(1, dirY)  # Right + up or down
      
      var secondaryPos = forkPoint
      # March until we reach top/bottom bounds (with safety cap)
      let maxSteps = max(mapWidth * 2, mapHeight * 2)
      var steps = 0
      while secondaryPos.y > mapBorder + RiverWidth and secondaryPos.y < mapHeight - mapBorder - RiverWidth and steps < maxSteps:
        secondaryPos.x += 1
        # Bias vertical move strongly toward edge, with small meander
        secondaryPos.y += secondaryDirection.y
        if randChance(r, 0.15):
          secondaryPos.y += sample(r, [-1, 0, 1]).int32
        if secondaryPos.x >= mapBorder and secondaryPos.x < mapWidth - mapBorder and
           secondaryPos.y >= mapBorder and secondaryPos.y < mapHeight - mapBorder:
          if not inCornerReserve(secondaryPos.x, secondaryPos.y, mapWidth, mapHeight, mapBorder, reserve):
            secondaryPath.add(secondaryPos)
        else:
          break
        inc steps
      # Ensure the branch touches the edge vertically with a short vertical run
      var tip = secondaryPos
      var pushSteps = 0
      let maxPush = mapHeight
      if dirY < 0:
        while tip.y > mapBorder and pushSteps < maxPush:
          dec tip.y
          if tip.x >= mapBorder and tip.x < mapWidth - mapBorder and tip.y >= mapBorder and tip.y < mapHeight - mapBorder:
            if not inCornerReserve(tip.x, tip.y, mapWidth, mapHeight, mapBorder, reserve):
              secondaryPath.add(tip)
          inc pushSteps
      else:
        while tip.y < mapHeight - mapBorder and pushSteps < maxPush:
          inc tip.y
          if tip.x >= mapBorder and tip.x < mapWidth - mapBorder and tip.y >= mapBorder and tip.y < mapHeight - mapBorder:
            if not inCornerReserve(tip.x, tip.y, mapWidth, mapHeight, mapBorder, reserve):
              secondaryPath.add(tip)
          inc pushSteps
    
    currentPos.x += 1  # Always move right
    if randChance(r, 0.3):
      currentPos.y += sample(r, [-1, 0, 0, 1]).int32  # Bias towards staying straight
  
  # Place water tiles for main river (skip reserved corners)
  for pos in riverPath:
    for dx in -RiverWidth div 2 .. RiverWidth div 2:
      for dy in -RiverWidth div 2 .. RiverWidth div 2:
        let waterPos = pos + toIVec2(dx, dy)
        if waterPos.x >= 0 and waterPos.x < mapWidth and
           waterPos.y >= 0 and waterPos.y < mapHeight:
          if not inCornerReserve(waterPos.x, waterPos.y, mapWidth, mapHeight, mapBorder, reserve):
            terrain[waterPos.x][waterPos.y] = Water
  
  # Place water tiles for secondary branch (skip reserved corners)
  for pos in secondaryPath:
    for dx in -(RiverWidth div 2 - 1) .. (RiverWidth div 2 - 1):
      for dy in -(RiverWidth div 2 - 1) .. (RiverWidth div 2 - 1):
        let waterPos = pos + toIVec2(dx, dy)
        if waterPos.x >= 0 and waterPos.x < mapWidth and
           waterPos.y >= 0 and waterPos.y < mapHeight:
          if not inCornerReserve(waterPos.x, waterPos.y, mapWidth, mapHeight, mapBorder, reserve):
            terrain[waterPos.x][waterPos.y] = Water

proc createTerrainCluster*(terrain: var TerrainGrid, centerX, centerY: int, size: int, 
                          mapWidth, mapHeight: int, terrainType: TerrainType, 
                          baseDensity: float, falloffRate: float, r: var Rand) =
  ## Create a terrain cluster around a center point with configurable density
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
            let chance = baseDensity - (dist / radius.float) * falloffRate
            if randChance(r, chance):
              terrain[x][y] = terrainType


proc generateWheatFields*(terrain: var TerrainGrid, mapWidth, mapHeight, mapBorder: int, r: var Rand) =
  ## Generate clustered wheat fields; 4x previous count for larger maps
  let numFields = randInclusive(r, 14, 20) * 4
  
  for i in 0 ..< numFields:
    # Try to place near water if possible
    var placed = false
    for attempt in 0 ..< 20:
      let x = randInclusive(r, mapBorder + 3, mapWidth - mapBorder - 3)
      let y = randInclusive(r, mapBorder + 3, mapHeight - mapBorder - 3)
      
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
        let fieldSize = randInclusive(r, 3, 10)  # Each field has 5-20 wheat tiles
        terrain.createTerrainCluster(x, y, fieldSize, mapWidth, mapHeight, Wheat, 1.0, 0.3, r)
        placed = true
        break
    
    # Fallback: place anywhere if no good spot found
    if not placed:
      let x = randInclusive(r, mapBorder + 3, mapWidth - mapBorder - 3)
      let y = randInclusive(r, mapBorder + 3, mapHeight - mapBorder - 3)
      let fieldSize = randInclusive(r, 3, 10)
      terrain.createTerrainCluster(x, y, fieldSize, mapWidth, mapHeight, Wheat, 1.0, 0.3, r)

proc generateTrees*(terrain: var TerrainGrid, mapWidth, mapHeight, mapBorder: int, r: var Rand) =
  ## Generate tree groves; 4x previous count for larger maps
  let numGroves = randInclusive(r, 14, 20) * 4
  
  for i in 0 ..< numGroves:
    let x = randInclusive(r, mapBorder + 3, mapWidth - mapBorder - 3)
    let y = randInclusive(r, mapBorder + 3, mapHeight - mapBorder - 3)
    let groveSize = randInclusive(r, 3, 10)  # Each grove has 3-10 trees
    terrain.createTerrainCluster(x, y, groveSize, mapWidth, mapHeight, Tree, 0.8, 0.4, r)

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
  
  result = @[]
  
  # Start on the left edge but away from corner reserves
  let reserve = max(8, min(mapWidth, mapHeight) div 10)
  let centerY = mapHeight div 2
  let span = max(6, mapHeight div 6)
  var startMin = max(mapBorder + RiverWidth + reserve, centerY - span)
  var startMax = min(mapHeight - mapBorder - RiverWidth - reserve, centerY + span)
  if startMin > startMax: swap(startMin, startMax)
  var currentPos = ivec2(mapBorder.int32, int32(randInclusive(r, startMin, startMax)))
  
  var hasFork = false
  var forkPoint: IVec2
  var secondaryPath: seq[IVec2] = @[]
  
  while currentPos.x >= mapBorder and currentPos.x < mapWidth - mapBorder and
        currentPos.y >= mapBorder and currentPos.y < mapHeight - mapBorder:
    result.add(currentPos)
    
    # Possible fork (scale with map width)
    if not hasFork and result.len > max(20, mapWidth div 8) and randChance(r, 0.5):
      hasFork = true
      forkPoint = currentPos
      
      let towardTop = int(forkPoint.y) - mapBorder
      let towardBottom = (mapHeight - mapBorder) - int(forkPoint.y)
      let dirY = (if towardTop < towardBottom: -1 else: 1)
      var secondaryDirection = toIVec2(1, dirY)
      var secondaryPos = forkPoint
      let maxSteps = max(mapWidth * 2, mapHeight * 2)
      var steps = 0
      while secondaryPos.y > mapBorder + RiverWidth and secondaryPos.y < mapHeight - mapBorder - RiverWidth and steps < maxSteps:
        secondaryPos.x += 1
        secondaryPos.y += secondaryDirection.y
        if randChance(r, 0.15):
          secondaryPos.y += sample(r, [-1, 0, 1]).int32
        if secondaryPos.x >= mapBorder and secondaryPos.x < mapWidth - mapBorder and
           secondaryPos.y >= mapBorder and secondaryPos.y < mapHeight - mapBorder:
          if not inCornerReserve(secondaryPos.x, secondaryPos.y, mapWidth, mapHeight, mapBorder, reserve):
            secondaryPath.add(secondaryPos)
        else:
          break
        inc steps
      # Ensure the branch touches the vertical edge
      var tip = secondaryPos
      var pushSteps = 0
      let maxPush = mapHeight
      if dirY < 0:
        while tip.y > mapBorder and pushSteps < maxPush:
          dec tip.y
          if tip.x >= mapBorder and tip.x < mapWidth - mapBorder and tip.y >= mapBorder and tip.y < mapHeight - mapBorder:
            if not inCornerReserve(tip.x, tip.y, mapWidth, mapHeight, mapBorder, reserve):
              secondaryPath.add(tip)
          inc pushSteps
      else:
        while tip.y < mapHeight - mapBorder and pushSteps < maxPush:
          inc tip.y
          if tip.x >= mapBorder and tip.x < mapWidth - mapBorder and tip.y >= mapBorder and tip.y < mapHeight - mapBorder:
            if not inCornerReserve(tip.x, tip.y, mapWidth, mapHeight, mapBorder, reserve):
              secondaryPath.add(tip)
          inc pushSteps
    
    # Move primarily right with meandering
    currentPos.x += 1
    if randChance(r, 0.3):
      currentPos.y += sample(r, [-1, 0, 0, 1]).int32
  
  # Place water tiles for main river (skip reserved corners)
  for pos in result:
    for dx in -RiverWidth div 2 .. RiverWidth div 2:
      for dy in -RiverWidth div 2 .. RiverWidth div 2:
        let waterX = pos.x + dx
        let waterY = pos.y + dy
        if waterX >= 0 and waterX < mapWidth and waterY >= 0 and waterY < mapHeight:
          if not inCornerReserve(waterX, waterY, mapWidth, mapHeight, mapBorder, reserve):
            terrain[waterX][waterY] = Water
  
  # Place water for secondary branch (skip reserved corners)
  for pos in secondaryPath:
    for dx in -(RiverWidth div 2 - 1) .. (RiverWidth div 2 - 1):
      for dy in -(RiverWidth div 2 - 1) .. (RiverWidth div 2 - 1):
        let waterX = pos.x + dx
        let waterY = pos.y + dy
        if waterX >= 0 and waterX < mapWidth and waterY >= 0 and waterY < mapHeight:
          if not inCornerReserve(waterX, waterY, mapWidth, mapHeight, mapBorder, reserve):
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
            if randChance(r, chance):
              terrain[x][y] = terrainType

proc placeWheatFields*(terrain: var TerrainGrid, mapWidth, mapHeight, mapBorder: int,
                      r: var Rand, numFields = 32) =
  ## Place wheat fields, preferring locations near water
  for i in 0 ..< numFields:
    var placed = false
    
    # Try to place near water
    for attempt in 0 ..< 20:
      let x = randInclusive(r, mapBorder + 3, mapWidth - mapBorder - 3)
      let y = randInclusive(r, mapBorder + 3, mapHeight - mapBorder - 3)
      
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
        let fieldSize = randInclusive(r, 5, 20)
        placeTerrainCluster(terrain, x, y, fieldSize, Wheat, mapWidth, mapHeight, r, 0.9)
        placed = true
        break
    
    # Fallback placement
    if not placed:
      let x = randInclusive(r, mapBorder + 3, mapWidth - mapBorder - 3)
      let y = randInclusive(r, mapBorder + 3, mapHeight - mapBorder - 3)
      let fieldSize = randInclusive(r, 5, 20)
      placeTerrainCluster(terrain, x, y, fieldSize, Wheat, mapWidth, mapHeight, r, 0.9)

proc placeTreeGroves*(terrain: var TerrainGrid, mapWidth, mapHeight, mapBorder: int,
                     r: var Rand, numGroves = 32) =
  ## Place tree groves across the map
  for i in 0 ..< numGroves:
    let x = randInclusive(r, mapBorder + 3, mapWidth - mapBorder - 3)
    let y = randInclusive(r, mapBorder + 3, mapHeight - mapBorder - 3)
    let groveSize = randInclusive(r, 5, 20)
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
    # Define tighter corner regions that scale with map size
    let cornerW = max(10, mapWidth div 8)
    let cornerH = max(10, mapHeight div 8)
    
    # Define the 4 corner regions with indices for tracking
    var cornerRegions: seq[tuple[id: int, minX, maxX, minY, maxY: int]] = @[]
    
    # Corner 0: Top-left
    cornerRegions.add((0, minX, min(minX + cornerW, maxX), 
                       minY, min(minY + cornerH, maxY)))
    # Corner 1: Top-right
    cornerRegions.add((1, max(maxX - cornerW, minX), maxX,
                       minY, min(minY + cornerH, maxY)))
    # Corner 2: Bottom-left
    cornerRegions.add((2, minX, min(minX + cornerW, maxX),
                       max(maxY - cornerH, minY), maxY))
    # Corner 3: Bottom-right
    cornerRegions.add((3, max(maxX - cornerW, minX), maxX,
                       max(maxY - cornerH, minY), maxY))
    
    # Filter out excluded corners
    var availableCorners: seq[tuple[id: int, minX, maxX, minY, maxY: int]] = @[]
    for corner in cornerRegions:
      if corner.id notin excludedCorners:
        availableCorners.add(corner)
    
    # Shuffle available corners for variety
    for i in countdown(availableCorners.len - 1, 1):
      let j = randInclusive(r, 0, i)
      swap(availableCorners[i], availableCorners[j])
    
    # Try each available corner region
    for region in availableCorners:
      # Random attempts first
      for attempt in 0 ..< max(maxAttempts div 3, 50):
        let x = randExclusive(r, region.minX, region.maxX)
        let y = randExclusive(r, region.minY, region.maxY)
        let pos = ivec2(x.int32, y.int32)
        if canPlaceAt(grid, terrain, pos, structure, mapWidth, mapHeight):
          return PlacementResult(success: true, position: pos, cornerUsed: region.id)
      # Systematic sweep inside the corner region before giving up
      for y in region.minY ..< region.maxY:
        for x in region.minX ..< region.maxX:
          let pos = ivec2(x.int32, y.int32)
          if canPlaceAt(grid, terrain, pos, structure, mapWidth, mapHeight):
            return PlacementResult(success: true, position: pos, cornerUsed: region.id)
  
  # Try random placement (original behavior)
  for attempt in 0 ..< maxAttempts:
    let x = randExclusive(r, minX, maxX)
    let y = randExclusive(r, minY, maxY)
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
    let x = randExclusive(r, mapBorder, mapWidth - mapBorder)
    let y = randExclusive(r, mapBorder, mapHeight - mapBorder)
    
    if isNil(grid[x][y]) and terrain[x][y] != Water:
      return ivec2(x.int32, y.int32)
  
  # If we couldn't find anything randomly, give up
  # In production, you might want to do a systematic search
  return ivec2(-1, -1)
