import std/[random, math], vmath

proc toIVec2*(x, y: int): IVec2 =
  ## Helper to create IVec2 from ints
  result.x = x.int32
  result.y = y.int32

type
  TerrainType* = enum
    Empty
    Water
    Wheat
    Tree

  TerrainGrid* = array[84, array[48, TerrainType]]  # Width doubled to 84 (80 + 4 border)

proc generateRiver*(terrain: var TerrainGrid, mapWidth, mapHeight, mapBorder: int, r: var Rand) =
  ## Generate a river that starts at one edge, flows through the map, and forks
  const riverWidth = 4
  
  # Choose random starting edge (0=top, 1=right, 2=bottom, 3=left)
  let startEdge = r.rand(0..3)
  var riverPath: seq[IVec2] = @[]
  
  # Set starting position based on edge
  var currentPos: IVec2
  var primaryDirection: IVec2
  
  case startEdge:
  of 0: # Top edge
    currentPos = toIVec2(r.rand(mapBorder + riverWidth .. mapWidth - mapBorder - riverWidth), mapBorder)
    primaryDirection = toIVec2(0, 1) # Flow down
  of 1: # Right edge
    currentPos = toIVec2(mapWidth - mapBorder - 1, r.rand(mapBorder + riverWidth .. mapHeight - mapBorder - riverWidth))
    primaryDirection = toIVec2(-1, 0) # Flow left
  of 2: # Bottom edge
    currentPos = toIVec2(r.rand(mapBorder + riverWidth .. mapWidth - mapBorder - riverWidth), mapHeight - mapBorder - 1)
    primaryDirection = toIVec2(0, -1) # Flow up
  else: # Left edge
    currentPos = toIVec2(mapBorder, r.rand(mapBorder + riverWidth .. mapHeight - mapBorder - riverWidth))
    primaryDirection = toIVec2(1, 0) # Flow right
  
  # Generate main river path
  var hasFork = false
  var forkPoint: IVec2
  var secondaryPath: seq[IVec2] = @[]
  
  while currentPos.x >= mapBorder and currentPos.x < mapWidth - mapBorder and
        currentPos.y >= mapBorder and currentPos.y < mapHeight - mapBorder:
    riverPath.add(currentPos)
    
    # Decide if we should fork (only once, after traveling some distance)
    if not hasFork and riverPath.len > 15 and r.rand(1.0) < 0.3:
      hasFork = true
      forkPoint = currentPos
      
      # Create secondary branch
      var secondaryDirection = if primaryDirection.x != 0:
        toIVec2(primaryDirection.x, r.sample(@[-1, 1]))  # Fork up or down if flowing horizontally
      else:
        toIVec2(r.sample(@[-1, 1]), primaryDirection.y)  # Fork left or right if flowing vertically
      
      var secondaryPos = forkPoint
      for i in 0 ..< 20:
        secondaryPos += secondaryDirection
        # Add some randomness to secondary path
        if r.rand(1.0) < 0.3:
          if secondaryDirection.x != 0:
            secondaryPos.y += r.sample(@[-1, 0, 1]).int32
          else:
            secondaryPos.x += r.sample(@[-1, 0, 1]).int32
        
        if secondaryPos.x >= mapBorder and secondaryPos.x < mapWidth - mapBorder and
           secondaryPos.y >= mapBorder and secondaryPos.y < mapHeight - mapBorder:
          secondaryPath.add(secondaryPos)
        else:
          break
    
    # Move in primary direction with some randomness
    currentPos += primaryDirection
    if r.rand(1.0) < 0.4:
      # Add lateral movement for more natural look
      if primaryDirection.x != 0:
        currentPos.y += r.sample(@[-1, 0, 1]).int32
      else:
        currentPos.x += r.sample(@[-1, 0, 1]).int32
  
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

proc createWheatField*(terrain: var TerrainGrid, centerX, centerY: int, size: int, r: var Rand) =
  ## Create a wheat field cluster around a center point
  let radius = (size.float / 2.0).int
  for dx in -radius .. radius:
    for dy in -radius .. radius:
      let x = centerX + dx
      let y = centerY + dy
      if x >= 0 and x < 84 and y >= 0 and y < 48:
        if terrain[x][y] == Empty:
          # Use distance from center to create more organic shape
          let dist = sqrt((dx * dx + dy * dy).float)
          if dist <= radius.float:
            let chance = 1.0 - (dist / radius.float) * 0.3  # Higher chance near center
            if r.rand(1.0) < chance:
              terrain[x][y] = Wheat

proc createTreeGrove*(terrain: var TerrainGrid, centerX, centerY: int, size: int, r: var Rand) =
  ## Create a tree grove cluster around a center point
  let radius = (size.float / 2.0).int
  for dx in -radius .. radius:
    for dy in -radius .. radius:
      let x = centerX + dx
      let y = centerY + dy
      if x >= 0 and x < 84 and y >= 0 and y < 48:
        if terrain[x][y] == Empty:
          # Use distance from center to create more organic shape
          let dist = sqrt((dx * dx + dy * dy).float)
          if dist <= radius.float:
            let chance = 0.8 - (dist / radius.float) * 0.4  # Trees less dense than wheat
            if r.rand(1.0) < chance:
              terrain[x][y] = Tree

proc generateWheatFields*(terrain: var TerrainGrid, mapWidth, mapHeight, mapBorder: int, r: var Rand) =
  ## Generate 6-8 clustered wheat fields for wider map
  let numFields = r.rand(6..8)
  
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
        let fieldSize = r.rand(5..20)  # Each field has 5-20 wheat tiles
        createWheatField(terrain, x, y, fieldSize, r)
        placed = true
        break
    
    # Fallback: place anywhere if no good spot found
    if not placed:
      let x = r.rand(mapBorder + 3 .. mapWidth - mapBorder - 3)
      let y = r.rand(mapBorder + 3 .. mapHeight - mapBorder - 3)
      let fieldSize = r.rand(5..20)
      createWheatField(terrain, x, y, fieldSize, r)

proc generateTrees*(terrain: var TerrainGrid, mapWidth, mapHeight, mapBorder: int, r: var Rand) =
  ## Generate 6-8 tree groves for wider map
  let numGroves = r.rand(6..8)
  
  for i in 0 ..< numGroves:
    let x = r.rand(mapBorder + 3 .. mapWidth - mapBorder - 3)
    let y = r.rand(mapBorder + 3 .. mapHeight - mapBorder - 3)
    let groveSize = r.rand(5..20)  # Each grove has 5-20 trees
    createTreeGrove(terrain, x, y, groveSize, r)

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