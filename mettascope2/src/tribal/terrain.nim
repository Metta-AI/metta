import std/[random, math], vmath, common

# TerrainType, TerrainGrid, and toIVec2 are now imported from common.nim

proc generateRiver*(terrain: var TerrainGrid, mapWidth, mapHeight, mapBorder: int, r: var Rand) =
  ## Generate a river that flows from left to right across the map
  const riverWidth = 4
  
  var riverPath: seq[IVec2] = @[]
  
  # Always start on the left edge, randomly positioned vertically
  var currentPos = toIVec2(mapBorder, r.rand(mapBorder + riverWidth .. mapHeight - mapBorder - riverWidth))
  var primaryDirection = toIVec2(1, 0) # Always flow right
  
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