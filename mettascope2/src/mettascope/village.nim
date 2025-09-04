import vmath, std/random, terrain

type
  HouseStructure* = object
    layout*: seq[seq[char]]  # 'a' = altar, '#' = wall, ' ' = empty, '.' = entrance
    width*: int
    height*: int
    centerPos*: IVec2  # Position of the altar within the house

proc createHouse*(): HouseStructure =
  ## Create a house with:
  ## - Altar in the center
  ## - Empty squares around the altar
  ## - Walls forming a ring with entrances at cardinal directions
  result.width = 5
  result.height = 5
  result.centerPos = ivec2(2, 2)
  
  # Initialize the layout
  # '#' = wall, 'a' = altar, ' ' = empty space inside, '.' = entrance
  result.layout = @[
    @['#', '#', '.', '#', '#'],  # Top row with north entrance
    @['#', ' ', ' ', ' ', '#'],  # Second row
    @['.', ' ', 'a', ' ', '.'],  # Middle row with altar and E/W entrances
    @['#', ' ', ' ', ' ', '#'],  # Fourth row
    @['#', '#', '.', '#', '#']   # Bottom row with south entrance
  ]

proc canPlaceHouse*(grid: ptr array[84, array[48, pointer]], terrain: ptr TerrainGrid, 
                    pos: IVec2, house: HouseStructure, 
                    mapWidth, mapHeight: int): bool =
  ## Check if a house can be placed at the given position
  ## pos is the top-left corner of the house
  
  # Check boundaries
  if pos.x < 0 or pos.y < 0:
    return false
  if pos.x + house.width > mapWidth or pos.y + house.height > mapHeight:
    return false
  
  # Check if all positions are empty (nil in grid) and not on water
  for y in 0 ..< house.height:
    for x in 0 ..< house.width:
      let gridX = pos.x + x
      let gridY = pos.y + y
      # Check if there's already something at this position
      if not isNil(grid[gridX][gridY]):
        return false
      # Check if this position is water
      if terrain[gridX][gridY] == Water:
        return false
  
  return true

proc findHouseLocation*(grid: ptr array[84, array[48, pointer]], terrain: ptr TerrainGrid,
                        house: HouseStructure,
                        mapWidth, mapHeight, mapBorder: int, r: var Rand): IVec2 =
  ## Find a suitable location for a house
  ## Returns ivec2(-1, -1) if no location found
  
  # Try random positions
  for attempt in 0 ..< 100:
    let x = r.rand(mapBorder ..< mapWidth - mapBorder - house.width)
    let y = r.rand(mapBorder ..< mapHeight - mapBorder - house.height)
    let pos = ivec2(x.int32, y.int32)
    
    if canPlaceHouse(grid, terrain, pos, house, mapWidth, mapHeight):
      return pos
  
  # If random attempts fail, do a systematic search
  for y in mapBorder ..< mapHeight - mapBorder - house.height:
    for x in mapBorder ..< mapWidth - mapBorder - house.width:
      let pos = ivec2(x.int32, y.int32)
      if canPlaceHouse(grid, terrain, pos, house, mapWidth, mapHeight):
        return pos
  
  return ivec2(-1, -1)  # No valid location found

proc getHouseElements*(house: HouseStructure, topLeft: IVec2): tuple[
  altar: IVec2,
  walls: seq[IVec2],
  entrances: seq[IVec2]
] =
  ## Get the positions of all house elements relative to the map
  ## topLeft is the position of the house's top-left corner on the map
  
  result.walls = @[]
  result.entrances = @[]
  
  for y in 0 ..< house.height:
    for x in 0 ..< house.width:
      let worldPos = topLeft + ivec2(x.int32, y.int32)
      case house.layout[y][x]:
      of 'a':
        result.altar = worldPos
      of '#':
        result.walls.add(worldPos)
      of '.':
        result.entrances.add(worldPos)
      else:
        discard  # Empty space