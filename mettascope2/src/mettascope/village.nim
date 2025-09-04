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
  ## - Four specialized buildings in the absolute corners
  ## - Walls forming a ring with entrances at cardinal directions
  result.width = 5
  result.height = 5
  result.centerPos = ivec2(2, 2)
  
  # Initialize the layout
  # '#' = wall, 'a' = altar, ' ' = empty space inside, '.' = entrance
  # 'A' = Armory, 'F' = Forge, 'C' = Clay Oven, 'W' = Weaving Loom
  result.layout = @[
    @['A', '#', '.', '#', 'F'],  # Top row with Armory (top-left), Forge (top-right)
    @['#', ' ', ' ', ' ', '#'],  # Second row
    @['.', ' ', 'a', ' ', '.'],  # Middle row with altar and E/W entrances
    @['#', ' ', ' ', ' ', '#'],  # Fourth row
    @['C', '#', '.', '#', 'W']   # Bottom row with Clay Oven (bottom-left), Weaving Loom (bottom-right)
  ]

# Placement logic moved to placement.nim for unified handling

# Location finding moved to placement.nim's findPlacement function

proc getHouseElements*(house: HouseStructure, topLeft: IVec2): tuple[
  altar: IVec2,
  walls: seq[IVec2],
  entrances: seq[IVec2],
  armory: IVec2,
  forge: IVec2,
  clayOven: IVec2,
  weavingLoom: IVec2
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
      of 'A':
        result.armory = worldPos
      of 'F':
        result.forge = worldPos
      of 'C':
        result.clayOven = worldPos
      of 'W':
        result.weavingLoom = worldPos
      else:
        discard  # Empty space