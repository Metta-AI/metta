import ../src/mettascope/tribal
import std/strformat
import vmath

# Create a new environment
let env = newEnvironment()

# Count objects on different terrain types
var objectsOnWater = 0
var objectsOnWheat = 0
var objectsOnTrees = 0
var objectsOnEmpty = 0
var borderWallsOnWater = 0
var randomObjectsOnWater = 0

for thing in env.things:
  let terrain = env.terrain[thing.pos.x.int][thing.pos.y.int]
  
  # Check if this is a border wall
  let isBorderWall = thing.kind == Wall and (
    thing.pos.x < MapBorder or 
    thing.pos.x >= MapWidth - MapBorder or
    thing.pos.y < MapBorder or 
    thing.pos.y >= MapHeight - MapBorder
  )
  
  case terrain:
  of Water: 
    inc objectsOnWater
    if isBorderWall:
      inc borderWallsOnWater
    else:
      inc randomObjectsOnWater
  of Wheat: inc objectsOnWheat
  of Tree: inc objectsOnTrees
  of Empty: inc objectsOnEmpty

echo fmt"\nObject Placement Analysis:"
echo fmt"  Total objects on water: {objectsOnWater}"
echo fmt"    - Border walls on water: {borderWallsOnWater} (expected, acts as bridges)"
echo fmt"    - Random objects on water: {randomObjectsOnWater} (should be 0)"
echo fmt"  Objects on wheat: {objectsOnWheat}"
echo fmt"  Objects on trees: {objectsOnTrees}"
echo fmt"  Objects on empty: {objectsOnEmpty}"
echo fmt"  Total objects: {env.things.len}"

if randomObjectsOnWater == 0:
  echo "\n✓ SUCCESS: No random objects spawned on water!"
else:
  echo fmt"\n✗ ISSUE: {randomObjectsOnWater} random objects found on water"
  # Show what types of objects are on water
  for thing in env.things:
    if env.terrain[thing.pos.x.int][thing.pos.y.int] == Water:
      let isBorderWall = thing.kind == Wall and (
        thing.pos.x < MapBorder or 
        thing.pos.x >= MapWidth - MapBorder or
        thing.pos.y < MapBorder or 
        thing.pos.y >= MapHeight - MapBorder
      )
      if not isBorderWall:
        echo fmt"  - {thing.kind} at ({thing.pos.x}, {thing.pos.y})"