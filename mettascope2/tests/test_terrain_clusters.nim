import ../src/mettascope/tribal
import std/strformat
import vmath

# Create a new environment
let env = newEnvironment()

# Print the map with terrain
echo "Village Map with Clustered Terrain:"
echo "===================================="
echo env.render()

# Print legend
echo "\nLegend:"
echo "  ~ = Water (passable, might be slower in future)"
echo "  . = Wheat field (clustered in 4-5 fields)"
echo "  T = Tree (clustered in 4-5 groves)"
echo "  # = Wall"
echo "  A = Agent"
echo "  g = Generator"
echo "  c = Converter"
echo "  a = Altar (in houses)"
echo "  (space) = Empty ground"

# Print terrain statistics
var waterCount, wheatCount, treeCount, emptyCount = 0
var wheatClusters = 0
var treeClusters = 0
var inWheatCluster = false
var inTreeCluster = false

for x in 0 ..< MapWidth:
  for y in 0 ..< MapHeight:
    case env.terrain[x][y]:
    of Water: inc waterCount
    of Wheat: 
      inc wheatCount
      if not inWheatCluster:
        inc wheatClusters
        inWheatCluster = true
    of Tree: 
      inc treeCount
      if not inTreeCluster:
        inc treeClusters
        inTreeCluster = true
    of Empty: 
      inc emptyCount
      inWheatCluster = false
      inTreeCluster = false

echo fmt"\nTerrain Statistics:"
echo fmt"  Water tiles: {waterCount} (passable)"
echo fmt"  Wheat tiles: {wheatCount} (in clusters)"
echo fmt"  Tree tiles: {treeCount} (in groves)"
echo fmt"  Empty tiles: {emptyCount}"
echo fmt"  Total tiles: {MapWidth * MapHeight}"

# Analyze object placement
var objectsOnWater = 0
var objectsOnWheat = 0
var objectsOnTrees = 0
var objectsOnEmpty = 0

for thing in env.things:
  case env.terrain[thing.pos.x.int][thing.pos.y.int]:
  of Water: inc objectsOnWater
  of Wheat: inc objectsOnWheat
  of Tree: inc objectsOnTrees
  of Empty: inc objectsOnEmpty

echo fmt"\nObject Placement:"
echo fmt"  Objects on water: {objectsOnWater} (now allowed)"
echo fmt"  Objects on wheat: {objectsOnWheat}"
echo fmt"  Objects on trees: {objectsOnTrees}"
echo fmt"  Objects on empty: {objectsOnEmpty}"
echo fmt"  Total objects: {env.things.len}"