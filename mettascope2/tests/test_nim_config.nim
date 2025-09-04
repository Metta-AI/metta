import ../src/mettascope/tribal
import std/strformat

# Create a new environment
let env = newEnvironment()

# Print the map with terrain
echo "Village Map with Terrain:"
echo "========================="
echo env.render()

# Print legend
echo "\nLegend:"
echo "  ~ = Water (impassable)"
echo "  . = Wheat field"
echo "  T = Tree"
echo "  # = Wall"
echo "  A = Agent"
echo "  g = Generator"
echo "  c = Converter"
echo "  a = Altar"
echo "  (space) = Empty ground"

# Print some statistics
var waterCount, wheatCount, treeCount, emptyCount = 0
for x in 0 ..< MapWidth:
  for y in 0 ..< MapHeight:
    case env.terrain[x][y]:
    of Water: inc waterCount
    of Wheat: inc wheatCount
    of Tree: inc treeCount
    of Empty: inc emptyCount

echo fmt"\nTerrain Statistics:"
echo fmt"  Water tiles: {waterCount}"
echo fmt"  Wheat tiles: {wheatCount}"
echo fmt"  Tree tiles: {treeCount}"
echo fmt"  Empty tiles: {emptyCount}"
echo fmt"  Total tiles: {MapWidth * MapHeight}"