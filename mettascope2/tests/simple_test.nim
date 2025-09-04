import ../src/mettascope/tribal

# Create environment and quickly check for temples
var env = newEnvironment()

var templeCount = 0
var clippyCount = 0
var houseCount = 0
var altarCount = 0

for thing in env.things:
  case thing.kind
  of Temple:
    templeCount += 1
  of Clippy:
    clippyCount += 1  
  of Altar:
    altarCount += 1
  else:
    discard

echo "Spawned objects:"
echo "  Houses/Altars: ", altarCount
echo "  Temples: ", templeCount  
echo "  Clippys: ", clippyCount
echo "Expected:"
echo "  Houses: ", MapRoomObjectsHouses
echo "  Temples: ", MapRoomObjectsHouses, " (same as houses)"
echo "  Initial Clippys: ", MapRoomObjectsHouses, " (one per temple)"