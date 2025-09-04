import strformat
import ../src/mettascope/tribal

# Create environment and check for temples
var env = newEnvironment()

# Count temples and clippys
var templeCount = 0
var clippyCount = 0

for thing in env.things:
  if thing.kind == Temple:
    templeCount += 1
    echo fmt"Temple at position ({thing.pos.x}, {thing.pos.y}), hp: {thing.hp}, cooldown: {thing.cooldown}"
  elif thing.kind == Clippy:
    clippyCount += 1  
    echo fmt"Clippy at position ({thing.pos.x}, {thing.pos.y}), hp: {thing.hp}, energy: {thing.energy}"

echo fmt"Total temples spawned: {templeCount}"
echo fmt"Total clippys spawned: {clippyCount}"
echo fmt"Expected temples: {MapRoomObjectsHouses}"
echo fmt"Expected initial clippys: {MapRoomObjectsHouses} (one per temple)"

# Print first 50 lines of the rendered environment to see map
let rendered = env.render()
var lineCount = 0
for line in rendered.split("\n"):
  if lineCount < 50:
    echo line
    lineCount += 1