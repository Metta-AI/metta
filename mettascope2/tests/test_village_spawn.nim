import ../src/mettascope/tribal
import std/strformat

# Create a new environment
let env = newEnvironment()

echo "Environment initialized with villages and tribe-colored agents!"
echo &"Total agents: {env.agents.len}"
echo &"Total houses: {MapRoomObjectsHouses}"
echo &"Agents per house: {MapAgentsPerHouse}"
echo ""

# Count agents by their colors to see village groupings
var colorGroups: seq[(int, int, int, int)] = @[]  # (r, g, b, count)

for agent in env.agents:
  if agent.kind == Agent:
    let color = agentVillageColors[agent.agentId]
    let r = int(color.r * 255)
    let g = int(color.g * 255)
    let b = int(color.b * 255)
    
    var found = false
    for i in 0 ..< colorGroups.len:
      if abs(colorGroups[i][0] - r) < 10 and 
         abs(colorGroups[i][1] - g) < 10 and 
         abs(colorGroups[i][2] - b) < 10:
        colorGroups[i] = (colorGroups[i][0], colorGroups[i][1], colorGroups[i][2], colorGroups[i][3] + 1)
        found = true
        break
    
    if not found:
      colorGroups.add((r, g, b, 1))

echo "Village color groups (RGB, agent count):"
for i, group in colorGroups:
  echo &"  Village {i+1}: RGB({group[0]}, {group[1]}, {group[2]}) - {group[3]} agents"

# Print a simple map visualization
echo ""
echo "Map visualization (A=Agent, a=Altar, #=Wall, T=Temple, C=Clippy):"
for y in 0 ..< min(30, MapHeight):
  var line = ""
  for x in 0 ..< min(50, MapWidth):
    if env.grid[x][y] != nil:
      case env.grid[x][y].kind:
      of Agent: line.add("A")
      of Altar: line.add("a")
      of Wall: line.add("#")
      of Temple: line.add("T")
      of Clippy: line.add("C")
      of Generator: line.add("G")
      of Mine: line.add("M")
    else:
      case env.terrain[x][y]:
      of Water: line.add("~")
      of Wheat: line.add(".")
      of Tree: line.add("t")
      of Empty: line.add(" ")
  echo line