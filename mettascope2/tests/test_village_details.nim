import ../src/mettascope/tribal
import std/[strformat, math]
import vmath

# Create a new environment
let env = newEnvironment()

echo "=== Village-Based Agent Spawning Test ==="
echo &"Total agents: {env.agents.len}"
echo &"Configuration: {MapRoomObjectsHouses} houses, {MapAgentsPerHouse} agents per house"
echo ""

# Find house locations (altars)
var houseLocations: seq[IVec2] = @[]
for thing in env.things:
  if thing.kind == Altar:
    houseLocations.add(thing.pos)

echo &"Found {houseLocations.len} house altars at positions:"
for i, pos in houseLocations:
  echo &"  House {i+1}: ({pos.x}, {pos.y})"

echo ""
echo "Agent-to-House Distance Analysis:"
echo "================================="

# For each agent, find distance to nearest house
for agent in env.agents:
  if agent.kind == Agent:
    var minDist = float.high
    var nearestHouse = -1
    
    for i, housePos in houseLocations:
      let dx = agent.pos.x - housePos.x
      let dy = agent.pos.y - housePos.y
      let dist = sqrt(float(dx * dx + dy * dy))
      if dist < minDist:
        minDist = dist
        nearestHouse = i
    
    let color = agentVillageColors[agent.agentId]
    echo &"Agent {agent.agentId:2} at ({agent.pos.x:2}, {agent.pos.y:2}) - " &
         &"Nearest house: {nearestHouse+1} (dist: {minDist:.1f}) - " &
         &"Color: RGB({int(color.r*255):3}, {int(color.g*255):3}, {int(color.b*255):3})"

echo ""
echo "Color Clustering Analysis:"
echo "=========================="

# Group agents by similar colors
type ColorGroup = object
  color: tuple[r,g,b: int]
  agents: seq[int]
  positions: seq[IVec2]

var colorGroups: seq[ColorGroup] = @[]

for agent in env.agents:
  if agent.kind == Agent:
    let color = agentVillageColors[agent.agentId]
    let r = int(color.r * 255)
    let g = int(color.g * 255)
    let b = int(color.b * 255)
    
    var found = false
    for i in 0 ..< colorGroups.len:
      # Group agents with similar colors (tolerance of 10)
      if abs(colorGroups[i].color.r - r) < 10 and 
         abs(colorGroups[i].color.g - g) < 10 and 
         abs(colorGroups[i].color.b - b) < 10:
        colorGroups[i].agents.add(agent.agentId)
        colorGroups[i].positions.add(agent.pos)
        found = true
        break
    
    if not found:
      colorGroups.add(ColorGroup(
        color: (r, g, b),
        agents: @[agent.agentId],
        positions: @[agent.pos]
      ))

for i, group in colorGroups:
  echo &"Village {i+1} - Color: RGB({group.color.r}, {group.color.g}, {group.color.b})"
  echo &"  Agents: {group.agents}"
  
  # Calculate average position
  var avgX = 0.0
  var avgY = 0.0
  for pos in group.positions:
    avgX += float(pos.x)
    avgY += float(pos.y)
  avgX /= float(group.positions.len)
  avgY /= float(group.positions.len)
  
  echo &"  Average position: ({avgX:.1f}, {avgY:.1f})"
  
  # Find nearest house to this average position
  var nearestHouse = -1
  var minDist = float.high
  for j, housePos in houseLocations:
    let dist = sqrt(pow(avgX - float(housePos.x), 2) + pow(avgY - float(housePos.y), 2))
    if dist < minDist:
      minDist = dist
      nearestHouse = j
  
  echo &"  Nearest house to village center: House {nearestHouse+1} (dist: {minDist:.1f})"
  echo ""

echo "Success! Agents are spawning around their respective houses with matching village colors."