## Consolidated village and terrain tests
import std/tables
import test_utils

echo "============================"
echo "Village & Terrain Tests"
echo "============================"
echo ""

# Test 1: Village spawning
proc testVillageSpawning() =
  echo "Test 1: Village Spawning"
  echo "------------------------"
  var env = newEnvironment()
  
  var villageCount = 0
  var agentsWithHomes = 0
  
  for thing in env.things:
    if thing.kind == Altar:
      villageCount += 1
      echo fmt"Village {villageCount}: Altar at ({thing.pos.x}, {thing.pos.y}) with {thing.hp} hearts"
  
  for agent in env.agents:
    if agent.homeAltar.x >= 0:
      agentsWithHomes += 1
  
  echo fmt"Total villages: {villageCount}"
  echo fmt"Agents with home altars: {agentsWithHomes}/{MapAgents}"
  echo ""

# Test 2: Terrain generation
proc testTerrainGeneration() =
  echo "Test 2: Terrain Features"
  echo "------------------------"
  var env = newEnvironment()
  
  var terrainCounts = initTable[TerrainType, int]()
  terrainCounts[Water] = 0
  terrainCounts[Wheat] = 0
  terrainCounts[Tree] = 0
  terrainCounts[Empty] = 0
  
  for x in 0 ..< MapWidth:
    for y in 0 ..< MapHeight:
      terrainCounts[env.terrain[x][y]] += 1
  
  let total = MapWidth * MapHeight
  echo fmt"Map size: {MapWidth}x{MapHeight} = {total} tiles"
  echo fmt"  Water: {terrainCounts[Water]} ({terrainCounts[Water] * 100 div total}%)"
  echo fmt"  Wheat: {terrainCounts[Wheat]} ({terrainCounts[Wheat] * 100 div total}%)"
  echo fmt"  Trees: {terrainCounts[Tree]} ({terrainCounts[Tree] * 100 div total}%)"
  echo fmt"  Empty: {terrainCounts[Empty]} ({terrainCounts[Empty] * 100 div total}%)"
  echo ""

# Test 3: Agent placement around villages
proc testAgentPlacement() =
  echo "Test 3: Agent Placement"
  echo "-----------------------"
  var env = newEnvironment()
  
  # Check distance of agents from their altars
  var distances: seq[int] = @[]
  
  for agent in env.agents:
    if agent.homeAltar.x >= 0:
      let dist = abs(agent.pos.x - agent.homeAltar.x) + abs(agent.pos.y - agent.homeAltar.y)
      distances.add(dist)
  
  if distances.len > 0:
    var minDist = distances[0]
    var maxDist = distances[0]
    var total = 0
    for d in distances:
      if d < minDist: minDist = d
      if d > maxDist: maxDist = d
      total += d
    
    echo fmt"Agent distances from home altar:"
    echo fmt"  Min: {minDist}, Max: {maxDist}, Avg: {total div distances.len}"
  echo ""

# Test 4: Resource placement
proc testResourcePlacement() =
  echo "Test 4: Resource Placement"
  echo "--------------------------"
  var env = newEnvironment()
  
  var mines = 0
  var generators = 0
  var temples = 0
  
  for thing in env.things:
    case thing.kind:
    of Mine: mines += 1
    of Generator: generators += 1
    of Temple: temples += 1
    else: discard
  
  echo fmt"Resources spawned:"
  echo fmt"  Mines: {mines}/{MapRoomObjectsMines}"
  echo fmt"  Generators: {generators}/{MapRoomObjectsGenerators}"
  echo fmt"  Temples: {temples}/{MapRoomObjectsHouses}"
  echo ""

# Test 5: Visual map
proc testVisualMap() =
  echo "Test 5: Map Visualization"
  echo "-------------------------"
  var env = newEnvironment()
  
  echo "Map legend:"
  echo "  A=Agent, a=Altar, t=Temple, C=Clippy"
  echo "  #=Wall, m=Mine, g=Generator"
  echo "  T=Tree, .=Wheat, ~=Water"
  echo ""
  echo "First 20 rows of map:"
  
  let lines = env.render().split('\n')
  for i in 0..min(20, lines.len-1):
    if lines[i].len > 0:
      echo lines[i]
  echo ""

# Run all tests
testVillageSpawning()
testTerrainGeneration()
testAgentPlacement()
testResourcePlacement()
testVisualMap()

echo "Village & terrain tests complete!"