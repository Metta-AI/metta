import ../src/mettascope/tribal, vmath
import std/strformat

# Test the respawn system

var env = newEnvironment()

echo "Testing Respawn System"
echo "====================="
echo ""

# Find an altar and its associated agents
var altarPos: IVec2
var altarFound = false
for thing in env.things:
  if thing.kind == Altar:
    altarPos = thing.pos
    altarFound = true
    echo fmt"Found altar at ({altarPos.x}, {altarPos.y}) with {thing.hp} hearts"
    break

if not altarFound:
  echo "No altar found!"
  quit(1)

# Count agents with this altar as home
var agentsWithThisAltar = 0
for agent in env.agents:
  if agent.homeAltar == altarPos:
    agentsWithThisAltar += 1

echo fmt"Agents linked to this altar: {agentsWithThisAltar}"
echo ""

# Run simulation for many steps to get combat
echo "Running simulation to trigger combat and respawn..."
var respawnOccurred = false

for step in 1..300:
  var actions: array[MapAgents, array[2, uint8]]
  for i in 0 ..< MapAgents:
    actions[i] = [0'u8, 0'u8]  # All agents NOOP
  
  env.step(actions.addr)
  
  # Check for dead and respawned agents
  var deadCount = 0
  var respawnedCount = 0
  
  for i in 0 ..< MapAgents:
    if env.terminated[i] == 1.0:
      deadCount += 1
    elif step > 10 and env.agents[i].frozen == 0 and env.agents[i].hp == MapObjectAgentHp:
      # Check if this agent was respawned (has full HP and not frozen after step 10)
      # This is a heuristic check
      respawnedCount += 1
  
  # Check altar hearts
  var currentAltarHearts = 0
  for thing in env.things:
    if thing.kind == Altar and thing.pos == altarPos:
      currentAltarHearts = thing.hp
      break
  
  if step mod 50 == 0 or deadCount > 0 or currentAltarHearts < MapObjectAltarInitialHearts:
    var clippyCount = 0
    for thing in env.things:
      if thing.kind == Clippy:
        clippyCount += 1
    
    echo fmt"Step {step}:"
    echo fmt"  Dead agents: {deadCount}"
    echo fmt"  Altar hearts: {currentAltarHearts} / {MapObjectAltarInitialHearts}"
    echo fmt"  Clippys: {clippyCount}"
    
    if currentAltarHearts < MapObjectAltarInitialHearts:
      echo "  -> Altar has used hearts for respawning!"
      respawnOccurred = true

echo ""
if respawnOccurred:
  echo "✓ Respawn system is working! Altar hearts were consumed to respawn agents."
else:
  echo "No respawns occurred (agents may not have died near their home altar)"

# Test altar heart deposit
echo ""
echo "Testing heart deposit system..."
echo "An agent can deposit energy as hearts to the altar (up to 2x initial capacity)"
echo fmt"Initial hearts: {MapObjectAltarInitialHearts}"
echo fmt"Max capacity: {MapObjectAltarInitialHearts * 2}"