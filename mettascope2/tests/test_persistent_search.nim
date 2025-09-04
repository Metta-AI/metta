import ../src/mettascope/controller
import ../src/mettascope/tribal
import std/random
import vmath

echo "Testing persistent search behavior"
echo "=================================="

# Create environment and controller
var env = newEnvironment()
var controller = newController(seed = 42)
var r = initRand(42)

# Initialize a simple environment with limited resources far from spawn
env.initSample(r)

# Clear existing mines and add ones far away
env.things.setLen(0)  # Clear all things

# Add an agent at center
let agent = Thing(
  kind: Agent,
  id: 0,
  pos: ivec2(50, 25),  # Center of 100x50 map
  homeAltar: ivec2(50, 25),
  orientation: N,
  inventoryOre: 0,
  inventoryBattery: 0
)
env.add(agent)
env.agents[0] = agent

# Add a mine very far away (near edge of map)
env.add(Thing(
  kind: Mine,
  pos: ivec2(90, 40),  # Far corner
  resources: 10,
  cooldown: 0
))

# Add an altar for battery deposit
env.add(Thing(
  kind: Altar,
  pos: ivec2(50, 25),
  hearts: 5
))

# Add a converter halfway
env.add(Thing(
  kind: Converter,
  pos: ivec2(70, 35),
  cooldown: 0
))

echo "\nInitial setup:"
echo "  Agent at: ", agent.pos
echo "  Mine at: ", ivec2(90, 40), " (distance: ~50)"
echo "  Converter at: ", ivec2(70, 35)
echo ""

# Simulate many steps to see if agent finds the distant mine
var foundMine = false
var usedMine = false
var stepCount = 0
let maxSteps = 200

echo "Simulating agent search (up to ", maxSteps, " steps)..."

while stepCount < maxSteps and not usedMine:
  stepCount += 1
  
  # Get agent decision
  let action = controller.decideAction(env, 0)
  
  # Check agent state
  let state = controller.agentStates[0]
  
  # Apply action
  case action[0]:
  of 0:  # Noop
    discard
  of 1:  # Move
    let orient = Orientation(action[1])
    let delta = case orient:
      of N: ivec2(0, -1)
      of S: ivec2(0, 1)
      of W: ivec2(-1, 0)
      of E: ivec2(1, 0)
      of NW: ivec2(-1, -1)
      of NE: ivec2(1, -1)
      of SW: ivec2(-1, 1)
      of SE: ivec2(1, 1)
    
    let newPos = env.agents[0].pos + delta
    if env.isEmpty(newPos):
      env.agents[0].pos = newPos
      
  of 3:  # Use
    # Check what agent is using
    if env.agents[0].inventoryOre == 0:
      # Must be using mine
      env.agents[0].inventoryOre = 1
      usedMine = true
      echo "  Step ", stepCount, ": Agent mined ore at ", env.agents[0].pos
    
  else:
    discard
  
  # Print progress every 20 steps
  if stepCount mod 20 == 0:
    echo "  Step ", stepCount, ": Agent at ", env.agents[0].pos, 
         ", Target: ", state.targetType, " at ", state.currentTarget,
         ", Spiral arcs: ", state.spiralArcsCompleted
  
  # Check if agent found the mine
  if not foundMine and state.targetType == Mine:
    foundMine = true
    echo "  Step ", stepCount, ": Agent found mine! Target set to ", state.currentTarget

echo "\nResults:"
echo "  Steps taken: ", stepCount
echo "  Found mine: ", foundMine
echo "  Used mine: ", usedMine

if usedMine:
  echo "✓ SUCCESS: Agent persistently searched and found the distant mine!"
else:
  echo "✗ FAILED: Agent gave up before finding the mine"