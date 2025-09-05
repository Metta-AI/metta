import ../src/tribal/controller as ctrl
import ../src/tribal/tribal as tribal
import std/[random, tables]
import vmath

echo "Testing persistent search behavior"
echo "=================================="

# Create environment and controller
var env = newEnvironment()
var controller = ctrl.newController(seed = 42)
var r = initRand(42)

# The environment is automatically initialized by newEnvironment()
# We'll manually add our test setup below

# Clear existing mines and add ones far away
env.things.setLen(0)  # Clear all things

# Add an agent at center
let agent = tribal.Thing(
  kind: tribal.Agent,
  id: 0,
  pos: ivec2(50, 25),  # Center of 100x50 map
  homeAltar: ivec2(50, 25),
  orientation: tribal.N,
  inventoryOre: 0,
  inventoryBattery: 0
)
env.add(agent)
env.agents[0] = agent

# Add a mine very far away (near edge of map)
env.add(tribal.Thing(
  kind: tribal.Mine,
  pos: ivec2(90, 40),  # Far corner
  resources: 10,
  cooldown: 0
))

# Add an altar for battery deposit
env.add(tribal.Thing(
  kind: tribal.Altar,
  pos: ivec2(50, 25),
  hearts: 5
))

# Add a converter halfway
env.add(tribal.Thing(
  kind: tribal.Converter,
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
  let state = if 0 in controller.agentStates: controller.agentStates[0] else: nil
  
  # Apply action
  case action[0]:
  of 0:  # Noop
    discard
  of 1:  # Move
    let orient = tribal.Orientation(action[1])
    let delta = case orient:
      of tribal.N: ivec2(0, -1)
      of tribal.S: ivec2(0, 1)
      of tribal.W: ivec2(-1, 0)
      of tribal.E: ivec2(1, 0)
      of tribal.NW: ivec2(-1, -1)
      of tribal.NE: ivec2(1, -1)
      of tribal.SW: ivec2(-1, 1)
      of tribal.SE: ivec2(1, 1)
    
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
  if not foundMine and state.targetType == ctrl.TargetType.Mine:
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