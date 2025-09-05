import std/[strformat, random]
import ../src/tribal/tribal_game
import vmath

echo "\n=== Hunger and Bread System Test ==="
echo "Testing hunger mechanics and bread production\n"

# Create environment with 4 bases
var env = newEnvironment()

# Check that we have 4 houses as configured
var houseCount = 0
var clayOvenCount = 0
for thing in env.things:
  if thing.kind == Altar:
    houseCount += 1
  elif thing.kind == ClayOven:
    clayOvenCount += 1

echo fmt"Houses/Altars spawned: {houseCount}"
echo fmt"Clay Ovens spawned: {clayOvenCount}"
echo ""

# Test hunger system
echo "Testing hunger mechanics:"
if env.agents.len > 0:
  let testAgent = env.agents[0]
  echo fmt"  Agent 0 starting hunger: {testAgent.hunger}"
  echo fmt"  Agent 0 home altar: ({testAgent.homeAltar.x}, {testAgent.homeAltar.y})"
  
  # Simulate some steps to see hunger decrease
  var actions: array[MapAgents, array[2, uint8]]
  for i in 0 ..< MapAgents:
    actions[i] = [0'u8, 0'u8]  # All agents do nothing
  
  # Run 10 steps
  for step in 1 .. 10:
    env.step(addr actions)
    if step mod 5 == 0:
      echo fmt"  After {step} steps - Agent 0 hunger: {testAgent.hunger}"
  
  echo ""
  echo "Testing bread production:"
  
  # Give agent some wheat
  testAgent.inventoryWheat = 3
  echo fmt"  Agent 0 wheat inventory: {testAgent.inventoryWheat}"
  echo fmt"  Agent 0 bread inventory: {testAgent.inventoryBread}"
  
  # Find a clay oven
  var clayOven: Thing = nil
  for thing in env.things:
    if thing.kind == ClayOven:
      clayOven = thing
      break
  
  if clayOven != nil:
    echo fmt"  Found Clay Oven at ({clayOven.pos.x}, {clayOven.pos.y})"
    
    # Move agent to clay oven
    testAgent.pos = clayOven.pos
    env.grid[clayOven.pos.x][clayOven.pos.y] = testAgent
    
    # Use the clay oven to bake bread (action 3 = use)
    actions[0] = [3'u8, 0'u8]  # Use action
    env.step(addr actions)
    
    echo fmt"  After using Clay Oven:"
    echo fmt"    Wheat inventory: {testAgent.inventoryWheat}"
    echo fmt"    Bread inventory: {testAgent.inventoryBread}"
    
    # Test auto-eating when hungry
    testAgent.hunger = 15  # Set low hunger
    echo fmt"\n  Setting hunger to 15 to trigger auto-eat"
    env.step(addr actions)
    echo fmt"  After step with low hunger:"
    echo fmt"    Hunger: {testAgent.hunger}"
    echo fmt"    Bread inventory: {testAgent.inventoryBread}"
  
  # Test death and respawn
  echo ""
  echo "Testing death from starvation:"
  testAgent.hunger = 1
  testAgent.inventoryBread = 0
  let originalPos = testAgent.pos
  echo fmt"  Setting hunger to 1 with no bread"
  echo fmt"  Original position: ({originalPos.x}, {originalPos.y})"
  
  # Find home altar hearts before death
  var altarHeartsBefore = 0
  for thing in env.things:
    if thing.kind == Altar and thing.pos == testAgent.homeAltar:
      altarHeartsBefore = thing.hearts
      break
  
  echo fmt"  Altar hearts before death: {altarHeartsBefore}"
  
  # Step to trigger death
  env.step(addr actions)
  
  echo fmt"  After starvation:"
  echo fmt"    New position: ({testAgent.pos.x}, {testAgent.pos.y})"
  echo fmt"    Hunger reset to: {testAgent.hunger}"
  echo fmt"    Frozen status: {testAgent.frozen}"
  
  # Check altar hearts after respawn
  var altarHeartsAfter = 0
  for thing in env.things:
    if thing.kind == Altar and thing.pos == testAgent.homeAltar:
      altarHeartsAfter = thing.hearts
      break
  
  echo fmt"    Altar hearts after respawn: {altarHeartsAfter}"
  
  if testAgent.pos != originalPos and testAgent.hunger == 100:
    echo "  ✓ Respawn successful!"
  else:
    echo "  ✗ Respawn may have issues"

echo "\nHunger system test complete!"