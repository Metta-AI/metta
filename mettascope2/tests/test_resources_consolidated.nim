## Consolidated resource system tests
import test_utils

echo "============================"
echo "Resource System Tests"
echo "============================"
echo ""

# Test 1: Mine usage
proc testMineUsage() =
  echo "Test 1: Mine System"
  echo "-------------------"
  var env = newEnvironment()
  
  # Find a mine
  for thing in env.things:
    if thing.kind == Mine:
      echo fmt"Mine at ({thing.pos.x}, {thing.pos.y})"
      echo fmt"  Resources: {thing.inputResource}"
      echo fmt"  Cooldown: {MapObjectMineCooldown} steps"
      echo "  Agent can mine with action 3 (use) to get ore"
      break
  echo ""

# Test 2: Generator usage
proc testGeneratorUsage() =
  echo "Test 2: Generator System"
  echo "------------------------"
  var env = newEnvironment()
  
  for thing in env.things:
    if thing.kind == Generator:
      echo fmt"Generator at ({thing.pos.x}, {thing.pos.y})"
      echo fmt"  Converts: 1 ore → {MapObjectGeneratorEnergyOutput} energy"
      echo fmt"  Cooldown: {MapObjectGeneratorCooldown} steps"
      echo "  Agent uses with action 3 while carrying ore"
      break
  echo ""

# Test 3: Altar hearts system
proc testAltarHearts() =
  echo "Test 3: Altar Hearts System"
  echo "---------------------------"
  var env = newEnvironment()
  
  for thing in env.things:
    if thing.kind == Altar:
      echo fmt"Altar at ({thing.pos.x}, {thing.pos.y})"
      echo fmt"  Current hearts: {thing.hp}/{MapObjectAltarInitialHearts * 2} (max)"
      echo fmt"  Deposit cost: {MapObjectAltarUseCost} energy → 1 heart"
      echo fmt"  Respawn cost: {MapObjectAltarRespawnCost} heart"
      echo "  Clippys remove 1 heart when they reach altar"
      break
  echo ""

# Test 4: Resource gathering from terrain
proc testTerrainResources() =
  echo "Test 4: Terrain Resource Gathering"
  echo "----------------------------------"
  echo "Agents can gather from terrain with action 5:"
  echo fmt"  Water tiles → water inventory (max 5)"
  echo fmt"  Wheat tiles → wheat inventory (max 5)"
  echo fmt"  Tree tiles → wood inventory (max 5)"
  echo "Gathering destroys the terrain tile (becomes empty)"
  echo ""

# Test 5: Full resource cycle
proc testResourceCycle() =
  echo "Test 5: Full Resource Cycle Simulation"
  echo "---------------------------------------"
  var env = newEnvironment()
  
  echo "Resource flow:"
  echo "1. Mine ore from mines"
  echo "2. Convert ore to energy at generators"
  echo "3. Deposit energy as hearts at altar"
  echo "4. Hearts used to respawn dead agents"
  echo ""
  
  # Run simulation
  env.runSteps(100)
  
  # Check if any resources were used
  var minesUsed = false
  var generatorsUsed = false
  
  for thing in env.things:
    case thing.kind:
    of Mine:
      if thing.inputResource < MapObjectMineInitialResources:
        minesUsed = true
    of Generator:
      if thing.cooldown > 0:
        generatorsUsed = true
    else: discard
  
  if minesUsed:
    echo "✓ Mines have been used"
  if generatorsUsed:
    echo "✓ Generators have been used"
  
  let hearts = env.getTotalAltarHearts()
  if hearts != MapObjectAltarInitialHearts * 3:  # 3 villages
    echo fmt"✓ Altar hearts changed: {hearts} (from {MapObjectAltarInitialHearts * 3})"
  echo ""

# Test 6: Inventory management
proc testInventoryManagement() =
  echo "Test 6: Agent Inventory"
  echo "-----------------------"
  var env = newEnvironment()
  
  echo "Agent inventory slots:"
  echo fmt"  Slot 1: Ore (from mines, max {MapObjectAgentMaxInventory})"
  echo fmt"  Slot 2: Water (from terrain, max 5)"
  echo fmt"  Slot 3: Wheat (from terrain, max 5)"
  echo fmt"  Slot 4: Wood (from terrain, max 5)"
  echo ""
  echo "Agents start with:"
  echo fmt"  Energy: {MapObjectAgentInitialEnergy}/{MapObjectAgentMaxEnergy}"
  echo fmt"  All inventory slots: 0"
  echo ""

# Run all tests
testMineUsage()
testGeneratorUsage()
testAltarHearts()
testTerrainResources()
testResourceCycle()
testInventoryManagement()

echo "Resource tests complete!"