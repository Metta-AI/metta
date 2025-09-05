## AI Behavior Test Suite
## Tests controller AI for agent decision making and coordination
import std/[strformat, strutils, sets, math, tables]
import vmath
import ../src/tribal/environment
import ../src/tribal/ai

# Test 1: Agent Controller Behavior
proc testAgentController() =
  echo "Test: Agent Controller"
  echo "-" & repeat("-", 40)
  
  var env = newEnvironment()
  let controller = newController(seed = 42)
  
  # Test controller decisions for first few agents
  for i in 0 ..< min(3, env.agents.len):
    let agent = env.agents[i]
    let action = controller.decideAction(env, i)
    
    # Analyze decision based on agent state
    var decision = "unknown"
    if action[0] == 0:
      decision = "noop"
    elif action[0] == 1:
      decision = fmt"move {action[1]}"
    elif action[0] == 3:
      decision = fmt"use {action[1]}"
    
    echo fmt"  Agent {i}: {decision} (ore: {agent.inventoryOre}, battery: {agent.inventoryBattery})"
  
  # Run simulation with controller
  echo "\n  Running 20 steps with controller..."
  var stats = (oreCollected: 0, batteriesCreated: 0, heartsDeposited: 0)
  
  for step in 0 ..< 20:
    var actions: array[MapAgents, array[2, uint8]]
    for i in 0 ..< env.agents.len:
      actions[i] = controller.decideAction(env, i)
    env.step(addr actions)
    controller.updateController()
    
    # Track progress
    for agent in env.agents:
      if agent.inventoryOre > 0:
        stats.oreCollected += 1
      if agent.inventoryBattery > 0:
        stats.batteriesCreated += 1
  
  echo fmt"  Results: {stats.oreCollected} ore collected, {stats.batteriesCreated} batteries created"
  
  if stats.oreCollected > 0 or stats.batteriesCreated > 0:
    echo "  ✓ Controller is making progress"
  else:
    echo "  ⚠ Controller may need more steps to show progress"
  echo ""

# Test 2: Village Agent Coordination
proc testVillageAgents() =
  echo "Test: Village Agent Groups"
  echo "-" & repeat("-", 40)
  
  var env = newEnvironment()
  
  # Group agents by home altar
  var villageGroups = initTable[string, seq[int]]()
  
  for i, agent in env.agents:
    let key = fmt"{agent.homeAltar.x},{agent.homeAltar.y}"
    if key notin villageGroups:
      villageGroups[key] = @[]
    villageGroups[key].add(i)
  
  echo fmt"  Found {villageGroups.len} village groups"
  
  for altar, agentIds in villageGroups:
    if agentIds.len > 0:
      echo fmt"  Village at {altar}: {agentIds.len} agents"
      
      # Check if agents are near their altar
      var nearAltar = 0
      for id in agentIds:
        let agent = env.agents[id]
        let dx = abs(agent.pos.x - agent.homeAltar.x)
        let dy = abs(agent.pos.y - agent.homeAltar.y)
        if dx <= 5 and dy <= 5:
          nearAltar += 1
      
      if nearAltar > 0:
        echo fmt"    {nearAltar}/{agentIds.len} agents near home altar"
  echo ""

# Test 4: Full Simulation Integration
proc testFullSimulation() =
  echo "Test: Full Simulation"
  echo "-" & repeat("-", 40)
  
  var env = newEnvironment()
  var agentController = newController(seed = 42)
  
  echo fmt"  Starting with {env.agents.len} agents, {env.things.len} total entities"
  
  var stats = (
    steps: 0,
    oreCollected: 0,
    batteriesCreated: 0,
    heartsDeposited: 0
  )
  
  # Run simulation
  for i in 0 ..< 50:
    var actions: array[MapAgents, array[2, uint8]]
    for j in 0 ..< env.agents.len:
      actions[j] = agentController.decideAction(env, j)
    env.step(addr actions)
    agentController.updateController()
    stats.steps += 1
    
    # Count resources
    for agent in env.agents:
      stats.oreCollected += agent.inventoryOre
      stats.batteriesCreated += agent.inventoryBattery
    
  
  # Count altar hearts
  for thing in env.things:
    if thing.kind == Altar:
      stats.heartsDeposited += thing.hearts - MapObjectAltarInitialHearts
  
  echo fmt"  After {stats.steps} steps:"
  echo fmt"    Total ore in circulation: {stats.oreCollected}"
  echo fmt"    Total batteries in circulation: {stats.batteriesCreated}"
  echo fmt"    Hearts deposited at altars: {stats.heartsDeposited}"
  
  if stats.oreCollected > 0 or stats.batteriesCreated > 0 or stats.heartsDeposited > 0:
    echo "  ✓ Simulation shows resource flow"
  else:
    echo "  ⚠ Limited activity (may need more steps)"
  echo ""

when isMainModule:
  echo "\n" & "=" & repeat("=", 50)
  echo "AI Behavior Test Suite"
  echo "=" & repeat("=", 50) & "\n"
  
  testAgentController()
  testVillageAgents()
  testFullSimulation()
  
  echo "=" & repeat("=", 50)
  echo "AI behavior tests completed"