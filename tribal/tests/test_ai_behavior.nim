## AI Behavior Test Suite
## Tests controller AI for agent decision making and coordination
import std/[strformat, strutils, tables]
import vmath
import ../src/tribal/environment
import ../src/tribal/ai
import ../src/tribal/common

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

# Test 5: Agent Role Assignments and Specialization
proc testAgentRoleAssignments() =
  echo "Test: Agent Role Assignments"
  echo "-" & repeat("-", 40)
  
  var env = newEnvironment()
  let controller = newController(seed = 42)
  
  # Run one step to initialize agent states and roles
  var actions: array[MapAgents, array[2, uint8]]
  for i in 0 ..< env.agents.len:
    actions[i] = controller.decideAction(env, i)
  
  # Analyze role distribution
  var roleCounts = initTable[AgentRole, int]()
  var lanternSpecialists: seq[int] = @[]
  
  for i in 0..<min(env.agents.len, 15):
    let agent = env.agents[i]
    let state = controller.agentStates[i]
    
    # Count roles
    roleCounts[state.role] = roleCounts.getOrDefault(state.role, 0) + 1
    
    if state.role == WeavingLoomSpecialist:
      lanternSpecialists.add(i)
    
    if i < 5:  # Show first few agents
      echo fmt"  Agent {i}: homeAltar={agent.homeAltar}, role={state.role}"
      if state.role == WeavingLoomSpecialist:
        echo fmt"    → Lantern Specialist! hasCompleted={state.hasCompletedRole}, target={state.currentTarget}"
  
  echo fmt"  Role distribution (first 15 agents):"
  for role, count in roleCounts:
    echo fmt"    {role}: {count} agents"
    
  if lanternSpecialists.len > 0:
    echo fmt"  ✓ Found {lanternSpecialists.len} lantern specialists"
  else:
    echo "  ⚠ No lantern specialists found"
  echo ""

# Test 6: Wheat Collection and Specialization Logic
proc testWheatCollectionSpecialization() =
  echo "Test: Wheat Collection and Specialization"
  echo "-" & repeat("-", 40)
  
  var env = newEnvironment()
  let controller = newController(seed = 42)
  
  # Count wheat tiles available
  var wheatCount = 0
  for x in 0..<MapWidth:
    for y in 0..<MapHeight:
      if env.terrain[x][y] == Wheat:
        wheatCount += 1
  echo fmt"  Total wheat tiles available: {wheatCount}"
  
  # Find lantern specialists and track their behavior
  var actions: array[MapAgents, array[2, uint8]]
  var lanternSpecialists: seq[int] = @[]
  
  # Initialize
  for i in 0 ..< env.agents.len:
    actions[i] = controller.decideAction(env, i)
    let state = controller.agentStates[i]
    if state.role == WeavingLoomSpecialist:
      lanternSpecialists.add(i)
  
  if lanternSpecialists.len == 0:
    echo "  ⚠ No lantern specialists found for wheat collection test"
    echo ""
    return
  
  echo fmt"  Tracking {lanternSpecialists.len} lantern specialists for wheat collection..."
  
  # Track behavior over several steps
  var wheatCollected = 0
  for step in 1..15:
    if step <= 3:  # Show details for first few steps
      echo fmt"  Step {step}:"
    
    for agentId in lanternSpecialists:
      let agent = env.agents[agentId]
      let state = controller.agentStates[agentId]
      
      if agent.inventoryWheat > wheatCollected:
        wheatCollected = agent.inventoryWheat
      
      if step <= 3:
        # Find nearest wheat for context
        var nearestWheat = ivec2(-1, -1)
        var minDist = 999
        for x in 0..<MapWidth:
          for y in 0..<MapHeight:
            if env.terrain[x][y] == Wheat:
              let dist = manhattanDistance(agent.pos, ivec2(x.int32, y.int32))
              if dist < minDist:
                minDist = dist
                nearestWheat = ivec2(x.int32, y.int32)
        let wheatDistance = if nearestWheat.x >= 0: minDist else: -1
        
        echo fmt"    Agent {agentId}: pos={agent.pos}, wheat={agent.inventoryWheat}, target={state.targetType}"
        echo fmt"      nearestWheat={nearestWheat}, distance={wheatDistance}"
      
      # Get next action and step
      actions[agentId] = controller.decideAction(env, agentId)
    
    env.step(addr actions)
    controller.updateController()
  
  echo fmt"  Results after 15 steps:"
  echo fmt"    Max wheat collected by specialists: {wheatCollected}"
  
  if wheatCollected > 0:
    echo "  ✓ Lantern specialists successfully collecting wheat"
  else:
    echo "  ⚠ Specialists may need more time or path optimization"
  echo ""

# Test 7: Agent Spiral Movement Pattern
proc testAgentSpiralMovement() =
  echo "Test: Agent Spiral Movement"
  echo "-" & repeat("-", 40)
  
  var controller = newController(123)
  var state = ControllerState(
    basePosition: ivec2(10, 10),
    currentTarget: ivec2(10, 10),
    targetType: NoTarget
  )
  
  controller.agentStates[0] = state
  
  echo fmt"  Base position: {state.basePosition}"
  echo "  Generated spiral points:"
  
  # Test first 12 spiral points to see the pattern
  var maxDistance = 0
  var validMoves = 0
  
  for i in 1..12:
    let nextPoint = controller.getNextWanderPoint(state)
    let distance = abs(nextPoint.x - state.basePosition.x) + abs(nextPoint.y - state.basePosition.y)
    maxDistance = max(maxDistance, distance)
    
    if nextPoint != state.basePosition:
      validMoves += 1
    
    if i <= 6:  # Show first few moves
      echo fmt"    Step {i}: {nextPoint} (distance: {distance})"
  
  echo fmt"  Spiral pattern analysis:"
  echo fmt"    Valid moves: {validMoves}/12"
  echo fmt"    Max distance from base: {maxDistance}"
  
  if validMoves >= 8 and maxDistance >= 3:
    echo "  ✓ Spiral movement pattern working correctly"
  else:
    echo "  ⚠ Spiral pattern may need adjustment"
  echo ""

when isMainModule:
  echo "\n" & "=" & repeat("=", 50)
  echo "AI Behavior Test Suite"
  echo "=" & repeat("=", 50) & "\n"
  
  testAgentController()
  testVillageAgents()
  testFullSimulation()
  testAgentRoleAssignments()
  testWheatCollectionSpecialization()
  testAgentSpiralMovement()
  
  echo "=" & repeat("=", 50)
  echo "AI behavior tests completed"