import src/mettascope/[tribal, controller]
import vmath
import std/[strformat, tables, math]

proc testExploration() =
  echo "Testing agent exploration with expanding circles..."
  
  # Create environment
  var env = newEnvironment()
  env.reset()
  
  # Create controller with known seed
  var controller = newController(42)
  
  # Find an agent
  var testAgent: Thing = nil
  for thing in env.things:
    if thing.kind == Agent:
      testAgent = thing
      break
  
  if testAgent == nil:
    echo "ERROR: Could not find agent in environment"
    return
  
  let agentId = testAgent.agentId
  
  # Clear agent's inventory to ensure wandering
  testAgent.inventoryOre = 0
  testAgent.inventoryBattery = 0
  
  # Initialize agent state
  discard controller.decideAction(env, agentId)
  
  if not controller.agentStates.hasKey(agentId):
    echo "ERROR: Agent state not initialized"
    return
  
  let state = controller.agentStates[agentId]
  
  echo &"Initial state:"
  echo &"  Starting position: {testAgent.pos}"
  echo &"  Wander radius: {state.wanderRadius}"
  echo &"  Wander angle: {state.wanderAngle:.2f} rad"
  echo ""
  
  # Track wander positions over multiple steps
  var wanderPositions: seq[IVec2] = @[]
  var radiusHistory: seq[int] = @[]
  
  # Simulate wandering for multiple steps
  for step in 1..24:  # 3 full circles (8 points each)
    # Make sure we stay in wander mode by removing nearby resources
    for thing in env.things:
      if thing.kind in [ThingKind.Mine, ThingKind.Converter] and thing != testAgent:
        let dist = abs(thing.pos.x - testAgent.pos.x) + abs(thing.pos.y - testAgent.pos.y)
        if dist < 10:
          thing.cooldown = 100  # Make resources unavailable
    
    let action = controller.decideAction(env, agentId)
    
    if state.targetType == Wander:
      wanderPositions.add(state.currentTarget)
      radiusHistory.add(state.wanderRadius)
      
      if step mod 8 == 0:
        echo &"After {step} steps (circle {step div 8}):"
        echo &"  Radius: {state.wanderRadius}"
        echo &"  Points visited in circle: {state.wanderPointsVisited}"
        echo &"  Current target: {state.currentTarget}"
        echo ""
  
  # Check that radius increased
  if radiusHistory.len > 0:
    let initialRadius = radiusHistory[0]
    let finalRadius = radiusHistory[^1]
    
    echo &"Radius progression:"
    echo &"  Initial radius: {initialRadius}"
    echo &"  Final radius: {finalRadius}"
    
    if finalRadius > initialRadius:
      echo "✓ PASS: Agent exploration radius expands properly"
    else:
      echo "✗ FAIL: Agent exploration radius did not expand"
  
  # Check that we're making circles
  if wanderPositions.len >= 16:
    echo ""
    echo "Circle pattern verification:"
    var distances: seq[float] = @[]
    for i in 0..7:
      let pos1 = wanderPositions[i]
      let dist = sqrt(((pos1.x - testAgent.pos.x) * (pos1.x - testAgent.pos.x) + 
                      (pos1.y - testAgent.pos.y) * (pos1.y - testAgent.pos.y)).float)
      distances.add(dist)
    
    let avgDist1 = (distances[0] + distances[1] + distances[2] + distances[3]) / 4.0
    
    distances = @[]
    for i in 8..15:
      let pos2 = wanderPositions[i]
      let dist = sqrt(((pos2.x - testAgent.pos.x) * (pos2.x - testAgent.pos.x) + 
                      (pos2.y - testAgent.pos.y) * (pos2.y - testAgent.pos.y)).float)
      distances.add(dist)
    
    let avgDist2 = (distances[0] + distances[1] + distances[2] + distances[3]) / 4.0
    
    echo &"  First circle avg distance: {avgDist1:.1f}"
    echo &"  Second circle avg distance: {avgDist2:.1f}"
    
    if avgDist2 > avgDist1:
      echo "✓ PASS: Circles are expanding outward"
    else:
      echo "✗ FAIL: Circles are not expanding properly"

when isMainModule:
  testExploration()