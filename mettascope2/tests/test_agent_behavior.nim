import src/mettascope/[tribal, controller, worldmap]
import vmath
import std/[strformat, tables]

proc testAgentBehavior() =
  echo "Testing agent behavior with ore collection..."
  
  # Create environment
  var env = newEnvironment()
  env.reset()
  
  # Create controller
  var controller = newController()
  
  # Find an agent and a mine
  var testAgent: Thing = nil
  var testMine: Thing = nil
  
  for thing in env.things:
    if thing.kind == Agent and testAgent == nil:
      testAgent = thing
    elif thing.kind == Mine and testMine == nil:
      testMine = thing
    
    if testAgent != nil and testMine != nil:
      break
  
  if testAgent == nil or testMine == nil:
    echo "ERROR: Could not find agent or mine in environment"
    return
  
  let agentId = testAgent.agentId
  
  echo &"Initial state:"
  echo &"  Agent at {testAgent.pos}, ore: {testAgent.inventoryOre}, battery: {testAgent.inventoryBattery}"
  echo &"  Mine at {testMine.pos}, resources: {testMine.resources}, cooldown: {testMine.cooldown}"
  
  # Simulate agent getting ore
  testAgent.inventoryOre = 1
  echo &"\nGave agent 1 ore"
  
  # Get action from controller with ore
  let actionWithOre = controller.decideAction(env, agentId)
  echo &"Action with ore: [{actionWithOre[0]}, {actionWithOre[1]}]"
  
  # Check controller state
  if controller.agentStates.hasKey(agentId):
    let state = controller.agentStates[agentId]
    echo &"  Target type: {state.targetType}"
    echo &"  Target position: {state.currentTarget}"
    echo &"  Has ore: {state.hasOre}"
    
    # The agent should be looking for a converter now
    if state.targetType == Converter or state.targetType == Wander:
      echo "✓ PASS: Agent correctly seeks converter or wanders when holding ore"
    else:
      echo "✗ FAIL: Agent should seek converter when holding ore, but target is {state.targetType}"
  
  # Test behavior when agent has no ore
  echo &"\nResetting agent inventory..."
  testAgent.inventoryOre = 0
  
  # Get action from controller without ore
  let actionWithoutOre = controller.decideAction(env, agentId)
  echo &"Action without ore: [{actionWithoutOre[0]}, {actionWithoutOre[1]}]"
  
  if controller.agentStates.hasKey(agentId):
    let state = controller.agentStates[agentId]
    echo &"  Target type: {state.targetType}"
    echo &"  Has ore: {state.hasOre}"
    
    if state.targetType == Mine or state.targetType == Wander:
      echo "✓ PASS: Agent correctly seeks mine when inventory is empty"
    else:
      echo "✗ FAIL: Agent should seek mine when inventory is empty, but target is {state.targetType}"
  
  # Test that agent abandons mine on cooldown
  echo &"\nSetting mine on cooldown..."
  testMine.cooldown = 10
  
  # Force re-evaluation by calling decide action again
  let actionWithMineCooldown = controller.decideAction(env, agentId)
  echo &"Action with mine on cooldown: [{actionWithMineCooldown[0]}, {actionWithMineCooldown[1]}]"
  
  if controller.agentStates.hasKey(agentId):
    let state = controller.agentStates[agentId]
    
    # If the agent was targeting this specific mine, it should abandon it
    if state.currentTarget == testMine.pos:
      echo "✗ FAIL: Agent should not target mine on cooldown"
    else:
      echo "✓ PASS: Agent correctly avoids mine on cooldown"

when isMainModule:
  testAgentBehavior()