## Compare the complex ai.nim vs simple_ai.nim
import std/[strformat, strutils]
import vmath
import ../src/tribal/environment
import ../src/tribal/ai
import ../src/tribal/simple_ai
import ../src/tribal/external_actions
import test_utils

proc testComplexAI() =
  echo "=== Testing Complex AI (ai.nim) ==="
  var env = setupTestEnvironment()
  setupTestController(seed = 42)
  
  # Find a lantern maker
  var trackedAgent = -1
  for i in 0..<min(env.agents.len, 15):
    if (i mod 5) == 4:  # Agent 4, 9, 14 should be WeavingLoomSpecialists
      trackedAgent = i
      break
  
  if trackedAgent >= 0:
    echo fmt"Tracking WeavingLoomSpecialist Agent {trackedAgent}"
    
    for step in 1..10:
      let agent = env.agents[trackedAgent]
      # Use the existing complex AI
      let action = if globalController != nil and globalController.controllerType == BuiltinAI:
        globalController.aiController.decideAction(env, trackedAgent)
      else:
        [0'u8, 0'u8]
      
      echo fmt"Step {step}: pos={agent.pos}, wheat={agent.inventoryWheat}, lantern={agent.inventoryLantern}, action=[{action[0]}, {action[1]}]"
      
      # Execute step
      var actions: array[MapAgents, array[2, uint8]]
      for i in 0..<MapAgents:
        if i == trackedAgent:
          actions[i] = action
        else:
          actions[i] = [0'u8, 0'u8]
      env.step(addr actions)

proc testSimpleAI() =
  echo "\n=== Testing Simple AI (simple_ai.nim) ==="
  var env = setupTestEnvironment()
  # Don't setup controller - use simple AI directly
  
  # Find a lantern maker (same agent ID as complex test)
  var trackedAgent = -1
  for i in 0..<min(env.agents.len, 15):
    if (i mod 5) == 4:
      trackedAgent = i
      break
  
  if trackedAgent >= 0:
    echo fmt"Tracking LanternMaker Agent {trackedAgent}"
    
    for step in 1..10:
      let agent = env.agents[trackedAgent]
      # Use simple AI
      let action = simpleDecideAction(env, trackedAgent)
      
      echo fmt"Step {step}: pos={agent.pos}, wheat={agent.inventoryWheat}, lantern={agent.inventoryLantern}, action=[{action[0]}, {action[1]}]"
      
      # Execute step
      var actions: array[MapAgents, array[2, uint8]]
      for i in 0..<MapAgents:
        if i == trackedAgent:
          actions[i] = action
        else:
          actions[i] = [0'u8, 0'u8]
      env.step(addr actions)

proc showComplexityComparison() =
  echo "\n=== Complexity Comparison ==="
  echo "Complex AI (ai.nim):"
  echo "  - ~1,200 lines of code"
  echo "  - Complex state management with ControllerState"
  echo "  - Position history tracking"
  echo "  - Escape mode logic"  
  echo "  - Dithering prevention"
  echo "  - Complex pathfinding"
  echo "  - Multiple nested case statements"
  echo ""
  echo "Simple AI (simple_ai.nim):"
  echo "  - ~120 lines of code (10x smaller!)"
  echo "  - Minimal state (just role assignment)"
  echo "  - Direct action selection"
  echo "  - Relies on environment for game mechanics"
  echo "  - Clear, readable logic"
  echo ""
  echo "Both should achieve the same gameplay results!"

when isMainModule:
  showComplexityComparison()
  testComplexAI()
  testSimpleAI()