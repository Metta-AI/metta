## Compare local controller vs global controller behavior
import std/[strformat, strutils, tables]
import vmath
import ../src/tribal/environment
import ../src/tribal/ai
import ../src/tribal/external_actions
import test_utils

proc compareControllers() =
  echo "Comparing Local Controller vs Global Controller"
  echo "=" & repeat("=", 50)
  
  # Test 1: Local Controller (old method)
  echo "\n--- Test 1: Local Controller (Old Method) ---"
  var env1 = newEnvironment()
  let localController = newController(seed = 42)
  
  # Run one step with local controller
  var actions1: array[MapAgents, array[2, uint8]]
  for i in 0 ..< env1.agents.len:
    actions1[i] = localController.decideAction(env1, i)
  env1.step(addr actions1)
  localController.updateController()
  
  # Check WeavingLoomSpecialists
  var localSpecialists = 0
  for i in 0..<min(env1.agents.len, 15):
    if localController.agentStates.hasKey(i):
      let state = localController.agentStates[i]
      if state.role == WeavingLoomSpecialist:
        localSpecialists += 1
        echo fmt"  Local Agent {i}: role={state.role}, target={state.targetType}"
  
  echo fmt"Local controller found {localSpecialists} WeavingLoomSpecialists"
  
  # Test 2: Global Controller (new method)
  echo "\n--- Test 2: Global Controller (New Method) ---"
  var env2 = setupTestEnvironment()
  setupTestController(seed = 42)
  
  # Run one step with global controller
  stepTestEnvironment(env2)
  
  # Check WeavingLoomSpecialists
  var globalSpecialists = 0
  if globalController != nil and globalController.controllerType == BuiltinAI:
    let controller = globalController.aiController
    for i in 0..<min(env2.agents.len, 15):
      if controller.agentStates.hasKey(i):
        let state = controller.agentStates[i]
        if state.role == WeavingLoomSpecialist:
          globalSpecialists += 1
          echo fmt"  Global Agent {i}: role={state.role}, target={state.targetType}"
  
  echo fmt"Global controller found {globalSpecialists} WeavingLoomSpecialists"
  
  # Compare results
  echo "\n--- Comparison Results ---"
  echo fmt"Local specialists: {localSpecialists}"
  echo fmt"Global specialists: {globalSpecialists}"
  
  if localSpecialists == globalSpecialists and localSpecialists > 0:
    echo "✓ Both controllers produce same specialist count"
  else:
    echo "⚠ Controllers produce different results - need investigation"

when isMainModule:
  compareControllers()