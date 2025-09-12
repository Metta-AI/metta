import std/[strformat, strutils, tables]
import vmath
import src/tribal/environment
import src/tribal/ai

proc debugSpecializationLogic() =
  echo "Debug: Specialization Logic for WeavingLoomSpecialists"
  echo "=" & repeat("=", 55)
  
  var env = newEnvironment()
  let controller = newController(seed = 42)
  
  # Find our lantern specialists
  var lanternSpecialists: seq[int] = @[]
  var actions: array[MapAgents, array[2, uint8]]
  
  # Run one step to initialize
  for i in 0 ..< env.agents.len:
    actions[i] = controller.decideAction(env, i)
    let state = controller.agentStates[i]
    if state.role == WeavingLoomSpecialist:
      lanternSpecialists.add(i)
  
  echo fmt"Found {lanternSpecialists.len} lantern specialists: {lanternSpecialists}"
  
  # Check specialization logic for each agent
  for agentId in lanternSpecialists:
    let agent = env.agents[agentId]
    let state = controller.agentStates[agentId]
    
    echo fmt"\nAgent {agentId}:"
    echo fmt"  role: {state.role}"
    echo fmt"  hasCompletedRole: {state.hasCompletedRole}"
    echo fmt"  condition (role != AltarSpecialist): {state.role != AltarSpecialist}"  
    echo fmt"  condition (not hasCompletedRole): {not state.hasCompletedRole}"
    echo fmt"  specialization active: {state.role != AltarSpecialist and not state.hasCompletedRole}"
    echo fmt"  inventoryWheat: {agent.inventoryWheat}"
    echo fmt"  currentTarget: {state.currentTarget}"
    echo fmt"  targetType: {state.targetType}"

when isMainModule:
  debugSpecializationLogic()