import std/[strformat, strutils, tables]
import src/tribal/environment
import src/tribal/ai

proc debugAgentRoles() =
  echo "Debug: Agent Role Assignments"
  echo "=" & repeat("=", 35)
  
  var env = newEnvironment()
  let controller = newController(seed = 42)
  
  # Run one step to initialize agent states and roles
  var actions: array[MapAgents, array[2, uint8]]
  for i in 0 ..< env.agents.len:
    actions[i] = controller.decideAction(env, i)
  
  # Show first few agents and their data
  for i in 0..<min(env.agents.len, 15):
    let agent = env.agents[i]
    let state = controller.agentStates[i]
    echo fmt"Agent {i}: homeAltar={agent.homeAltar}, role={state.role}"
    
    if state.role == WeavingLoomSpecialist:
      echo fmt"  → This is a Lantern Specialist!"

when isMainModule:
  debugAgentRoles()