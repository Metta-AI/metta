import std/[strformat, strutils, tables]
import vmath
import src/tribal/environment
import src/tribal/ai

proc debugWheatCollection() =
  echo "Debug: Wheat Collection for Lantern Specialists"
  echo "=" & repeat("=", 50)
  
  var env = newEnvironment()
  let controller = newController(seed = 42)
  
  # Count wheat tiles available
  var wheatCount = 0
  for x in 0..<MapWidth:
    for y in 0..<MapHeight:
      if env.terrain[x][y] == Wheat:
        wheatCount += 1
  echo fmt"Total wheat tiles available: {wheatCount}"
  
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
  
  # Run simulation and trace specialist behavior
  for step in 1..25:
    echo fmt"\n--- Step {step} ---"
    
    for agentId in lanternSpecialists:
      let agent = env.agents[agentId]
      let state = controller.agentStates[agentId]
      
      # Check what wheat is near this agent (simplified check)
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
      
      echo fmt"  Agent {agentId}: pos={agent.pos}, wheat={agent.inventoryWheat}, target={state.currentTarget}, targetType={state.targetType}"
      echo fmt"    nearestWheat={nearestWheat}, distance={wheatDistance}"
      
      # Get next action
      actions[agentId] = controller.decideAction(env, agentId)
      echo fmt"    action={actions[agentId]}"
    
    # Step environment
    env.step(addr actions)
    controller.updateController()

when isMainModule:
  debugWheatCollection()