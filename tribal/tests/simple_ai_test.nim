## Test showing how simple the AI could actually be
import std/[strformat, tables, random, strutils]
import vmath
import ../src/tribal/environment
import ../src/tribal/common
import test_utils

# Simple AI state - just track what each agent should be doing
type
  SimpleRole = enum
    DoNothing, CollectWheatMakeLantern, CollectWoodMakeArmor
  
  SimpleAgentState = object
    role: SimpleRole
    targetPos: IVec2
    targetType: TerrainType  # What resource/building we're seeking

var simpleAgentStates: Table[int, SimpleAgentState]

proc simpleDecideAction(env: Environment, agentId: int): array[2, uint8] =
  let agent = env.agents[agentId]
  
  # Initialize agent state if needed
  if agentId notin simpleAgentStates:
    let role = case agentId mod 3:
      of 0: DoNothing
      of 1: CollectWoodMakeArmor  
      of 2: CollectWheatMakeLantern
      else: DoNothing
    
    simpleAgentStates[agentId] = SimpleAgentState(
      role: role,
      targetPos: ivec2(-1, -1),
      targetType: Empty
    )
  
  var state = simpleAgentStates[agentId]
  
  # Simple role-based logic
  case state.role:
  of CollectWheatMakeLantern:
    if agent.inventoryLantern > 0:
      # Plant lantern at current position (direction 0 = North)
      return [6'u8, 0'u8]  # Plant action
    elif agent.inventoryWheat > 0:
      # Find WeavingLoom and craft
      for thing in env.things:
        if thing.kind == WeavingLoom and abs(thing.pos.x - agent.pos.x) + abs(thing.pos.y - agent.pos.y) <= 1:
          # Adjacent to loom, craft
          let dir = thing.pos - agent.pos
          let orientation = if dir.x > 0: 3 elif dir.x < 0: 2 elif dir.y > 0: 1 else: 0
          return [5'u8, orientation.uint8]  # Put action to craft lantern
        elif abs(thing.pos.x - agent.pos.x) + abs(thing.pos.y - agent.pos.y) < 10:
          # Move toward loom
          let dir = thing.pos - agent.pos
          let moveDir = if abs(dir.x) > abs(dir.y):
            if dir.x > 0: 3 else: 2  # East or West
          else:
            if dir.y > 0: 1 else: 0  # South or North
          return [1'u8, moveDir.uint8]  # Move action
    else:
      # Find wheat and collect
      for x in 0..<MapWidth:
        for y in 0..<MapHeight:
          if env.terrain[x][y] == Wheat:
            let wheatPos = ivec2(x.int32, y.int32)
            let dist = abs(wheatPos.x - agent.pos.x) + abs(wheatPos.y - agent.pos.y)
            if dist == 1:
              # Adjacent to wheat, harvest it
              let dir = wheatPos - agent.pos
              let orientation = if dir.x > 0: 3 elif dir.x < 0: 2 elif dir.y > 0: 1 else: 0
              return [3'u8, orientation.uint8]  # Get action
            elif dist < 15:
              # Move toward wheat
              let dir = wheatPos - agent.pos
              let moveDir = if abs(dir.x) > abs(dir.y):
                if dir.x > 0: 3 else: 2  # East or West
              else:
                if dir.y > 0: 1 else: 0  # South or North
              return [1'u8, moveDir.uint8]  # Move action
  
  of CollectWoodMakeArmor:
    if agent.inventoryWood > 0:
      # Find Armory and craft
      for thing in env.things:
        if thing.kind == Armory:
          let dist = abs(thing.pos.x - agent.pos.x) + abs(thing.pos.y - agent.pos.y)
          if dist == 1:
            # Adjacent, craft armor
            let dir = thing.pos - agent.pos
            let orientation = if dir.x > 0: 3 elif dir.x < 0: 2 elif dir.y > 0: 1 else: 0
            return [5'u8, orientation.uint8]  # Put action
          elif dist < 15:
            # Move toward armory
            let dir = thing.pos - agent.pos
            let moveDir = if abs(dir.x) > abs(dir.y):
              if dir.x > 0: 3 else: 2
            else:
              if dir.y > 0: 1 else: 0
            return [1'u8, moveDir.uint8]
    else:
      # Find wood and collect (same pattern as wheat)
      for x in 0..<MapWidth:
        for y in 0..<MapHeight:
          if env.terrain[x][y] == Tree:
            let treePos = ivec2(x.int32, y.int32)
            let dist = abs(treePos.x - agent.pos.x) + abs(treePos.y - agent.pos.y)
            if dist == 1:
              let dir = treePos - agent.pos
              let orientation = if dir.x > 0: 3 elif dir.x < 0: 2 elif dir.y > 0: 1 else: 0
              return [3'u8, orientation.uint8]  # Get action
            elif dist < 15:
              let dir = treePos - agent.pos
              let moveDir = if abs(dir.x) > abs(dir.y):
                if dir.x > 0: 3 else: 2
              else:
                if dir.y > 0: 1 else: 0
              return [1'u8, moveDir.uint8]
  
  else:
    discard  # Other roles can be implemented similarly
  
  # Default: do nothing
  return [0'u8, 0'u8]

proc testSimpleAI() =
  echo "Testing Simple AI Implementation"
  echo "=" & repeat("=", 35)
  
  var env = setupTestEnvironment()
  setupTestController(seed = 42)
  
  # Override the AI controller's decideAction with our simple version
  # (This is just for testing - would normally replace the whole ai.nim)
  
  # Find a lantern-making agent to track
  var trackedAgent = -1
  for i in 0..<min(env.agents.len, 15):
    if (i mod 3) == 2:  # CollectWheatMakeLantern
      trackedAgent = i
      break
  
  if trackedAgent >= 0:
    echo fmt"Tracking CollectWheatMakeLantern Agent {trackedAgent}"
    
    for step in 1..20:
      # Use simple AI for this agent
      let action = simpleDecideAction(env, trackedAgent)
      let agent = env.agents[trackedAgent]
      
      echo fmt"Step {step}: pos={agent.pos}, wheat={agent.inventoryWheat}, lantern={agent.inventoryLantern}, action=[{action[0]}, {action[1]}]"
      
      # Execute the action (simulated)
      var actions: array[MapAgents, array[2, uint8]]
      for i in 0..<MapAgents:
        if i == trackedAgent:
          actions[i] = action
        else:
          actions[i] = [0'u8, 0'u8]  # Others do nothing
      
      env.step(addr actions)
      
      let newAgent = env.agents[trackedAgent]
      if newAgent.inventoryLantern > agent.inventoryLantern:
        echo fmt"  ✓ Lantern crafted!"
      if newAgent.inventoryWheat > agent.inventoryWheat:
        echo fmt"  ✓ Wheat collected!"

when isMainModule:
  testSimpleAI()