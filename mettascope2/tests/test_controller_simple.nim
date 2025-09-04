import std/[strformat]
import vmath
import ../src/mettascope/tribal
import ../src/mettascope/controller

proc main() =
  echo "Testing simple controller movement..."
  
  # Create environment
  var env = newEnvironment()
  
  # Create controller
  var controller = newController(seed = 2024)
  
  # Test for just 5 steps with debug output
  for step in 0 ..< 5:
    echo fmt"\n=== Step {step} ==="
    
    # Create actions array for all agents
    var actions: array[MapAgents, array[2, uint8]]
    
    # Get action from controller for first agent only for debugging
    if MapAgents > 0:
      let agent = env.agents[0]
      echo fmt"Agent 0 before: pos=({agent.pos.x},{agent.pos.y}), frozen={agent.frozen}, ore={agent.inventory}, energy={agent.energy}"
      
      actions[0] = controller.decideAction(env, 0)
      echo fmt"Agent 0 action: [{actions[0][0]}, {actions[0][1]}]"
      
      # Decode action
      case actions[0][0]:
      of 0: echo "  -> NOOP"
      of 1: echo fmt"  -> MOVE (arg={actions[0][1]})"
      of 2: echo fmt"  -> ROTATE to {actions[0][1]}"
      of 3: echo fmt"  -> USE (arg={actions[0][1]})"
      of 4: echo fmt"  -> ATTACK (arg={actions[0][1]})"
      of 5: echo fmt"  -> GET (arg={actions[0][1]})"
      of 6: echo fmt"  -> SHIELD (arg={actions[0][1]})"
      of 7: echo "  -> GIFT"
      of 8: echo fmt"  -> SWAP (arg={actions[0][1]})"
      else: echo "  -> UNKNOWN"
    
    # Get actions for other agents silently
    for agentId in 1 ..< MapAgents:
      actions[agentId] = controller.decideAction(env, agentId)
    
    # Step the environment
    env.step(addr actions)
    
    # Update controller
    controller.updateController()
    
    # Show result for agent 0
    if MapAgents > 0:
      let agent = env.agents[0]
      echo fmt"Agent 0 after: pos=({agent.pos.x},{agent.pos.y}), frozen={agent.frozen}, ore={agent.inventory}, energy={agent.energy}"

when isMainModule:
  main()