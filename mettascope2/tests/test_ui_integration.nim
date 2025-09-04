## Test UI integration and directional movement
import test_utils
import ../src/mettascope/controller

echo "============================"
echo "UI Integration Tests"
echo "============================"
echo ""

# Test 1: Verify action mapping
proc testActionMapping() =
  echo "Test 1: Action Mapping"
  echo "----------------------"
  echo "Movement actions:"
  echo "  Action [1, 0] = Move North (auto-rotate to N)"
  echo "  Action [1, 1] = Move South (auto-rotate to S)"  
  echo "  Action [1, 2] = Move East (auto-rotate to E)"
  echo "  Action [1, 3] = Move West (auto-rotate to W)"
  echo ""
  echo "Use actions (direction-based):"
  echo "  Action [3, 0] = Use North"
  echo "  Action [3, 1] = Use South"
  echo "  Action [3, 2] = Use East"
  echo "  Action [3, 3] = Use West"
  echo ""
  echo "Get actions (harvest terrain):"
  echo "  Action [5, 0] = Get from North"
  echo "  Action [5, 1] = Get from South"
  echo "  Action [5, 2] = Get from East"
  echo "  Action [5, 3] = Get from West"
  echo ""

# Test 2: Controller directional movement
proc testControllerMovement() =
  echo "Test 2: AI Controller Movement"
  echo "------------------------------"
  var env = newEnvironment()
  var controller = newController(seed = 2024)
  
  # Test first 5 agents
  for i in 0 ..< min(5, MapAgents):
    let agent = env.agents[i]
    let action = controller.decideAction(env, i)
    
    echo fmt"Agent {i} at ({agent.pos.x}, {agent.pos.y}):"
    
    if action[0] == 1:  # Movement action
      case action[1]:
      of 0: echo "  Moving North"
      of 1: echo "  Moving South"
      of 2: echo "  Moving East"
      of 3: echo "  Moving West"
      else: echo "  Invalid movement"
    elif action[0] == 3:  # Use action
      case action[1]:
      of 0: echo "  Using object to the North"
      of 1: echo "  Using object to the South"
      of 2: echo "  Using object to the East"
      of 3: echo "  Using object to the West"
      else: echo "  Invalid use direction"
    elif action[0] == 0:
      echo "  No operation (idle)"
    else:
      echo fmt"  Other action: [{action[0]}, {action[1]}]"
  echo ""

# Test 3: Full simulation with directional movement
proc testSimulationFlow() =
  echo "Test 3: Simulation Flow (10 steps)"
  echo "-----------------------------------"
  var env = newEnvironment()
  var controller = newController(seed = 2024)
  var actionsArray: array[MapAgents, array[2, uint8]]
  
  for step in 1..10:
    # Get actions from controller
    for i in 0 ..< MapAgents:
      actionsArray[i] = controller.decideAction(env, i)
    
    # Step the environment
    env.step(addr actionsArray)
    controller.updateController()
    
    if step mod 5 == 0:
      let counts = env.countEntities()
      echo fmt"Step {step}: {counts.agents} agents, {counts.clippys} clippys, {counts.temples} temples"
      
      # Check a sample agent's orientation
      if env.agents.len > 0:
        let agent = env.agents[0]
        echo fmt"  Agent 0: pos ({agent.pos.x}, {agent.pos.y}), facing {agent.orientation}"
  echo ""

# Test 4: Verify auto-rotation
proc testAutoRotation() =
  echo "Test 4: Auto-Rotation on Movement"
  echo "----------------------------------"
  var env = newEnvironment()
  var actionsArray: array[MapAgents, array[2, uint8]]
  
  if env.agents.len > 0:
    let agent = env.agents[0]
    let startPos = agent.pos
    
    echo fmt"Initial: pos ({agent.pos.x}, {agent.pos.y}), facing {agent.orientation}"
    
    # Move North
    actionsArray[0] = [1'u8, 0'u8]
    env.step(addr actionsArray)
    echo fmt"After North: pos ({agent.pos.x}, {agent.pos.y}), facing {agent.orientation}"
    assert agent.orientation == N, "Should face North after moving North"
    
    # Move East
    actionsArray[0] = [1'u8, 2'u8]
    env.step(addr actionsArray)
    echo fmt"After East: pos ({agent.pos.x}, {agent.pos.y}), facing {agent.orientation}"
    assert agent.orientation == E, "Should face East after moving East"
    
    # Move South
    actionsArray[0] = [1'u8, 1'u8]
    env.step(addr actionsArray)
    echo fmt"After South: pos ({agent.pos.x}, {agent.pos.y}), facing {agent.orientation}"
    assert agent.orientation == S, "Should face South after moving South"
    
    # Move West
    actionsArray[0] = [1'u8, 3'u8]
    env.step(addr actionsArray)
    echo fmt"After West: pos ({agent.pos.x}, {agent.pos.y}), facing {agent.orientation}"
    assert agent.orientation == W, "Should face West after moving West"
    
    echo "✓ Auto-rotation working correctly!"
  echo ""

# Run all tests
testActionMapping()
testControllerMovement()
testSimulationFlow()
testAutoRotation()

echo "UI integration tests complete!"