import src/mettascope/tribal

# Test directional movement with auto-rotation
proc testDirectionalMovement() =
  echo "Testing directional movement with auto-rotation..."
  
  # Create environment
  var env = newEnvironment()
  
  # Get first agent
  if env.agents.len > 0:
    let agent = env.agents[0]
    let startPos = agent.pos
    let startOrientation = agent.orientation
    
    echo "Agent starting at ", startPos, " facing ", startOrientation
    
    # Test movement North (argument 0)
    var actions: array[MapAgents, array[2, uint8]]
    actions[0] = [1'u8, 0'u8]  # Move North
    env.step(addr actions)
    
    echo "After moving North:"
    echo "  Position: ", agent.pos
    echo "  Orientation: ", agent.orientation
    echo "  Expected: Position Y-1, Orientation N"
    
    # Test movement East (argument 2)
    actions[0] = [1'u8, 2'u8]  # Move East
    env.step(addr actions)
    
    echo "After moving East:"
    echo "  Position: ", agent.pos
    echo "  Orientation: ", agent.orientation
    echo "  Expected: Position X+1, Orientation E"
    
    # Test movement South (argument 1)
    actions[0] = [1'u8, 1'u8]  # Move South
    env.step(addr actions)
    
    echo "After moving South:"
    echo "  Position: ", agent.pos
    echo "  Orientation: ", agent.orientation
    echo "  Expected: Position Y+1, Orientation S"
    
    # Test movement West (argument 3)
    actions[0] = [1'u8, 3'u8]  # Move West
    env.step(addr actions)
    
    echo "After moving West:"
    echo "  Position: ", agent.pos
    echo "  Orientation: ", agent.orientation
    echo "  Expected: Back to start position, Orientation W"
    
    if agent.pos == startPos:
      echo "✓ Agent returned to starting position"
    else:
      echo "✗ Agent did not return to starting position"
  else:
    echo "No agents found in environment"

when isMainModule:
  testDirectionalMovement()