## Test resource gathering with get action
import ../src/mettascope/tribal
import vmath, std/strformat

proc testResourceGathering() =
  echo "=== Testing Resource Gathering System ==="
  echo ""
  
  # Create environment
  let env = newEnvironment()
  
  # Find an agent and nearby resources
  if env.agents.len > 0:
    let agent = env.agents[0]
    echo fmt"Agent 0 starting position: ({agent.pos.x}, {agent.pos.y})"
    echo fmt"Starting inventory: ore={agent.inventory}, water={agent.inventoryWater}, wheat={agent.inventoryWheat}, wood={agent.inventoryWood}"
    echo ""
    
    # Look for nearby terrain features
    echo "Scanning for resources near agent..."
    var foundResources = false
    
    for dy in -3..3:
      for dx in -3..3:
        let x = agent.pos.x + dx
        let y = agent.pos.y + dy
        if x >= 0 and x < MapWidth and y >= 0 and y < MapHeight:
          case env.terrain[x][y]:
          of Water:
            echo fmt"  Water at ({x}, {y}) - {abs(dx)+abs(dy)} steps away"
            foundResources = true
          of Wheat:
            echo fmt"  Wheat at ({x}, {y}) - {abs(dx)+abs(dy)} steps away"
            foundResources = true
          of Tree:
            echo fmt"  Tree at ({x}, {y}) - {abs(dx)+abs(dy)} steps away"
            foundResources = true
          else:
            discard
    
    if not foundResources:
      echo "  No resources found nearby"
    
    echo ""
    
    # Place a wheat tile next to the agent for testing
    echo "Placing wheat next to agent for testing..."
    let testPos = agent.pos + ivec2(1, 0)  # Place wheat to the east
    if testPos.x < MapWidth and testPos.y < MapHeight:
      env.terrain[testPos.x][testPos.y] = Wheat
      echo fmt"Placed wheat at ({testPos.x}, {testPos.y})"
    
    # Simulate gathering wheat if we can find some
    echo "Attempting to gather resources..."
    
    # Look for wheat next to agent
    for dir in 0..3:
      let orientation = Orientation(dir)
      agent.orientation = orientation
      let targetPos = agent.pos + orientationToVec(orientation)
      
      if targetPos.x >= 0 and targetPos.x < MapWidth and 
         targetPos.y >= 0 and targetPos.y < MapHeight:
        
        let terrainType = env.terrain[targetPos.x][targetPos.y]
        if terrainType != Empty:
          echo fmt"Facing {orientation}: Found {terrainType} at ({targetPos.x}, {targetPos.y})"
          
          # Simulate get action (action 5)
          var actions: array[MapAgents, array[2, uint8]]
          actions[0][0] = 5  # Get action
          actions[0][1] = 0  # Argument
          
          # Execute the action
          env.step(actions.addr)
          
          echo fmt"After gathering:"
          echo fmt"  Inventory: ore={agent.inventory}, water={agent.inventoryWater}, wheat={agent.inventoryWheat}, wood={agent.inventoryWood}"
          echo fmt"  Terrain at target is now: {env.terrain[targetPos.x][targetPos.y]}"
          
          # Try to gather again (should fail if inventory full after 5)
          if agent.inventoryWheat < 5 or agent.inventoryWater < 5 or agent.inventoryWood < 5:
            env.step(actions.addr)
            echo "Tried gathering again (terrain should be empty now)"
          
          break
    
    # Final inventory check
    echo ""
    echo "Final inventory status:"
    echo fmt"  Ore: {agent.inventory}/5"
    echo fmt"  Water: {agent.inventoryWater}/5"
    echo fmt"  Wheat: {agent.inventoryWheat}/5"
    echo fmt"  Wood: {agent.inventoryWood}/5"

testResourceGathering()