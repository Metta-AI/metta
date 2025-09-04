## Test diagonal movement fix for using objects
import std/[strformat]
import ../src/mettascope/tribal
import ../src/mettascope/controller
import vmath

echo "============================"
echo "Diagonal Movement Fix Test"
echo "============================"
echo ""

proc testDiagonalMovement() =
  echo "Testing agent behavior when diagonally adjacent to objects..."
  echo ""
  
  var env = newEnvironment()
  var controller = newController(seed = 2024)
  
  # Find a mine with an empty diagonal position
  var minePos: IVec2
  var diagonalPos: IVec2
  var foundGoodMine = false
  
  for thing in env.things:
    if thing.kind == Mine:
      minePos = thing.pos
      
      # Check all 4 diagonal positions
      let diagonals = @[
        minePos + ivec2(1, 1),   # Southeast
        minePos + ivec2(1, -1),  # Northeast
        minePos + ivec2(-1, 1),  # Southwest
        minePos + ivec2(-1, -1)  # Northwest
      ]
      
      for diagPos in diagonals:
        if diagPos.x >= 0 and diagPos.x < MapWidth and
           diagPos.y >= 0 and diagPos.y < MapHeight and
           env.isEmpty(diagPos):
          diagonalPos = diagPos
          foundGoodMine = true
          break
      
      if foundGoodMine:
        break
  
  if not foundGoodMine:
    echo "No mine with empty diagonal position found"
    return
  
  echo fmt"Found mine at ({minePos.x}, {minePos.y}) with empty diagonal at ({diagonalPos.x}, {diagonalPos.y})"
  
  # Manually position first agent diagonally from mine
  if env.agents.len > 0:
    let agent = env.agents[0]
    
    # Clear old position
    env.grid[agent.pos.x][agent.pos.y] = nil
    
    # Set new position
    agent.pos = diagonalPos
    agent.inventoryOre = 0  # Ensure agent needs to mine
    env.grid[agent.pos.x][agent.pos.y] = agent
        
    echo fmt"Positioned agent at ({agent.pos.x}, {agent.pos.y}) - diagonally from mine"
    
    # Get action from controller
    let action = controller.decideAction(env, 0)
    
    echo fmt"Controller decision: action=[{action[0]}, {action[1]}]"
    
    if action[0] == 1:  # Movement action
      case action[1]:
      of 0: echo "  Moving North to get cardinally adjacent"
      of 1: echo "  Moving South to get cardinally adjacent"
      of 2: echo "  Moving East to get cardinally adjacent"
      of 3: echo "  Moving West to get cardinally adjacent"
      else: echo "  Invalid movement"
      
      # Execute the move
      var actionsArray: array[MapAgents, array[2, uint8]]
      actionsArray[0] = action
      env.step(addr actionsArray)
      
      echo fmt"Agent now at ({agent.pos.x}, {agent.pos.y})"
      
      # Check if now cardinally adjacent
      let dx = abs(agent.pos.x - minePos.x)
      let dy = abs(agent.pos.y - minePos.y)
      
      if (dx == 1 and dy == 0) or (dx == 0 and dy == 1):
        echo "✓ Agent successfully moved to cardinal position!"
        
        # Get next action - should be use action
        let nextAction = controller.decideAction(env, 0)
        if nextAction[0] == 3:
          echo "✓ Next action is USE - agent can now mine!"
        else:
          echo fmt"Next action: [{nextAction[0]}, {nextAction[1]}]"
      else:
        echo "✗ Agent not in cardinal position"
    elif action[0] == 3:
      echo "✗ Controller tried to USE while diagonal (shouldn't happen)"
    else:
      echo fmt"  Other action type: {action[0]}"
  else:
    echo "No agents found"

# Run the test
testDiagonalMovement()

echo ""
echo "Test complete!"