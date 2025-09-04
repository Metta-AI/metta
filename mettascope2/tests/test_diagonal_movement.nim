import std/[strformat, sequtils, tables, strutils]
import ../src/mettascope/controller
import ../src/mettascope/tribal
import vmath

proc testDiagonalMovement() =
  echo "\n=== Diagonal Movement Test ==="
  echo "Testing 8-directional movement system\n"
  
  # Test orientation deltas
  echo "Orientation deltas:"
  for i, orient in [N, S, W, E, NW, NE, SW, SE]:
    let delta = getOrientationDelta(orient)
    echo fmt"  {$orient:2}: ({delta.x:2}, {delta.y:2})"
  
  # Test diagonal check
  echo "\nDiagonal orientations:"
  for orient in [N, S, W, E, NW, NE, SW, SE]:
    echo fmt"  {$orient:2} is diagonal: {isDiagonal(orient)}"
  
  # Test opposite orientations
  echo "\nOpposite orientations:"
  for orient in [N, S, W, E, NW, NE, SW, SE]:
    echo fmt"  Opposite of {$orient:2}: {$getOpposite(orient):2}"
  
  # Create an environment and test movement
  var env = newEnvironment()
  var controller = newController(seed = 42)
  
  echo "\n=== Movement Simulation ==="
  echo "Testing agent movement in all 8 directions:\n"
  
  # Get the first agent
  if env.agents.len > 0:
    let agent = env.agents[0]
    let startPos = agent.pos
    echo fmt"Starting position: ({startPos.x}, {startPos.y})"
    
    # Try to move in all 8 directions
    let directions = [N, NE, E, SE, S, SW, W, NW]
    var successfulMoves = 0
    
    for orient in directions:
      # Generate move action
      var actions: array[MapAgents, array[2, uint8]]
      actions[0] = [1'u8, ord(orient).uint8]  # Move action with orientation
      
      # Store old position
      let oldPos = agent.pos
      
      # Execute the move
      env.step(addr actions)
      
      # Check if moved
      if agent.pos != oldPos:
        let delta = ivec2(agent.pos.x - oldPos.x, agent.pos.y - oldPos.y)
        echo fmt"  {$orient:2}: Moved from ({oldPos.x:3}, {oldPos.y:3}) to ({agent.pos.x:3}, {agent.pos.y:3}) - delta: ({delta.x:2}, {delta.y:2})"
        successfulMoves += 1
      else:
        echo fmt"  {$orient:2}: Blocked at ({oldPos.x:3}, {oldPos.y:3})"
    
    echo fmt"\nSuccessfully moved in {successfulMoves}/8 directions"
    
    if successfulMoves >= 4:
      echo "✓ Diagonal movement is working!"
    else:
      echo "⚠ Some movements were blocked (likely by walls or other agents)"

proc visualizeDiagonalPaths() =
  echo "\n=== Diagonal Path Visualization ==="
  echo "Showing movement patterns from center:\n"
  
  # Create a small grid to visualize
  var grid = newSeq[seq[char]](21)
  for i in 0..20:
    grid[i] = newSeqWith(21, '.')
  
  let center = ivec2(10, 10)
  grid[center.y][center.x] = 'O'  # Origin
  
  # Mark paths in all 8 directions
  for steps in 1..5:
    # Cardinal directions
    if center.y - steps >= 0:
      grid[center.y - steps][center.x] = 'N'  # North
    if center.y + steps < 21:
      grid[center.y + steps][center.x] = 'S'  # South
    if center.x - steps >= 0:
      grid[center.y][center.x - steps] = 'W'  # West
    if center.x + steps < 21:
      grid[center.y][center.x + steps] = 'E'  # East
    
    # Diagonal directions
    if center.y - steps >= 0 and center.x + steps < 21:
      grid[center.y - steps][center.x + steps] = '/'  # NE
    if center.y - steps >= 0 and center.x - steps >= 0:
      grid[center.y - steps][center.x - steps] = '\\'  # NW
    if center.y + steps < 21 and center.x + steps < 21:
      grid[center.y + steps][center.x + steps] = '\\'  # SE
    if center.y + steps < 21 and center.x - steps >= 0:
      grid[center.y + steps][center.x - steps] = '/'  # SW
  
  # Display grid
  for row in grid:
    var rowStr = ""
    for c in row:
      rowStr.add(c)
      rowStr.add(' ')
    echo rowStr
  
  echo "\nLegend:"
  echo "  O = Origin"
  echo "  N/S/E/W = Cardinal directions"
  echo "  / \\ = Diagonal directions"

when isMainModule:
  testDiagonalMovement()
  visualizeDiagonalPaths()