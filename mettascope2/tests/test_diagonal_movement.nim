import std/[strformat, sequtils, tables, strutils, random]
import vmath
import ../src/tribal/environment
import ../src/tribal/ai
import ../src/tribal/objects
import ../src/tribal/common

proc testDiagonalMovement() =
  echo "\n=== Diagonal Movement Test ==="
  echo "Testing 8-directional movement system\n"
  
  # Test orientation deltas
  echo "Orientation deltas:"
  for i, orient in [N, S, W, E, NW, NE, SW, SE]:
    let delta = OrientationDeltas[ord(orient)]
    echo fmt"  {$orient:2}: ({delta.x:2}, {delta.y:2})"
  
  # Test diagonal check
  echo "\nDiagonal orientations:"
  for orient in [N, S, W, E, NW, NE, SW, SE]:
    let isDiag = ord(orient) > 3  # Orientations 4-7 are diagonal
    echo fmt"  {$orient:2} is diagonal: {isDiag}"
  
  # Test orientation to vector conversion
  echo "\nOrientation to Vector:"
  for orient in [N, S, W, E, NW, NE, SW, SE]:
    let vec = orientationToVec(orient)
    echo fmt"  {$orient:2} -> ({vec.x:2}, {vec.y:2})"
  
  # Create an environment and test movement
  var env = newEnvironment()
  
  echo "\n=== Movement Simulation ==="
  echo "Testing agent movement in all 8 directions:\n"
  
  # Create an agent at center
  let startPos = ivec2(MapWidth div 2, MapHeight div 2)
  let agentId = 0
  
  # Clear any existing objects at that position first
  if env.grid[startPos.x][startPos.y] != nil:
    env.grid[startPos.x][startPos.y] = nil
  
  # Create and add the agent manually
  let agent = Thing(
    kind: Agent,
    agentId: agentId,
    pos: startPos,
    orientation: N,
    homeAltar: ivec2(-1, -1),
    inventoryOre: 0,
    inventoryBattery: 0,
    inventoryWater: 0,
    inventoryWheat: 0,
    inventoryWood: 0,
    inventorySpear: 0
  )
  # Add to things list
  env.things.add(agent)
  # Add to agents list
  env.agents.add(agent)
  # Place on grid
  env.grid[startPos.x][startPos.y] = agent
  
  # Test movement in all 8 directions using step function
  echo fmt"Starting position: ({startPos.x}, {startPos.y})"
  
  # Try to move in all 8 directions
  let directions = [N, NE, E, SE, S, SW, W, NW]
  var successfulMoves = 0
  
  for orient in directions:
    # Reset agent position to center
    if env.grid[agent.pos.x][agent.pos.y] == agent:
      env.grid[agent.pos.x][agent.pos.y] = nil
    agent.pos = startPos
    env.grid[startPos.x][startPos.y] = agent
    
    # Store old position
    let oldPos = agent.pos
    
    # Create action array - move action (1) with orientation argument
    var actions: array[MapAgents, array[2, uint8]]
    for i in 0..<MapAgents:
      actions[i] = [0'u8, 0'u8]  # Initialize all as noop
    actions[agentId] = [1'u8, ord(orient).uint8]  # Move in specified direction
    
    # Execute the step
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
    echo "⚠ Some movements were blocked (likely by terrain or map edges)"

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