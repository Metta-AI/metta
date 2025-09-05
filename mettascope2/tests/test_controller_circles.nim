import ../src/tribal/game
import std/[math, tables, strformat, sequtils, strutils]
import ../src/tribal/controller
import vmath

proc visualizeSpiralOnGrid() =
  echo "\n=== Visual Grid Spiral Test ==="
  echo "Showing the spiral pattern on a grid\n"
  
  var controller = newController(seed = 42)
  let basePos = ivec2(15, 15)  # Center of a 31x31 grid
  
  # Create a simple state to test wander pattern
  var state = ControllerState(
    wanderRadius: 3,  # DEPRECATED
    wanderAngle: 0.0,  # DEPRECATED
    wanderStartAngle: 0.0,  # DEPRECATED
    wanderPointsVisited: 0,  # DEPRECATED
    # New spiral fields
    spiralArcLength: 1,
    spiralStepsInArc: 0,
    spiralDirection: 0,
    spiralArcsCompleted: 0,
    basePosition: basePos,
    hasOre: false,
    hasBattery: false,
    currentTarget: basePos,
    targetType: NoTarget
  )
  
  # Create a grid to visualize the spiral
  var grid = newSeq[seq[char]](31)
  for i in 0..30:
    grid[i] = newSeqWith(31, '.')
  
  # Mark the base position
  grid[basePos.y][basePos.x] = 'X'
  
  # Generate spiral points and mark them on the grid
  var step = 0
  for i in 1..100:
    let point = controller.getNextWanderPoint(state)
    
    # Check if point is within grid bounds
    if point.x >= 0 and point.x < 31 and point.y >= 0 and point.y < 31:
      # Mark the point with a character based on arc number
      let marker = case (state.spiralArcsCompleted - 1) mod 10:
        of 0, 1: '1'
        of 2, 3: '2'
        of 4, 5: '3'
        of 6, 7: '4'
        of 8, 9: '5'
        else: '+'
      
      if grid[point.y][point.x] == '.':
        grid[point.y][point.x] = marker
    
    step += 1
    if step > 100:
      break
  
  # Display the grid
  echo "Grid visualization (X = base, numbers = arc pairs):"
  echo "  ", (0..30).mapIt($(it mod 10)).join("")
  for y, row in grid:
    echo fmt"{y:2} ", row.join("")
  
  echo "\nLegend:"
  echo "  X = Starting/base position"
  echo "  1 = Arc 1-2 (first steps outward)"
  echo "  2 = Arc 3-4 (expanding)"
  echo "  3 = Arc 5-6 (further out)"
  echo "  4 = Arc 7-8 (even further)"
  echo "  5 = Arc 9-10 (outer region)"
  echo "  + = Arc 11+ (far out)"

when isMainModule:
  visualizeSpiralOnGrid()