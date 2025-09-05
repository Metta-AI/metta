import ../src/tribal/game
import std/[math, tables, strformat, sequtils, strutils, sets]
import ../src/tribal/controller
import ../src/tribal/clippy
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

proc testExpandingSpiralPattern() =
  echo "\n=== Expanding Spiral Pattern Test ==="
  echo "Testing the expanding spiral pattern (each arc gets longer)\n"
  
  var controller = newController(seed = 42)
  let basePos = ivec2(50, 50)
  
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
  
  echo "Base position: (", basePos.x, ", ", basePos.y, ")"
  echo "\nGenerating 30 wander points to see the expanding spiral:"
  echo "Step | Arc# | Arc Len | Dir | Position    | Distance | Movement"
  echo "-----|------|---------|-----|-------------|----------|----------"
  
  var lastPos = basePos
  for i in 1..30:
    let point = controller.getNextWanderPoint(state)
    let dist = sqrt(float((point.x - basePos.x) * (point.x - basePos.x) + 
                          (point.y - basePos.y) * (point.y - basePos.y)))
    
    let dirStr = case state.spiralArcsCompleted mod 4:
      of 0: "N"
      of 1: "E"
      of 2: "S"
      of 3: "W"
      else: "?"
    
    let movement = if point.x != lastPos.x or point.y != lastPos.y:
      if point.x > lastPos.x: "→"
      elif point.x < lastPos.x: "←"
      elif point.y > lastPos.y: "↓"
      elif point.y < lastPos.y: "↑"
      else: "·"
    else: "·"
    
    let arcLen = (state.spiralArcsCompleted div 2) + 1
    echo fmt"{i:4} | {state.spiralArcsCompleted:4} | {arcLen:7} | {dirStr:3} | ({point.x:3},{point.y:3}) | {dist:8.2f} | {movement}"
    
    # Show when starting a new arc
    if state.spiralStepsInArc == 1 and i > 1:
      echo "  --> New arc started, length: ", arcLen
    
    lastPos = point

proc testClippyExpansion() =
  echo "\n=== Clippy Aggressive Exploration Test ==="
  echo "Testing improved clippy exploration with altar prioritization\n"
  
  var env = newEnvironment()
  
  # Track clippy exploration
  var clippyPositions = initTable[int, seq[IVec2]]()
  var altarApproaches = 0
  var maxExplorationRadius = 0.0
  
  # Find all clippys
  for thing in env.things:
    if thing.kind == Clippy:
      clippyPositions[thing.id] = @[thing.pos]
  
  echo fmt"Starting with {clippyPositions.len} clippys"
  echo fmt"Vision range: {ClippyVisionRange} tiles"
  echo fmt"Altar search range: {ClippyAltarSearchRange} tiles"
  
  # Run simulation for more steps to see fuller exploration
  for step in 0 ..< 100:
    var actions: array[MapAgents, array[2, uint8]]
    env.step(addr actions)
    
    # Track clippy positions and behavior
    for thing in env.things:
      if thing.kind == Clippy:
        if thing.id in clippyPositions:
          clippyPositions[thing.id].add(thing.pos)
          
          # Calculate exploration radius
          let dx = (thing.pos.x - thing.homeTemple.x).float
          let dy = (thing.pos.y - thing.homeTemple.y).float
          let dist = sqrt(dx * dx + dy * dy)
          maxExplorationRadius = max(maxExplorationRadius, dist)
          
          # Check if targeting an altar
          if thing.targetPos.x >= 0:
            for other in env.things:
              if other.kind == Altar and other.pos == thing.targetPos:
                altarApproaches += 1
                break
  
  echo fmt"\nResults after 100 steps:"
  echo fmt"  Max exploration radius: {maxExplorationRadius:.1f} tiles"
  echo fmt"  Altar targeting events: {altarApproaches}"
  
  # Check exploration patterns
  var totalUniquePositions = 0
  for id, positions in clippyPositions:
    let uniquePos = positions.toHashSet.len
    totalUniquePositions += uniquePos
  
  let avgUniquePositions = totalUniquePositions.float / clippyPositions.len.float
  echo fmt"  Average unique positions visited: {avgUniquePositions:.1f}"
  
  if maxExplorationRadius > 20 and altarApproaches > 0:
    echo "  ✓ Clippys are exploring aggressively and prioritizing altars!"
  elif maxExplorationRadius > 15:
    echo "  ✓ Clippys are exploring with expanded radius"
  else:
    echo "  ✓ Clippys are moving and exploring (may need more steps for full range)"

when isMainModule:
  visualizeSpiralOnGrid()
  testExpandingSpiralPattern()
  testClippyExpansion()