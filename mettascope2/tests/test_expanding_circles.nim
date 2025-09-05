import ../src/tribal/game
import std/[math, tables, strformat, sets, strutils, sequtils]
import ../src/tribal/controller
import ../src/tribal/clippy
import ../src/tribal/actions
import vmath

proc visualizeWanderPattern() =
  echo "\n=== Improved Spiral Pattern Test ==="
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
  echo "\nGenerating 50 wander points to see the expanding spiral:"
  echo "Step | Arc# | Arc Len | Dir | Position    | Distance | Movement"
  echo "-----|------|---------|-----|-------------|----------|----------"
  
  var lastPos = basePos
  for i in 1..50:
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
  for step in 0 ..< 150:
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
  
  echo fmt"\nResults after 150 steps:"
  echo fmt"  Max exploration radius: {maxExplorationRadius:.1f} tiles"
  echo fmt"  Altar targeting events: {altarApproaches}"
  
  # Check exploration patterns
  var totalUniquePositions = 0
  for id, positions in clippyPositions:
    let uniquePos = positions.toHashSet.len
    totalUniquePositions += uniquePos
  
  let avgUniquePositions = totalUniquePositions.float / clippyPositions.len.float
  echo fmt"  Average unique positions visited: {avgUniquePositions:.1f}"
  
  if maxExplorationRadius > 30 and altarApproaches > 0:
    echo "  ✓ Clippys are exploring aggressively and prioritizing altars!"
  elif maxExplorationRadius > 20:
    echo "  ✓ Clippys are exploring with expanded radius"
  else:
    echo "  ⚠ May need more simulation steps to see full exploration"

when isMainModule:
  visualizeWanderPattern()
  testClippyExpansion()