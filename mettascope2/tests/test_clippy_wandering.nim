import std/[strformat, tables, strutils, math, sets]
import vmath
import ../src/mettascope/tribal
import ../src/mettascope/clippy

proc visualizeClippyMovement(positions: seq[IVec2], templePos: IVec2, width: int = 30, height: int = 20) =
  ## Create ASCII visualization of clippy movement pattern
  var grid = newSeq[string](height)
  for y in 0 ..< height:
    grid[y] = repeat(".", width)
  
  # Mark temple position
  if templePos.x >= 0 and templePos.x < width and 
     templePos.y >= 0 and templePos.y < height:
    grid[templePos.y][templePos.x] = 'T'
  
  # Mark clippy positions with numbers showing order
  for i, pos in positions:
    if pos.x >= 0 and pos.x < width and 
       pos.y >= 0 and pos.y < height:
      let marker = if i < 10: char('0'.ord + i) 
                   elif i < 36: char('a'.ord + i - 10)
                   else: '*'
      grid[pos.y][pos.x] = marker
  
  # Print grid
  echo "\nClippy Movement Pattern (T=Temple, 0-9,a-z=order):"
  echo repeat("-", width + 2)
  for row in grid:
    echo "|" & row & "|"
  echo repeat("-", width + 2)

proc calculateDistance(a, b: IVec2): float =
  let dx = (a.x - b.x).float
  let dy = (a.y - b.y).float
  sqrt(dx * dx + dy * dy)

proc main() =
  echo "Testing Clippy Concentric Circle Wandering..."
  echo repeat("=", 60)
  
  # Initialize environment
  env = newEnvironment()
  
  # Find a temple and its clippy
  var templePos = ivec2(-1, -1)
  var clippyPtr: Thing = nil
  
  for thing in env.things:
    if thing.kind == Temple:
      templePos = thing.pos
      echo fmt"Found temple at ({templePos.x}, {templePos.y})"
      break
  
  # Find the clippy associated with this temple
  for thing in env.things:
    if thing.kind == Clippy and thing.homeTemple == templePos:
      clippyPtr = thing
      echo fmt"Found clippy at ({thing.pos.x}, {thing.pos.y}) for temple"
      break
  
  if clippyPtr == nil:
    echo "No clippy found!"
    return
  
  # Track clippy movement over multiple steps
  var positions: seq[IVec2] = @[clippyPtr.pos]
  var visitedPositions = initHashSet[string]()
  visitedPositions.incl(fmt"{clippyPtr.pos.x},{clippyPtr.pos.y}")
  
  echo "\nSimulating clippy wandering pattern..."
  echo "Step | Position | Distance from Temple | Radius | Angle"
  echo repeat("-", 55)
  
  # Simulate movement for multiple steps
  for step in 0 ..< 30:
    # Get clippy move direction (simulate without agents/altars nearby)
    var emptyThings: seq[pointer] = @[]
    emptyThings.add(cast[pointer](clippyPtr))  # Just add the clippy itself
    
    var rng = initRand(42)  # Deterministic for testing
    let moveDir = getClippyMoveDirection(clippyPtr.pos, emptyThings, rng)
    
    # Update clippy position
    let newPos = clippyPtr.pos + moveDir
    if env.isEmpty(newPos):
      clippyPtr.pos = newPos
      positions.add(newPos)
      
      # Track unique positions
      let posKey = fmt"{newPos.x},{newPos.y}"
      if posKey notin visitedPositions:
        visitedPositions.incl(posKey)
    
    # Calculate distance from temple
    let distance = calculateDistance(clippyPtr.pos, templePos)
    
    # Print status
    echo fmt"{step:4} | ({clippyPtr.pos.x:2}, {clippyPtr.pos.y:2}) | {distance:6.2f} | " &
         fmt"{clippyPtr.wanderRadius:6} | {clippyPtr.wanderAngle:5.2f}"
    
    # Check if we're completing circles (angle wraps around)
    if step > 0 and clippyPtr.wanderAngle < 0.1:
      echo "  --> Completed circle, radius expanded!"
  
  # Visualize movement pattern
  visualizeClippyMovement(positions, templePos)
  
  echo "\n" & repeat("=", 60)
  echo "=== Analysis ==="
  
  # Analyze the pattern
  echo fmt"Total positions visited: {positions.len}"
  echo fmt"Unique positions visited: {visitedPositions.len}"
  echo fmt"Starting radius: 3"
  echo fmt"Final radius: {clippyPtr.wanderRadius}"
  
  # Calculate average distance progression
  var avgDistances: seq[float] = @[]
  for i in 0 ..< 3:
    var sum = 0.0
    var count = 0
    for j in i * 10 ..< min((i + 1) * 10, positions.len):
      sum += calculateDistance(positions[j], templePos)
      count += 1
    if count > 0:
      avgDistances.add(sum / count.float)
  
  echo "\nAverage distance from temple by phase:"
  for i, dist in avgDistances:
    echo fmt"  Phase {i + 1}: {dist:6.2f}"
  
  # Verify expansion
  var checks = 0
  var passed = 0
  
  # Check 1: Radius should expand
  checks += 1
  if clippyPtr.wanderRadius > 3:
    echo fmt"✅ Radius expanded from 3 to {clippyPtr.wanderRadius}"
    passed += 1
  else:
    echo "❌ Radius did not expand"
  
  # Check 2: Should visit multiple unique positions
  checks += 1
  if visitedPositions.len >= 10:
    echo fmt"✅ Visited {visitedPositions.len} unique positions"
    passed += 1
  else:
    echo fmt"❌ Only visited {visitedPositions.len} positions"
  
  # Check 3: Average distance should increase over time
  checks += 1
  if avgDistances.len >= 2 and avgDistances[^1] > avgDistances[0]:
    echo "✅ Average distance from temple increased over time"
    passed += 1
  else:
    echo "❌ Distance pattern not expanding properly"
  
  echo "\n" & repeat("=", 60)
  echo fmt"Result: {passed}/{checks} checks passed"
  
  if passed == checks:
    echo "🎉 Clippy concentric circle wandering is working correctly!"
  else:
    echo "⚠️  Some aspects of wandering need investigation"

when isMainModule:
  main()