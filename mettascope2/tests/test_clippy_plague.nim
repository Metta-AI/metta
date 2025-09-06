import ../src/tribal/environment
import ../src/tribal/objects
import ../src/tribal/ai
## Test for Plague-Wave Clippy Expansion
## Tests the improved clippy behavior that spreads outward like a plague
import std/[strformat, strutils, sets, math, tables, sequtils]
import vmath

proc testPlagueWaveExpansion() =
  echo "Test: Plague-Wave Expansion Pattern"
  echo "-" & repeat("-", 40)
  
  var env = newEnvironment()
  
  # Track expansion metrics
  var clippyTrails = initTable[int, seq[IVec2]]()
  var homePositions = initTable[int, IVec2]()
  var maxDistances = initTable[int, float]()
  var clusteringScore = 0.0
  var expansionSpeed = 0.0
  
  # Find all clippys and their home positions
  for thing in env.things:
    if thing.kind == Clippy:
      clippyTrails[thing.id] = @[thing.pos]
      homePositions[thing.id] = thing.homeSpawner
      maxDistances[thing.id] = 0.0
  
  let initialClippyCount = clippyTrails.len
  echo fmt"  Starting with {initialClippyCount} clippys"
  
  # Run simulation for extended period
  for step in 0 ..< 200:
    var actions: array[MapAgents, array[2, uint8]]
    env.step(addr actions)
    
    # Track all clippy positions
    for thing in env.things:
      if thing.kind == Clippy:
        # Track new clippys that spawn during simulation
        if thing.id notin clippyTrails:
          clippyTrails[thing.id] = @[]
          homePositions[thing.id] = thing.homeSpawner
          maxDistances[thing.id] = 0.0
        
        clippyTrails[thing.id].add(thing.pos)
        
        # Calculate distance from home
        let dx = (thing.pos.x - thing.homeSpawner.x).float
        let dy = (thing.pos.y - thing.homeSpawner.y).float
        let dist = sqrt(dx * dx + dy * dy)
        maxDistances[thing.id] = max(maxDistances[thing.id], dist)
    
    # Calculate clustering score every 50 steps
    if step mod 50 == 0 and step > 0:
      var avgNearbyClippys = 0.0
      var clippyPositions: seq[IVec2] = @[]
      
      for thing in env.things:
        if thing.kind == Clippy:
          clippyPositions.add(thing.pos)
      
      # For each clippy, count how many others are nearby
      for i, pos1 in clippyPositions:
        var nearbyCount = 0
        for j, pos2 in clippyPositions:
          if i != j:
            let dist = abs(pos1.x - pos2.x) + abs(pos1.y - pos2.y)
            if dist <= 5:  # Within 5 tiles
              nearbyCount += 1
        avgNearbyClippys += nearbyCount.float
      
      if clippyPositions.len > 0:
        avgNearbyClippys = avgNearbyClippys / clippyPositions.len.float
        echo fmt"    Step {step}: Avg nearby clippys = {avgNearbyClippys:.2f} (lower is better spread)"
        clusteringScore = avgNearbyClippys
  
  # Analyze expansion patterns
  echo fmt"\n  Final clippy count: {clippyTrails.len}"
  
  # Calculate average max distance (expansion radius)
  var totalMaxDist = 0.0
  for id, dist in maxDistances:
    totalMaxDist += dist
  let avgMaxDist = if clippyTrails.len > 0: totalMaxDist / clippyTrails.len.float else: 0.0
  
  # Calculate territory coverage
  var allVisitedPositions = initHashSet[string]()
  for id, trail in clippyTrails:
    for pos in trail:
      allVisitedPositions.incl(fmt"{pos.x},{pos.y}")
  
  echo fmt"  Average max distance from home: {avgMaxDist:.1f} tiles"
  echo fmt"  Total unique positions covered: {allVisitedPositions.len}"
  echo fmt"  Final clustering score: {clusteringScore:.2f} (lower = better spread)"
  
  # Calculate expansion speed (distance per step)
  for id, trail in clippyTrails:
    if trail.len >= 2:
      let startPos = trail[0]
      let endPos = trail[^1]
      let dx = (endPos.x - startPos.x).float
      let dy = (endPos.y - startPos.y).float
      let totalDist = sqrt(dx * dx + dy * dy)
      expansionSpeed += totalDist / trail.len.float
  
  if clippyTrails.len > 0:
    expansionSpeed = expansionSpeed / clippyTrails.len.float
    echo fmt"  Average expansion speed: {expansionSpeed:.3f} tiles/step"
  
  # Evaluate plague-wave success
  if avgMaxDist > 30 and clusteringScore < 1.0 and allVisitedPositions.len > 1000:
    echo "  ✓ Excellent plague-wave expansion! Clippys spread far with minimal clustering"
  elif avgMaxDist > 20 and clusteringScore < 2.0:
    echo "  ✓ Good plague-wave expansion with reasonable spreading"
  elif avgMaxDist > 15:
    echo "  ✓ Moderate expansion - clippys are moving outward"
  else:
    echo "  ⚠ Limited expansion - may need parameter tuning"
  echo ""

proc testDirectionalExpansion() =
  echo "Test: Directional Expansion Away from Origin"
  echo "-" & repeat("-", 40)
  
  var env = newEnvironment()
  
  # Track clippy movement directions relative to home
  var movementVectors: seq[tuple[dx: float, dy: float]] = @[]
  var outwardMovements = 0
  var inwardMovements = 0
  
  # Find initial clippy positions
  var initialPositions = initTable[int, IVec2]()
  for thing in env.things:
    if thing.kind == Clippy:
      initialPositions[thing.id] = thing.pos
  
  # Run short simulation to analyze movement bias
  for step in 0 ..< 50:
    var actions: array[MapAgents, array[2, uint8]]
    
    # Track positions before step
    var prevPositions = initTable[int, IVec2]()
    for thing in env.things:
      if thing.kind == Clippy:
        prevPositions[thing.id] = thing.pos
    
    env.step(addr actions)
    
    # Analyze movement directions
    for thing in env.things:
      if thing.kind == Clippy and thing.id in prevPositions:
        let prevPos = prevPositions[thing.id]
        let currPos = thing.pos
        
        if prevPos != currPos:  # Clippy moved
          # Calculate if movement was away from or toward home
          let homeDist1 = sqrt(((prevPos.x - thing.homeSpawner.x) * (prevPos.x - thing.homeSpawner.x) + 
                               (prevPos.y - thing.homeSpawner.y) * (prevPos.y - thing.homeSpawner.y)).float)
          let homeDist2 = sqrt(((currPos.x - thing.homeSpawner.x) * (currPos.x - thing.homeSpawner.x) + 
                               (currPos.y - thing.homeSpawner.y) * (currPos.y - thing.homeSpawner.y)).float)
          
          if homeDist2 > homeDist1:
            outwardMovements += 1
          elif homeDist2 < homeDist1:
            inwardMovements += 1
          
          # Track movement vector
          let dx = (currPos.x - prevPos.x).float
          let dy = (currPos.y - prevPos.y).float
          movementVectors.add((dx: dx, dy: dy))
  
  # Analyze results
  let totalMovements = outwardMovements + inwardMovements
  let outwardRatio = if totalMovements > 0: outwardMovements.float / totalMovements.float else: 0.0
  
  echo fmt"  Outward movements: {outwardMovements}"
  echo fmt"  Inward movements: {inwardMovements}"
  echo fmt"  Outward ratio: {outwardRatio*100:.2f}%"
  
  if outwardRatio > 0.7:
    echo "  ✓ Strong outward expansion bias - plague wave working!"
  elif outwardRatio > 0.5:
    echo "  ✓ Moderate outward bias detected"
  else:
    echo "  ⚠ Insufficient outward bias - check expansion logic"
  echo ""

proc testClippyAvoidance() =
  echo "Test: Clippy Mutual Avoidance"
  echo "-" & repeat("-", 40)
  
  var env = newEnvironment()
  
  # Count clippy proximity events
  var tooCloseEvents = 0
  var wellSpacedEvents = 0
  
  # Run simulation and check spacing
  for step in 0 ..< 100:
    var actions: array[MapAgents, array[2, uint8]]
    env.step(addr actions)
    
    # Check clippy spacing
    var clippyPositions: seq[IVec2] = @[]
    for thing in env.things:
      if thing.kind == Clippy:
        clippyPositions.add(thing.pos)
    
    # Check pairwise distances
    for i in 0 ..< clippyPositions.len:
      for j in i+1 ..< clippyPositions.len:
        let dist = abs(clippyPositions[i].x - clippyPositions[j].x) + 
                   abs(clippyPositions[i].y - clippyPositions[j].y)
        if dist <= 3:
          tooCloseEvents += 1
        elif dist >= 6:
          wellSpacedEvents += 1
  
  echo fmt"  Too close events (≤3 tiles): {tooCloseEvents}"
  echo fmt"  Well spaced events (≥6 tiles): {wellSpacedEvents}"
  
  let spacingRatio = if (tooCloseEvents + wellSpacedEvents) > 0:
    wellSpacedEvents.float / (tooCloseEvents + wellSpacedEvents).float
  else: 0.0
  
  echo fmt"  Well-spaced ratio: {spacingRatio*100:.2f}%"
  
  if spacingRatio > 0.8:
    echo "  ✓ Excellent mutual avoidance - clippys spreading out!"
  elif spacingRatio > 0.6:
    echo "  ✓ Good spacing between clippys"
  else:
    echo "  ⚠ Clippys may be clustering too much"
  echo ""

when isMainModule:
  echo "\n" & "=" & repeat("=", 50)
  echo "Clippy Plague-Wave Expansion Test Suite"
  echo "=" & repeat("=", 50) & "\n"
  
  testPlagueWaveExpansion()
  testDirectionalExpansion()
  testClippyAvoidance()
  
  echo "=" & repeat("=", 50)
  echo "Plague-wave tests completed"