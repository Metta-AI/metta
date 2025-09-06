## Test for Clippy Kamikaze Behavior
## Verifies that clippys sacrifice themselves to damage altars
import std/[strformat, strutils]
import vmath
import ../src/tribal/environment
import ../src/tribal/objects
import ../src/tribal/common

proc testClippyAltarKamikaze() =
  echo "Test: Clippy Kamikaze Attack on Altar"
  echo "-" & repeat("-", 40)
  
  # Create environment with controlled setup
  var env = newEnvironment()
  
  # Find an altar and a clippy
  var altar: Thing = nil
  var clippy: Thing = nil
  var initialAltarHearts = 0
  var initialClippyCount = 0
  
  # Count initial clippys
  for thing in env.things:
    if thing.kind == Clippy:
      initialClippyCount += 1
      if clippy == nil:
        clippy = thing
    elif thing.kind == Altar and altar == nil:
      altar = thing
      initialAltarHearts = thing.hearts
  
  if altar != nil and clippy != nil:
    echo fmt"  Initial state:"
    echo fmt"    Altar at ({altar.pos.x}, {altar.pos.y}) with {initialAltarHearts} hearts"
    echo fmt"    Clippy at ({clippy.pos.x}, {clippy.pos.y})"
    echo fmt"    Total clippys: {initialClippyCount}"
    
    # Move clippy to be adjacent to altar
    clippy.pos = ivec2(altar.pos.x + 1, altar.pos.y)
    env.grid[clippy.pos.x][clippy.pos.y] = clippy
    clippy.targetPos = altar.pos  # Set altar as target
    
    echo fmt"    Moved clippy to ({clippy.pos.x}, {clippy.pos.y}) next to altar"
    
    # Run one simulation step - clippy should attack altar
    var actions: array[MapAgents, array[2, uint8]]
    env.step(addr actions)
    
    # Check results
    var finalClippyCount = 0
    var finalAltarHearts = altar.hearts
    var clippyStillExists = false
    
    for thing in env.things:
      if thing.kind == Clippy:
        finalClippyCount += 1
        if thing == clippy:
          clippyStillExists = true
    
    echo fmt"\n  After clippy attacks altar:"
    echo fmt"    Altar hearts: {initialAltarHearts} -> {finalAltarHearts}"
    echo fmt"    Clippy count: {initialClippyCount} -> {finalClippyCount}"
    echo fmt"    Original clippy exists: {clippyStillExists}"
    
    # Verify the kamikaze attack worked
    if finalAltarHearts == initialAltarHearts - 1 and not clippyStillExists:
      echo "  ✓ Clippy successfully sacrificed itself to damage altar!"
    elif finalAltarHearts < initialAltarHearts:
      echo fmt"  ✓ Altar was damaged (lost {initialAltarHearts - finalAltarHearts} hearts)"
      if clippyStillExists:
        echo "  ⚠ But clippy didn't die - check removal logic"
    else:
      echo "  ✗ Kamikaze attack failed - altar not damaged"
  else:
    echo "  ⚠ Could not find both altar and clippy to test"
  echo ""

proc testMultipleClippyAttacks() =
  echo "Test: Multiple Clippys Attacking Same Altar"
  echo "-" & repeat("-", 40)
  
  env = newEnvironment()
  
  # Find an altar
  var altar: Thing = nil
  for thing in env.things:
    if thing.kind == Altar:
      altar = thing
      break
  
  if altar != nil:
    let initialHearts = altar.hearts
    echo fmt"  Altar starting with {initialHearts} hearts at ({altar.pos.x}, {altar.pos.y})"
    
    # Manually create multiple clippys around the altar
    var testClippys: seq[Thing] = @[]
    let positions = @[
      ivec2(altar.pos.x + 1, altar.pos.y),  # East
      ivec2(altar.pos.x - 1, altar.pos.y),  # West
      ivec2(altar.pos.x, altar.pos.y + 1),  # South
    ]
    
    for i, pos in positions:
      if env.isEmpty(pos):
        let testClippy = Thing(
          kind: Clippy,
          pos: pos,
          orientation: N,
          homeSpawner: ivec2(0, 0),
          targetPos: altar.pos
        )
        env.things.add(testClippy)
        env.grid[pos.x][pos.y] = testClippy
        testClippys.add(testClippy)
        echo fmt"    Added test clippy {i+1} at ({pos.x}, {pos.y})"
    
    # Run simulation for multiple steps
    for step in 0 ..< 3:
      var actions: array[MapAgents, array[2, uint8]]
      env.step(addr actions)
      echo fmt"    Step {step+1}: Altar has {altar.hearts} hearts"
    
    let heartsLost = initialHearts - altar.hearts
    
    # Count surviving clippys
    var survivingClippys = 0
    for thing in env.things:
      if thing.kind == Clippy and thing in testClippys:
        survivingClippys += 1
    
    echo fmt"\n  Results:"
    echo fmt"    Hearts lost: {heartsLost}"
    echo fmt"    Clippys died: {testClippys.len - survivingClippys}"
    
    if heartsLost > 0:
      echo "  ✓ Multiple clippys successfully damaged the altar!"
    else:
      echo "  ✗ No damage dealt by clippys"
  else:
    echo "  ⚠ No altar found in environment"
  echo ""

proc testClippyPrioritizesAltars() =
  echo "Test: Clippy Prioritizes Altars Over Wandering"
  echo "-" & repeat("-", 40)
  
  env = newEnvironment()
  
  # Find a clippy and altar that are somewhat close
  var clippy: Thing = nil
  var altar: Thing = nil
  
  for thing in env.things:
    if thing.kind == Clippy and clippy == nil:
      clippy = thing
    elif thing.kind == Altar and altar == nil:
      altar = thing
  
  if clippy != nil and altar != nil:
    # Position clippy within search range of altar
    clippy.pos = ivec2(altar.pos.x + ClippyAltarSearchRange - 2, altar.pos.y)
    env.grid[clippy.pos.x][clippy.pos.y] = clippy
    
    let initialDist = abs(clippy.pos.x - altar.pos.x) + abs(clippy.pos.y - altar.pos.y)
    echo fmt"  Clippy at ({clippy.pos.x}, {clippy.pos.y})"
    echo fmt"  Altar at ({altar.pos.x}, {altar.pos.y})"
    echo fmt"  Initial distance: {initialDist} tiles"
    echo fmt"  Altar search range: {ClippyAltarSearchRange} tiles"
    
    # Track clippy movement
    var distances: seq[int] = @[initialDist.int]
    
    for step in 0 ..< 10:
      var actions: array[MapAgents, array[2, uint8]]
      env.step(addr actions)
      
      # Check if clippy still exists
      var clippyFound = false
      for thing in env.things:
        if thing == clippy:
          clippyFound = true
          let dist = abs(thing.pos.x - altar.pos.x) + abs(thing.pos.y - altar.pos.y)
          distances.add(dist.int)
          if dist <= 1:
            echo fmt"    Step {step+1}: Clippy reached altar!"
            break
      
      if not clippyFound:
        echo fmt"    Step {step+1}: Clippy died (presumably attacked altar)"
        break
    
    # Check if clippy moved toward altar
    var movedCloser = false
    for i in 1 ..< distances.len:
      if distances[i] < distances[0]:
        movedCloser = true
        break
    
    if movedCloser:
      echo "  ✓ Clippy moved toward altar when in range!"
    else:
      echo "  ⚠ Clippy didn't prioritize altar movement"
  else:
    echo "  ⚠ Could not find clippy and altar"
  echo ""

when isMainModule:
  echo "\n" & "=" & repeat("=", 50)
  echo "Clippy Kamikaze Behavior Test Suite"
  echo "=" & repeat("=", 50) & "\n"
  
  testClippyAltarKamikaze()
  testMultipleClippyAttacks()
  testClippyPrioritizesAltars()
  
  echo "=" & repeat("=", 50)
  echo "Kamikaze tests completed"