## Debug clippy movement to understand behavior
import std/[strformat, random]
import ../src/mettascope/tribal
import ../src/mettascope/clippy
import vmath

echo "======================================"
echo "Clippy Movement Debug"
echo "======================================"
echo ""

proc debugClippyMovement() =
  var env = newEnvironment()
  
  # Find a clippy
  var clippy: Thing = nil
  for thing in env.things:
    if thing.kind == Clippy:
      clippy = thing
      break
  
  if clippy == nil:
    echo "No clippy found"
    return
  
  echo fmt"Clippy at ({clippy.pos.x}, {clippy.pos.y})"
  echo fmt"Temple at ({clippy.homeTemple.x}, {clippy.homeTemple.y})"
  echo fmt"Initial radius: {clippy.wanderRadius}"
  echo ""
  
  # Manually test the movement logic
  var r = initRand(42)
  
  for step in 0..<10:
    echo fmt"Step {step}:"
    
    # Get things as pointers
    var thingPtrs: seq[pointer] = @[]
    for t in env.things:
      thingPtrs.add(cast[pointer](t))
    
    # Get movement direction
    let moveDir = getClippyMoveDirection(clippy.pos, thingPtrs, r)
    echo fmt"  Current pos: ({clippy.pos.x}, {clippy.pos.y})"
    echo fmt"  Move direction: ({moveDir.x}, {moveDir.y})"
    echo fmt"  Target pos: ({clippy.targetPos.x}, {clippy.targetPos.y})"
    echo fmt"  Wander radius: {clippy.wanderRadius}"
    echo fmt"  Wander angle: {clippy.wanderAngle:.2f}"
    echo fmt"  Wander steps remaining: {clippy.wanderStepsRemaining}"
    
    let newPos = clippy.pos + moveDir
    echo fmt"  New pos would be: ({newPos.x}, {newPos.y})"
    
    # Check what's at the new position
    if newPos.x >= 0 and newPos.x < MapWidth and newPos.y >= 0 and newPos.y < MapHeight:
      if env.isEmpty(newPos):
        echo "  ✓ Position is empty, can move"
        # Actually move
        env.grid[clippy.pos.x][clippy.pos.y] = nil
        clippy.pos = newPos
        env.grid[newPos.x][newPos.y] = clippy
      else:
        let blocking = env.grid[newPos.x][newPos.y]
        echo fmt"  ✗ Position blocked by {blocking.kind}"
    else:
      echo "  ✗ Position out of bounds"
    
    echo ""

debugClippyMovement()

echo "======================================"