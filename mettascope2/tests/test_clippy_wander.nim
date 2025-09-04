## Test clippy wander point generation directly
import std/[strformat, random, math]
import ../src/mettascope/clippy
import ../src/mettascope/tribal
import vmath

echo "======================================"
echo "Clippy Wander Point Test"
echo "======================================"
echo ""

proc testWanderPointGeneration() =
  # Create a mock clippy Thing
  var clippy = Thing(
    kind: Clippy,
    pos: ivec2(50, 50),
    homeTemple: ivec2(50, 50),
    wanderRadius: 5,
    wanderAngle: 0.0,
    targetPos: ivec2(-1, -1),
    wanderStepsRemaining: 0
  )
  
  var r = initRand(42)
  
  echo "Testing wander point generation directly..."
  echo fmt"Starting: radius={clippy.wanderRadius}, angle={clippy.wanderAngle:.2f}"
  echo ""
  
  for i in 0..<20:
    let beforeAngle = clippy.wanderAngle
    let beforeRadius = clippy.wanderRadius
    
    # Call the wander point function
    let wanderPoint = getConcentricWanderPoint(cast[pointer](clippy), r)
    
    # Calculate actual distance from temple
    let dist = sqrt(pow((wanderPoint.x - 50).float, 2) + pow((wanderPoint.y - 50).float, 2))
    
    echo fmt"Point {i+1}:"
    echo fmt"  Before: angle={beforeAngle:.2f}, radius={beforeRadius}"
    echo fmt"  After:  angle={clippy.wanderAngle:.2f}, radius={clippy.wanderRadius}"
    echo fmt"  Wander target: ({wanderPoint.x}, {wanderPoint.y}), dist={dist:.1f}"
    
    if clippy.wanderAngle != beforeAngle:
      echo "  ✓ Angle updated!"
    
    if clippy.wanderRadius != beforeRadius:
      echo "  ✓ Radius expanded!"
    echo ""

testWanderPointGeneration()

echo "======================================"