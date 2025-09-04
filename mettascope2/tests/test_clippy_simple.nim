import std/[strformat, strutils, math, sets, random]
import vmath
import ../src/mettascope/clippy

proc testConcentricWandering() =
  echo "Testing Concentric Circle Wandering Logic..."
  echo repeat("=", 60)
  
  # Create a mock clippy pointer with the necessary fields
  var mockClippy = (
    kind: 6,  # Clippy kind
    pos: ivec2(10, 10),
    id: 0,
    layer: 0,
    hearts: 0,
    resources: 0,
    cooldown: 0,
    agentId: -1,
    orientation: 0,
    inventoryOre: 0,
    inventoryBattery: 0,
    inventoryWater: 0,
    inventoryWheat: 0,
    inventoryWood: 0,
    reward: 0.0'f32,
    homeAltar: ivec2(-1, -1),
    homeTemple: ivec2(10, 10),  # Temple at same position
    wanderRadius: 3,
    wanderAngle: 0.0,
    targetPos: ivec2(10, 10)
  )
  
  var rng = initRand(42)
  var visitedPositions = initHashSet[string]()
  
  echo "\nSimulating concentric wandering pattern:"
  echo "Step | Target Position | Distance from Temple | Radius"
  echo repeat("-", 50)
  
  for step in 0 ..< 20:
    # Get next wander point
    let wanderTarget = getConcentricWanderPoint(addr mockClippy, rng)
    
    # Track unique positions
    let posKey = fmt"{wanderTarget.x},{wanderTarget.y}"
    visitedPositions.incl(posKey)
    
    # Calculate distance
    let dx = (wanderTarget.x - mockClippy.homeTemple.x).float
    let dy = (wanderTarget.y - mockClippy.homeTemple.y).float
    let distance = sqrt(dx * dx + dy * dy)
    
    echo fmt"{step:4} | ({wanderTarget.x:3}, {wanderTarget.y:3}) | {distance:8.2f} | {mockClippy.wanderRadius:6}"
  
  echo "\n" & repeat("=", 60)
  echo fmt"Total unique positions: {visitedPositions.len}"
  echo fmt"Final radius: {mockClippy.wanderRadius}"
  
  # Verify the wandering expands
  if mockClippy.wanderRadius > 3:
    echo "✅ Radius expanded correctly"
  else:
    echo "❌ Radius did not expand"

proc testDirectionLogic() =
  echo "\nTesting Direction Calculation Logic..."
  echo repeat("=", 60)
  
  # Test cases for getDirectionToward
  let testCases = [
    (ivec2(5, 5), ivec2(8, 5), ivec2(1, 0), "East"),
    (ivec2(5, 5), ivec2(2, 5), ivec2(-1, 0), "West"),
    (ivec2(5, 5), ivec2(5, 8), ivec2(0, 1), "South"),
    (ivec2(5, 5), ivec2(5, 2), ivec2(0, -1), "North"),
    (ivec2(5, 5), ivec2(5, 5), ivec2(0, 0), "Same position"),
  ]
  
  for tc in testCases:
    let result = getDirectionToward(tc[0], tc[1])
    if result == tc[2]:
      echo fmt"✅ {tc[3]}: ({tc[0].x}, {tc[0].y}) → ({tc[1].x}, {tc[1].y}) = ({result.x}, {result.y})"
    else:
      echo fmt"❌ {tc[3]}: Expected ({tc[2].x}, {tc[2].y}), got ({result.x}, {result.y})"

when isMainModule:
  testConcentricWandering()
  testDirectionLogic()
  echo "\nAll clippy movement tests completed!"