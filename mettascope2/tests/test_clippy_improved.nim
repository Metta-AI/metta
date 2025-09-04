## Test improved clippy behavior with extended range
import std/[strformat, sets, math]
import ../src/mettascope/tribal
import ../src/mettascope/clippy
import vmath

echo "======================================"
echo "Improved Clippy Behavior Test"
echo "======================================"
echo ""

proc testImprovedClippyRange() =
  echo "Testing clippy extended range and behavior..."
  echo ""
  
  var env = newEnvironment()
  
  # Find a clippy and track its behavior
  var clippy: Thing = nil
  for thing in env.things:
    if thing.kind == Clippy:
      clippy = thing
      break
  
  if clippy == nil:
    echo "No clippy found in environment"
    return
  
  echo fmt"Clippy starting at ({clippy.pos.x}, {clippy.pos.y})"
  echo fmt"  Home temple: ({clippy.homeTemple.x}, {clippy.homeTemple.y})"
  echo fmt"  Initial wander radius: {clippy.wanderRadius}"
  echo fmt"  Initial wander angle: {clippy.wanderAngle:.2f}"
  echo ""
  
  # Track clippy movement over time
  var positions = initHashSet[string]()
  var maxDistFromTemple: float = 0.0
  var targetsSeen = 0
  var chasedAgents = 0
  
  echo "Simulating 100 steps..."
  for step in 0..<100:
    # Store position
    let posKey = fmt"{clippy.pos.x},{clippy.pos.y}"
    positions.incl(posKey)
    
    # Calculate distance from temple
    let distFromTemple = sqrt(pow((clippy.pos.x - clippy.homeTemple.x).float, 2) + 
                             pow((clippy.pos.y - clippy.homeTemple.y).float, 2))
    if distFromTemple > maxDistFromTemple:
      maxDistFromTemple = distFromTemple
    
    # Check if clippy has a target
    if clippy.targetPos.x >= 0 and clippy.targetPos.y >= 0:
      targetsSeen += 1
      
      # Check if targeting an agent
      for agent in env.agents:
        if agent.pos == clippy.targetPos:
          chasedAgents += 1
          break
    
    # Report significant events
    if step mod 20 == 0:
      echo fmt"  Step {step}: Clippy at ({clippy.pos.x}, {clippy.pos.y}), radius={clippy.wanderRadius}, dist from temple={distFromTemple:.1f}"
    
    # Simulate one step
    var actions: array[MapAgents, array[2, uint8]]
    env.step(addr actions)
    
    # Check if clippy was eliminated
    var stillExists = false
    for thing in env.things:
      if thing == clippy:
        stillExists = true
        break
    
    if not stillExists:
      echo fmt"  Clippy was eliminated at step {step}"
      break
  
  echo ""
  echo "Behavior Summary:"
  echo fmt"  Unique positions visited: {positions.len}"
  echo fmt"  Max distance from temple: {maxDistFromTemple:.1f}"
  echo fmt"  Times a target was tracked: {targetsSeen}"
  echo fmt"  Times chased agents: {chasedAgents}"
  echo fmt"  Final wander radius: {clippy.wanderRadius}"
  echo ""
  
  # Evaluate performance
  if maxDistFromTemple > 30:
    echo "✓ Clippy successfully ranged far from temple (>30 tiles)"
  elif maxDistFromTemple > 20:
    echo "✓ Clippy explored moderate distance from temple (>20 tiles)"
  else:
    echo "⚠ Clippy stayed close to temple (<20 tiles)"
  
  if positions.len > 50:
    echo "✓ Excellent exploration coverage (>50 unique positions)"
  elif positions.len > 30:
    echo "✓ Good exploration coverage (>30 unique positions)"
  else:
    echo "⚠ Limited exploration coverage (<30 unique positions)"

# Run the test
testImprovedClippyRange()

echo ""
echo "======================================"
echo "Test complete!"