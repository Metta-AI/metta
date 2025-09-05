## Test for Heatmap Color System
## Verifies that entities properly modify tile colors
import std/[strformat, math, strutils]
import vmath
import ../src/tribal/environment

proc testHeatmapSystem() =
  echo "Test: Heatmap Color Modifications"
  echo "-" & repeat("-", 40)
  
  var env = newEnvironment()
  
  # Get initial tile color at a test position
  let testPos = ivec2(50, 25)
  let initialR = env.tileColors[testPos.x][testPos.y].r
  let initialG = env.tileColors[testPos.x][testPos.y].g
  let initialB = env.tileColors[testPos.x][testPos.y].b
  
  echo fmt"  Initial color at (50,25): R={initialR:.3f}, G={initialG:.3f}, B={initialB:.3f}"
  
  # Find an agent and move it to test position
  var testAgent: Thing = nil
  for agent in env.agents:
    testAgent = agent
    break
  
  if not isNil(testAgent):
    # Move agent to test position
    env.grid[testAgent.pos.x][testAgent.pos.y] = nil
    testAgent.pos = testPos
    env.grid[testPos.x][testPos.y] = testAgent
    
    echo fmt"  Moved agent {testAgent.agentId} to test position"
    
    # Step environment multiple times to trigger heatmap updates
    var actions: array[MapAgents, array[2, uint8]]
    for i in 0 ..< 10:
      env.step(addr actions)
    
    # Check if color changed
    let newR = env.tileColors[testPos.x][testPos.y].r
    let newG = env.tileColors[testPos.x][testPos.y].g
    let newB = env.tileColors[testPos.x][testPos.y].b
    
    echo fmt"  After 10 steps: R={newR:.3f}, G={newG:.3f}, B={newB:.3f}"
    
    if abs(newR - initialR) > 0.001 or abs(newG - initialG) > 0.001 or abs(newB - initialB) > 0.001:
      echo "  ✓ Agent movement modified tile color"
    else:
      echo "  ✗ Agent movement did not modify tile color"
  
  # Test clippy cold effect
  echo ""
  echo "  Testing clippy cold effect..."
  
  # Find a clippy
  var testClippy: Thing = nil
  for thing in env.things:
    if thing.kind == Clippy:
      testClippy = thing
      break
  
  if not isNil(testClippy):
    let clippyPos = ivec2(60, 30)
    let initialClippyR = env.tileColors[clippyPos.x][clippyPos.y].r
    let initialClippyB = env.tileColors[clippyPos.x][clippyPos.y].b
    
    # Move clippy to test position
    env.grid[testClippy.pos.x][testClippy.pos.y] = nil
    testClippy.pos = clippyPos
    env.grid[clippyPos.x][clippyPos.y] = testClippy
    
    # Step to trigger update
    var actions2: array[MapAgents, array[2, uint8]]
    for i in 0 ..< 10:
      env.step(addr actions2)
    
    let newClippyR = env.tileColors[clippyPos.x][clippyPos.y].r
    let newClippyB = env.tileColors[clippyPos.x][clippyPos.y].b
    
    echo fmt"  Clippy position color change: R {initialClippyR:.3f} -> {newClippyR:.3f}, B {initialClippyB:.3f} -> {newClippyB:.3f}"
    
    if newClippyR < initialClippyR and newClippyB > initialClippyB:
      echo "  ✓ Clippy made tile colder (less red, more blue)"
    else:
      echo "  ✗ Clippy cold effect not working properly"
  
  # Test altar brightness
  echo ""
  echo "  Testing altar brightness effect..."
  
  for thing in env.things:
    if thing.kind == Altar and thing.houseSize > 0:
      let houseX = thing.houseTopLeft.x + 2
      let houseY = thing.houseTopLeft.y + 2
      if houseX >= 0 and houseX < MapWidth and houseY >= 0 and houseY < MapHeight:
        let intensity = env.tileColors[houseX][houseY].intensity
        echo fmt"  Altar with {thing.hearts} hearts -> house tile intensity: {intensity:.2f}"
        
        if intensity > 1.0:
          echo "  ✓ Altar brightness affecting house tiles"
        else:
          echo "  ✗ Altar brightness not working"
        break
  
  echo ""

when isMainModule:
  echo "\n" & "=" & repeat("=", 50)
  echo "Heatmap System Test"
  echo "=" & repeat("=", 50) & "\n"
  
  testHeatmapSystem()
  
  echo "=" & repeat("=", 50)
  echo "Heatmap test completed"