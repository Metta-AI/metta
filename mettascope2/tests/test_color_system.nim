## Comprehensive Color and Heatmap System Test Suite
## Tests agent colors, house colors, and heatmap modifications
import std/[strformat, strutils, tables, math]
import vmath
import ../src/tribal/environment

proc testAgentWarmColors(env: Environment): bool =
  ## Verify agents have warm colors (reds/oranges/yellows)
  echo "  [1] Agent Warm Colors:"
  echo "  " & repeat("-", 38)
  
  result = true
  var warmCount = 0
  var totalAgents = 0
  
  for i, agent in env.agents:
    if i < agentVillageColors.len:
      let color = agentVillageColors[i]
      totalAgents += 1
      
      # Check if color is warm (high red component, moderate green, low blue)
      if color.r >= 0.8 and color.b <= 0.6:
        warmCount += 1
      elif color.r >= 0.7:
        warmCount += 1
      else:
        result = false
        echo fmt"    Agent {i}: R={color.r:.2f}, G={color.g:.2f}, B={color.b:.2f} - NOT WARM!"
  
  echo fmt"    Checked {totalAgents} agents: {warmCount}/{totalAgents} have warm colors"
  if result:
    echo "    ✓ All agents have warm colors"
  else:
    echo "    ✗ Some agents don't have warm colors"
  echo ""

proc testHouseTeamColors(env: Environment): bool =
  ## Verify house tiles match their team colors
  echo "  [2] House Team Colors:"
  echo "  " & repeat("-", 38)
  
  result = true
  var housesChecked = 0
  var housesCorrect = 0
  
  for thing in env.things:
    if thing.kind == Altar and thing.houseSize > 0 and thing.houseTopLeft.x >= 0:
      housesChecked += 1
      let altarPos = thing.pos
      let teamColor = altarColors[altarPos]
      
      # Check center house tile
      let centerX = thing.houseTopLeft.x + 2
      let centerY = thing.houseTopLeft.y + 2
      
      if centerX >= 0 and centerX < MapWidth and centerY >= 0 and centerY < MapHeight:
        let baseColor = env.baseTileColors[centerX][centerY]
        
        # Check if base color matches team color
        if abs(baseColor.r - teamColor.r) < 0.01 and 
           abs(baseColor.g - teamColor.g) < 0.01 and
           abs(baseColor.b - teamColor.b) < 0.01:
          housesCorrect += 1
        else:
          result = false
          echo fmt"    House at ({thing.houseTopLeft.x},{thing.houseTopLeft.y}):"
          echo fmt"      Expected: R={teamColor.r:.2f}, G={teamColor.g:.2f}, B={teamColor.b:.2f}"
          echo fmt"      Got:      R={baseColor.r:.2f}, G={baseColor.g:.2f}, B={baseColor.b:.2f}"
  
  echo fmt"    Checked {housesChecked} houses: {housesCorrect}/{housesChecked} have correct team colors"
  if result:
    echo "    ✓ All houses have correct team colors"
  else:
    echo "    ✗ Some houses have incorrect team colors"
  echo ""

proc testAgentHeatmapEffect(env: Environment): bool =
  ## Test that agent movement creates warm trails
  echo "  [3] Agent Heatmap Effect:"
  echo "  " & repeat("-", 38)
  
  result = false
  
  # Find an agent and move it to test position
  if env.agents.len > 0:
    let agent = env.agents[0]
    let testPos = ivec2(50, 25)
    
    # Record initial color
    let initialR = env.tileColors[testPos.x][testPos.y].r
    let initialG = env.tileColors[testPos.x][testPos.y].g
    let initialB = env.tileColors[testPos.x][testPos.y].b
    
    # Move agent
    env.grid[agent.pos.x][agent.pos.y] = nil
    agent.pos = testPos
    env.grid[testPos.x][testPos.y] = agent
    
    # Step environment to trigger heatmap
    var actions: array[MapAgents, array[2, uint8]]
    for i in 0 ..< 10:
      env.step(addr actions)
    
    # Check if color changed (warmer)
    let newR = env.tileColors[testPos.x][testPos.y].r
    let newG = env.tileColors[testPos.x][testPos.y].g
    let newB = env.tileColors[testPos.x][testPos.y].b
    
    if abs(newR - initialR) > 0.001 or abs(newG - initialG) > 0.001 or abs(newB - initialB) > 0.001:
      result = true
      echo fmt"    Initial: R={initialR:.3f}, G={initialG:.3f}, B={initialB:.3f}"
      echo fmt"    After:   R={newR:.3f}, G={newG:.3f}, B={newB:.3f}"
      echo "    ✓ Agent movement creates color trail"
    else:
      echo "    ✗ Agent movement did not modify tile color"
  else:
    echo "    ✗ No agents to test"
  echo ""

proc testClippyHeatmapEffect(env: Environment): bool =
  ## Test that clippy movement creates cold trails
  echo "  [4] Clippy Cold Effect:"
  echo "  " & repeat("-", 38)
  
  result = false
  
  # Find a clippy
  var testClippy: Thing = nil
  for thing in env.things:
    if thing.kind == Clippy:
      testClippy = thing
      break
  
  if not isNil(testClippy):
    let clippyPos = ivec2(60, 30)
    
    # Clear position and move clippy
    env.grid[testClippy.pos.x][testClippy.pos.y] = nil
    if not isNil(env.grid[clippyPos.x][clippyPos.y]):
      env.grid[clippyPos.x][clippyPos.y] = nil
    
    testClippy.pos = clippyPos
    env.grid[clippyPos.x][clippyPos.y] = testClippy
    
    # Record initial color
    let initialR = env.tileColors[clippyPos.x][clippyPos.y].r
    let initialB = env.tileColors[clippyPos.x][clippyPos.y].b
    
    # Step to trigger update
    var actions: array[MapAgents, array[2, uint8]]
    for i in 0 ..< 10:
      env.step(addr actions)
    
    # Check the tile where clippy ended up
    let finalPos = testClippy.pos
    let newR = env.tileColors[finalPos.x][finalPos.y].r
    let newB = env.tileColors[finalPos.x][finalPos.y].b
    
    if newR < initialR and newB > initialB:
      result = true
      echo fmt"    Initial: R={initialR:.3f}, B={initialB:.3f}"
      echo fmt"    After:   R={newR:.3f}, B={newB:.3f}"
      echo "    ✓ Clippy makes tiles colder (less red, more blue)"
    else:
      echo fmt"    Change: R {initialR:.3f} -> {newR:.3f}, B {initialB:.3f} -> {newB:.3f}"
      echo "    ✗ Clippy cold effect not working properly"
  else:
    echo "    ✗ No clippies to test"
  echo ""

proc testAltarBrightness(env: Environment): bool =
  ## Test that altar hearts affect house tile brightness
  echo "  [5] Altar Brightness Effect:"
  echo "  " & repeat("-", 38)
  
  result = false
  
  for thing in env.things:
    if thing.kind == Altar and thing.houseSize > 0:
      let houseX = thing.houseTopLeft.x + 2
      let houseY = thing.houseTopLeft.y + 2
      if houseX >= 0 and houseX < MapWidth and houseY >= 0 and houseY < MapHeight:
        let intensity = env.tileColors[houseX][houseY].intensity
        echo fmt"    Altar with {thing.hearts} hearts -> intensity: {intensity:.2f}"
        
        if intensity > 1.0:
          result = true
          echo "    ✓ Altar brightness affecting house tiles"
        else:
          echo "    ✗ Altar brightness not working"
        break
  
  if not result:
    echo "    ⚠ No altars with houses found to test"
  echo ""

proc testColorDecay(env: Environment): bool =
  ## Test that modified colors decay back to base colors
  echo "  [6] Color Decay System:"
  echo "  " & repeat("-", 38)
  
  # Modify a tile color
  let testX = 70
  let testY = 35
  let baseColor = env.baseTileColors[testX][testY]
  
  # Artificially modify the color
  env.tileColors[testX][testY].r = min(baseColor.r + 0.2, 1.0)
  env.tileColors[testX][testY].g = min(baseColor.g + 0.1, 1.0)
  let modifiedR = env.tileColors[testX][testY].r
  
  # Step environment 30 times to trigger decay
  var actions: array[MapAgents, array[2, uint8]]
  for i in 0 ..< 31:
    env.step(addr actions)
  
  let decayedR = env.tileColors[testX][testY].r
  
  result = decayedR < modifiedR and decayedR > baseColor.r
  
  echo fmt"    Base:     R={baseColor.r:.3f}"
  echo fmt"    Modified: R={modifiedR:.3f}"
  echo fmt"    Decayed:  R={decayedR:.3f}"
  
  if result:
    echo "    ✓ Colors decay back toward base colors"
  else:
    echo "    ✗ Color decay not working properly"
  echo ""

proc runColorSystemTests() =
  echo "Running Comprehensive Color System Tests"
  echo "=" & repeat("=", 40)
  echo ""
  
  # Create environment for testing
  var env = newEnvironment()
  
  var allPassed = true
  
  # Run all test suites
  allPassed = allPassed and testAgentWarmColors(env)
  allPassed = allPassed and testHouseTeamColors(env)
  allPassed = allPassed and testAgentHeatmapEffect(env)
  allPassed = allPassed and testClippyHeatmapEffect(env)
  allPassed = allPassed and testAltarBrightness(env)
  allPassed = allPassed and testColorDecay(env)
  
  # Summary
  echo "=" & repeat("=", 40)
  if allPassed:
    echo "✅ ALL COLOR SYSTEM TESTS PASSED"
  else:
    echo "❌ SOME COLOR SYSTEM TESTS FAILED"
  echo ""

when isMainModule:
  runColorSystemTests()