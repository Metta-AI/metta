## Core Systems Test Suite
## Tests fundamental game mechanics: movement, resources, and interactions
import std/[strformat, strutils, tables]
import vmath
import ../src/mettascope/tribal
import ../src/mettascope/controller
import ../src/mettascope/common

# Test 1: Directional Movement with Auto-rotation
proc testDirectionalMovement() =
  echo "Test: Directional Movement"
  echo "-" & repeat("-", 40)
  
  var env = newEnvironment()
  let agent = env.agents[0]
  let startPos = agent.pos
  
  # Test movement in each direction
  let movements = [
    (direction: 0, expected: ivec2(0, -1), name: "North"),
    (direction: 1, expected: ivec2(0, 1), name: "South"),
    (direction: 2, expected: ivec2(1, 0), name: "East"),
    (direction: 3, expected: ivec2(-1, 0), name: "West")
  ]
  
  for move in movements:
    # Reset position
    agent.pos = startPos
    env.grid[agent.pos.x][agent.pos.y] = agent
    
    # Apply movement
    var actions: array[MapAgents, array[2, uint8]]
    actions[0] = [1'u8, move.direction.uint8]
    env.step(addr actions)
    
    let actualMove = agent.pos - startPos
    if actualMove == move.expected:
      echo fmt"  ✓ {move.name} movement correct"
    else:
      echo fmt"  ✗ {move.name} movement failed: expected {move.expected}, got {actualMove}"
  echo ""

# Test 2: Resource Collection and Processing
proc testResourceSystem() =
  echo "Test: Resource System"
  echo "-" & repeat("-", 40)
  
  var env = newEnvironment()
  let agent = env.agents[0]
  
  # Find a mine
  var mine: Thing = nil
  for thing in env.things:
    if thing.kind == Mine and thing.resources > 0:
      mine = thing
      break
  
  if mine != nil:
    # Move agent next to mine
    agent.pos = ivec2(mine.pos.x + 1, mine.pos.y)
    env.grid[agent.pos.x][agent.pos.y] = agent
    
    # Use mine to get ore
    var actions: array[MapAgents, array[2, uint8]]
    actions[0] = [3'u8, 3'u8]  # Use action, west direction
    env.step(addr actions)
    
    if agent.inventoryOre > 0:
      echo "  ✓ Ore collection successful"
    else:
      echo "  ✗ Ore collection failed"
    
    # Test converter
    agent.inventoryOre = 1
    var conv: Thing = nil
    for thing in env.things:
      if thing.kind == Converter:
        conv = thing
        break
    
    if conv != nil:
      agent.pos = ivec2(conv.pos.x + 1, conv.pos.y)
      env.grid[agent.pos.x][agent.pos.y] = agent
      
      actions[0] = [3'u8, 3'u8]  # Use converter
      env.step(addr actions)
      
      if agent.inventoryBattery > 0 and agent.inventoryOre == 0:
        echo "  ✓ Ore to battery conversion successful"
      else:
        echo "  ✗ Conversion failed"
  echo ""

# Test 3: Altar Interaction
proc testAltarDeposit() =
  echo "Test: Altar Deposit"
  echo "-" & repeat("-", 40)
  
  var env = newEnvironment()
  let agent = env.agents[0]
  
  # Find agent's home altar
  var altar: Thing = nil
  for thing in env.things:
    if thing.kind == Altar and thing.pos == agent.homeAltar:
      altar = thing
      break
  
  if altar != nil:
    # Give agent a battery
    agent.inventoryBattery = 1
    let initialHearts = altar.hearts
    
    # Move next to altar
    agent.pos = ivec2(altar.pos.x + 1, altar.pos.y)
    env.grid[agent.pos.x][agent.pos.y] = agent
    
    # Deposit battery
    var actions: array[MapAgents, array[2, uint8]]
    actions[0] = [3'u8, 3'u8]  # Use altar
    env.step(addr actions)
    
    if agent.inventoryBattery == 0 and altar.hearts > initialHearts:
      echo fmt"  ✓ Battery deposited, altar hearts: {initialHearts} → {altar.hearts}"
    else:
      echo "  ✗ Battery deposit failed"
  echo ""

# Test 4: Map Generation Variety
proc testMapGeneration() =
  echo "Test: Map Generation"
  echo "-" & repeat("-", 40)
  
  # Generate two environments with different seeds
  let env1 = newEnvironment()
  let env2 = newEnvironment()
  
  # Count entities in each
  var counts1 = (mines: 0, converters: 0, altars: 0, walls: 0)
  var counts2 = (mines: 0, converters: 0, altars: 0, walls: 0)
  
  for thing in env1.things:
    case thing.kind:
    of Mine: counts1.mines += 1
    of Converter: counts1.converters += 1
    of Altar: counts1.altars += 1
    of Wall: counts1.walls += 1
    else: discard
  
  for thing in env2.things:
    case thing.kind:
    of Mine: counts2.mines += 1
    of Converter: counts2.converters += 1
    of Altar: counts2.altars += 1
    of Wall: counts2.walls += 1
    else: discard
  
  echo fmt"  Environment 1: {counts1.mines} mines, {counts1.converters} converters, {counts1.altars} altars"
  echo fmt"  Environment 2: {counts2.mines} mines, {counts2.converters} converters, {counts2.altars} altars"
  
  # Check agent starting positions differ
  var agentPosDiff = 0
  for i in 0 ..< min(env1.agents.len, env2.agents.len):
    if env1.agents[i].pos != env2.agents[i].pos:
      agentPosDiff += 1
  
  if agentPosDiff > 0:
    echo fmt"  ✓ {agentPosDiff} agents have different positions"
  else:
    echo "  ⚠ Maps may be identical (check seed generation)"
  echo ""

when isMainModule:
  echo "\n" & "=" & repeat("=", 50)
  echo "Core Systems Test Suite"
  echo "=" & repeat("=", 50) & "\n"
  
  testDirectionalMovement()
  testResourceSystem()
  testAltarDeposit()
  testMapGeneration()
  
  echo "=" & repeat("=", 50)
  echo "Core systems tests completed"