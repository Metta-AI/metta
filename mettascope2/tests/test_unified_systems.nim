## Unified Systems Test Suite  
## Tests the consolidated GET/PUT action system and core game mechanics
import std/[strformat, strutils, tables]
import vmath
import ../src/tribal/environment
import ../src/tribal/objects
import ../src/tribal/common

# Common test utilities
proc createTestEnvironment(): Environment =
  result = newEnvironment()

proc findObjectOfKind(env: Environment, kind: ThingKind): Thing =
  for thing in env.things:
    if thing.kind == kind:
      return thing
  return nil

proc moveAgentTo(env: Environment, agent: Thing, pos: IVec2) =
  if agent.pos.x >= 0 and agent.pos.y >= 0 and 
     agent.pos.x < MapWidth and agent.pos.y < MapHeight:
    env.grid[agent.pos.x][agent.pos.y] = nil
  agent.pos = pos
  env.grid[pos.x][pos.y] = agent

# Test 1: Unified GET Action System
proc testGetActionSystem() =
  echo "Test: Unified GET Action System"
  echo "-" & repeat("-", 40)
  
  var env = createTestEnvironment()
  let agent = env.agents[0]
  
  # Test GET from mine (ore)
  let mine = findObjectOfKind(env, Mine)
  if mine != nil and mine.resources > 0:
    let adjacentPos = ivec2(mine.pos.x + 1, mine.pos.y)
    moveAgentTo(env, agent, adjacentPos)
    
    let initialOre = agent.inventoryOre
    let initialMineRes = mine.resources
    
    var actions: array[MapAgents, array[2, uint8]]
    actions[0] = [3'u8, 3'u8]  # GET action, facing west toward mine
    env.step(addr actions)
    
    if agent.inventoryOre > initialOre and mine.resources < initialMineRes:
      echo "  ✓ GET from mine: ore extracted successfully"
    else:
      echo "  ✗ GET from mine failed"
  
  # Test GET from converter (battery)
  let conv = findObjectOfKind(env, Converter)
  if conv != nil:
    agent.inventoryOre = 1  # Give ore to convert
    let adjacentPos = ivec2(conv.pos.x + 1, conv.pos.y)
    moveAgentTo(env, agent, adjacentPos)
    
    var actions: array[MapAgents, array[2, uint8]]
    actions[0] = [3'u8, 3'u8]  # GET from converter
    env.step(addr actions)
    
    if agent.inventoryBattery > 0 and agent.inventoryOre == 0:
      echo "  ✓ GET from converter: ore→battery conversion successful"
    else:
      echo "  ✗ GET from converter failed"
  
  echo ""

# Test 2: Unified PUT Action System  
proc testPutActionSystem() =
  echo "Test: Unified PUT Action System"
  echo "-" & repeat("-", 40)
  
  var env = createTestEnvironment()
  let agent = env.agents[0]
  
  # Test PUT to forge (wood→spear)
  let forge = findObjectOfKind(env, Forge)
  if forge != nil:
    agent.inventoryWood = 2
    let adjacentPos = ivec2(forge.pos.x + 1, forge.pos.y)
    moveAgentTo(env, agent, adjacentPos)
    
    let initialSpears = agent.inventorySpear
    let initialWood = agent.inventoryWood
    
    var actions: array[MapAgents, array[2, uint8]]
    actions[0] = [5'u8, 3'u8]  # PUT action, facing west toward forge
    env.step(addr actions)
    
    if agent.inventorySpear > initialSpears and agent.inventoryWood < initialWood:
      echo "  ✓ PUT to forge: wood→spear crafting successful"
    else:
      echo "  ✗ PUT to forge failed"
  
  # Test PUT to altar (battery→hearts)
  let altar = findObjectOfKind(env, Altar)
  if altar != nil:
    agent.inventoryBattery = 1
    let adjacentPos = ivec2(altar.pos.x + 1, altar.pos.y)
    moveAgentTo(env, agent, adjacentPos)
    
    let initialHearts = altar.hearts
    
    var actions: array[MapAgents, array[2, uint8]]
    actions[0] = [5'u8, 3'u8]  # PUT to altar
    env.step(addr actions)
    
    if altar.hearts > initialHearts and agent.inventoryBattery == 0:
      echo "  ✓ PUT to altar: battery→hearts successful"
    else:
      echo "  ✗ PUT to altar failed"
      
  echo ""

# Test 3: Defense System Integration
proc testDefenseSystem() =
  echo "Test: Defense Items System" 
  echo "-" & repeat("-", 40)
  
  var env = createTestEnvironment()
  let agent = env.agents[0]
  
  # Test weaving loom (wheat→hat)
  let loom = findObjectOfKind(env, WeavingLoom)
  if loom != nil:
    agent.inventoryWheat = 2
    let adjacentPos = ivec2(loom.pos.x + 1, loom.pos.y)
    moveAgentTo(env, agent, adjacentPos)
    
    var actions: array[MapAgents, array[2, uint8]]
    actions[0] = [5'u8, 3'u8]  # PUT to weaving loom
    env.step(addr actions)
    
    if agent.inventoryHat > 0:
      echo "  ✓ Weaving loom: wheat→hat successful"
    else:
      echo "  ✗ Weaving loom failed"
  
  # Test armory (ore→armor)
  let armory = findObjectOfKind(env, Armory)
  if armory != nil:
    agent.inventoryOre = 1
    let adjacentPos = ivec2(armory.pos.x + 1, armory.pos.y)
    moveAgentTo(env, agent, adjacentPos)
    
    var actions: array[MapAgents, array[2, uint8]]
    actions[0] = [5'u8, 3'u8]  # PUT to armory
    env.step(addr actions)
    
    if agent.inventoryArmor > 0:
      echo "  ✓ Armory: ore→armor successful"
    else:
      echo "  ✗ Armory failed"
  
  echo ""

# Test 4: Combat and Defense Integration  
proc testCombatDefense() =
  echo "Test: Combat and Defense Integration"
  echo "-" & repeat("-", 40)
  
  var env = createTestEnvironment()
  let agent = env.agents[0]
  
  # Give agent defense items
  agent.inventoryHat = 1
  agent.inventoryArmor = 2
  
  # Test defense item detection (combat functions not implemented yet)
  let hasDefenseItems = agent.inventoryArmor > 0 or agent.inventoryHat > 0
  if hasDefenseItems:
    echo "  ✓ Agent has defense items"
  else:
    echo "  ✗ Agent has no defense items"
  
  # Test defense consumption (simulated)
  echo "  ⚠ Combat defense functions not implemented yet - skipping defense consumption tests"
  # let survived1 = defendAgainstAttack(agent, env)
  # if survived1 and agent.inventoryArmor == 1:
  #   echo "  ✓ Armor defense: consumed 1 use, agent survived"
  # else:
  #   echo "  ✗ Armor defense failed"
  
  echo ""

# Test 5: Comprehensive System Integration
proc testFullIntegration() =
  echo "Test: Full System Integration"
  echo "-" & repeat("-", 40)
  
  var env = createTestEnvironment()
  let agent = env.agents[0]
  
  # Full resource chain: mine → converter → altar
  let mine = findObjectOfKind(env, Mine)
  let conv = findObjectOfKind(env, Converter)  
  let altar = findObjectOfKind(env, Altar)
  
  if mine != nil and conv != nil and altar != nil:
    var initialHearts = altar.hearts
    
    # Step 1: GET ore from mine
    let minePos = ivec2(mine.pos.x + 1, mine.pos.y)
    moveAgentTo(env, agent, minePos)
    
    var actions: array[MapAgents, array[2, uint8]]
    actions[0] = [3'u8, 3'u8]  # GET from mine
    env.step(addr actions)
    
    if agent.inventoryOre > 0:
      echo "  ✓ Step 1: Successfully extracted ore from mine"
    
      # Step 2: GET battery from converter (converts ore to battery)
      let convPos = ivec2(conv.pos.x + 1, conv.pos.y)
      moveAgentTo(env, agent, convPos)
      
      actions[0] = [3'u8, 3'u8]  # GET from converter
      env.step(addr actions)
      
      if agent.inventoryBattery > 0 and agent.inventoryOre == 0:
        echo "  ✓ Step 2: Successfully converted ore to battery"
        
        # Step 3: PUT battery to altar (increases hearts)
        let altarPos = ivec2(altar.pos.x + 1, altar.pos.y)
        moveAgentTo(env, agent, altarPos)
        
        actions[0] = [5'u8, 3'u8]  # PUT to altar
        env.step(addr actions)
        
        if agent.inventoryBattery == 0 and altar.hearts > initialHearts:
          echo "  ✓ Step 3: Successfully deposited battery to altar"
          echo fmt"    Altar hearts: {initialHearts} → {altar.hearts}"
          echo "  ✓ Complete resource chain functional!"
        else:
          echo "  ✗ PUT to altar failed"
      else:
        echo "  ✗ Converter failed to convert ore to battery"
    else:
      echo "  ✗ Failed to extract ore from mine"
  else:
    echo "  ⚠ Missing required objects for full test"
  
  echo ""

when isMainModule:
  echo "\n" & "=" & repeat("=", 50)
  echo "Unified Systems Test Suite"
  echo "=" & repeat("=", 50) & "\n"
  
  testGetActionSystem()
  testPutActionSystem()
  testDefenseSystem()
  testCombatDefense()
  testFullIntegration()
  
  echo "=" & repeat("=", 50)
  echo "Unified systems tests completed"