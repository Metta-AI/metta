import ../src/mettascope/attack, vmath, std/random

# Test the attack system with forge and spears
proc testAttackSystem() =
  echo "Testing Attack System with Forge and Spears"
  echo "==========================================="
  
  # Create environment
  let env = newEnvironment()
  
  # Find a forge and an agent
  var forge: Thing = nil
  var agent: Thing = nil
  var clippy: Thing = nil
  
  for thing in env.things:
    if thing.kind == Forge and forge == nil:
      forge = thing
      echo "Found Forge at position: ", forge.pos.x, ", ", forge.pos.y
    elif thing.kind == Agent and agent == nil and thing.agentId == 0:
      agent = thing
      echo "Found Agent 0 at position: ", agent.pos.x, ", ", agent.pos.y
    elif thing.kind == Clippy and clippy == nil:
      clippy = thing
      echo "Found Clippy at position: ", clippy.pos.x, ", ", clippy.pos.y
  
  if forge == nil:
    echo "ERROR: No forge found in the environment!"
    return
  
  if agent == nil:
    echo "ERROR: No agent found!"
    return
  
  echo "\n--- Initial State ---"
  echo "Agent wood inventory: ", agent.inventoryWood
  echo "Agent spear inventory: ", agent.inventorySpear
  echo "Forge cooldown: ", forge.cooldown
  
  # Give agent some wood for testing
  echo "\n--- Giving Agent Wood ---"
  agent.inventoryWood = 3
  echo "Agent wood inventory: ", agent.inventoryWood
  
  # Test 1: Craft a spear at the forge
  echo "\n--- Test 1: Crafting Spear at Forge ---"
  
  # Move agent to adjacent to forge if not already
  let forgeAdjacent = @[
    forge.pos + ivec2(0, -1),  # North
    forge.pos + ivec2(0, 1),   # South
    forge.pos + ivec2(-1, 0),  # West
    forge.pos + ivec2(1, 0)    # East
  ]
  
  var foundValidPos = false
  for adjPos in forgeAdjacent:
    if adjPos.x >= 0 and adjPos.x < MapWidth and 
       adjPos.y >= 0 and adjPos.y < MapHeight and
       env.isEmpty(adjPos):
      # Move agent to this position
      env.grid[agent.pos.x][agent.pos.y] = nil
      agent.pos = adjPos
      env.grid[agent.pos.x][agent.pos.y] = agent
      foundValidPos = true
      echo "Moved agent to position adjacent to forge: ", agent.pos.x, ", ", agent.pos.y
      break
  
  if not foundValidPos:
    echo "Could not find valid position adjacent to forge"
    return
  
  # Use the forge
  env.useForgeAction(0, agent, forge)
  
  echo "After using forge:"
  echo "  Agent wood inventory: ", agent.inventoryWood
  echo "  Agent spear inventory: ", agent.inventorySpear
  echo "  Forge cooldown: ", forge.cooldown
  echo "  Agent reward: ", agent.reward
  
  # Test 2: Attack with spear
  echo "\n--- Test 2: Attack with Spear ---"
  
  if clippy != nil:
    echo "Clippy position: ", clippy.pos.x, ", ", clippy.pos.y
    echo "Agent position: ", agent.pos.x, ", ", agent.pos.y
    
    # Calculate direction to clippy
    let dx = clippy.pos.x - agent.pos.x
    let dy = clippy.pos.y - agent.pos.y
    let distance = abs(dx) + abs(dy)  # Manhattan distance
    
    echo "Distance to Clippy: ", distance
    
    if distance <= 2:
      # Determine attack direction
      var attackDir = -1
      if abs(dx) > abs(dy):
        if dx > 0:
          attackDir = 2  # East
        else:
          attackDir = 3  # West
      else:
        if dy > 0:
          attackDir = 1  # South
        else:
          attackDir = 0  # North
      
      if attackDir >= 0:
        echo "Attacking in direction: ", attackDir
        let initialThingCount = env.things.len
        env.attackWithSpearAction(0, agent, attackDir)
        echo "After attack:"
        echo "  Agent spear inventory: ", agent.inventorySpear
        echo "  Agent reward: ", agent.reward
        echo "  Things count: ", env.things.len, " (was ", initialThingCount, ")"
        
        # Check if Clippy was destroyed
        var clippyStillExists = false
        for thing in env.things:
          if thing == clippy:
            clippyStillExists = true
            break
        
        if not clippyStillExists:
          echo "  ✓ Clippy was successfully destroyed!"
        else:
          echo "  ✗ Clippy still exists (attack may have missed)"
    else:
      echo "Clippy is too far away for spear attack (need distance <= 2)"
  else:
    echo "No Clippy found to test attack"
  
  # Test 3: Try to craft another spear while on cooldown
  echo "\n--- Test 3: Forge Cooldown Test ---"
  agent.inventoryWood = 2  # Give more wood
  echo "Agent wood inventory: ", agent.inventoryWood
  echo "Forge cooldown: ", forge.cooldown
  
  env.useForgeAction(0, agent, forge)
  echo "After trying to use forge during cooldown:"
  echo "  Agent wood inventory: ", agent.inventoryWood
  echo "  Agent spear inventory: ", agent.inventorySpear
  echo "  (Should be unchanged due to cooldown)"
  
  # Simulate time passing to reduce cooldown
  echo "\n--- Simulating 5 steps to clear cooldown ---"
  for i in 0 ..< 5:
    if forge.cooldown > 0:
      forge.cooldown -= 1
  echo "Forge cooldown after 5 steps: ", forge.cooldown
  
  if forge.cooldown == 0:
    echo "Trying to craft again..."
    env.useForgeAction(0, agent, forge)
    echo "After using forge:"
    echo "  Agent wood inventory: ", agent.inventoryWood
    echo "  Agent spear inventory: ", agent.inventorySpear
  
  echo "\n=== Test Complete ==="

when isMainModule:
  testAttackSystem()