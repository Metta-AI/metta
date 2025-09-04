import std/[strformat, tables, strutils, math, sets, random, unittest]
import vmath
import ../src/mettascope/tribal
import ../src/mettascope/clippy

suite "Clippy Behavior Tests":
  
  test "Concentric Circle Wandering":
    # Test that Clippy wanders in expanding concentric circles
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
      reward: 0.0f32,
      homeAltar: ivec2(-1, -1),
      homeTemple: ivec2(15, 15),  # Temple at center
      wanderRadius: 2,
      wanderAngle: 0.0,
      targetPos: ivec2(-1, -1)
    )
    
    var r = initRand(2024)
    let clippyPtr = cast[pointer](mockClippy.addr)
    
    # Collect positions over multiple wander steps
    var positions: seq[IVec2] = @[]
    var radiusSeen = initHashSet[int]()
    
    for i in 0 ..< 20:
      let wanderPos = getConcentricWanderPoint(clippyPtr, r)
      positions.add(wanderPos)
      
      # Calculate distance from home temple
      let dist = abs(wanderPos.x - mockClippy.homeTemple.x) + 
                 abs(wanderPos.y - mockClippy.homeTemple.y)
      radiusSeen.incl(mockClippy.wanderRadius)
    
    # Check that radius expands over time
    check radiusSeen.len > 1  # Should have multiple radius values
    check 2 in radiusSeen  # Should start with radius 2
    
  test "Clippy Target Prioritization":
    # Test that Clippy prioritizes agents > altars > wandering
    var env = newEnvironment()
    
    # Create a simple test environment
    var things: seq[pointer] = @[]
    
    # Add an agent nearby
    var agent = (
      kind: 0,  # Agent kind
      pos: ivec2(12, 10),
      id: 1,
      layer: 0,
      hearts: 0,
      resources: 0,
      cooldown: 0,
      agentId: 0,
      orientation: 0,
      inventoryOre: 0,
      inventoryBattery: 0,
      inventoryWater: 0,
      inventoryWheat: 0,
      inventoryWood: 0,
      reward: 0.0f32,
      homeAltar: ivec2(-1, -1),
      homeTemple: ivec2(-1, -1),
      wanderRadius: 0,
      wanderAngle: 0.0,
      targetPos: ivec2(-1, -1)
    )
    things.add(cast[pointer](agent.addr))
    
    # Add an altar further away
    var altar = (
      kind: 4,  # Altar kind  
      pos: ivec2(20, 10),
      id: 2,
      layer: 0,
      hearts: 5,
      resources: 0,
      cooldown: 0,
      agentId: -1,
      orientation: 0,
      inventoryOre: 0,
      inventoryBattery: 0,
      inventoryWater: 0,
      inventoryWheat: 0,
      inventoryWood: 0,
      reward: 0.0f32,
      homeAltar: ivec2(-1, -1),
      homeTemple: ivec2(-1, -1),
      wanderRadius: 0,
      wanderAngle: 0.0,
      targetPos: ivec2(-1, -1)
    )
    things.add(cast[pointer](altar.addr))
    
    # Add the clippy itself
    var clippy = (
      kind: 6,  # Clippy kind
      pos: ivec2(10, 10),
      id: 3,
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
      reward: 0.0f32,
      homeAltar: ivec2(-1, -1),
      homeTemple: ivec2(5, 5),
      wanderRadius: 2,
      wanderAngle: 0.0,
      targetPos: ivec2(-1, -1)
    )
    things.add(cast[pointer](clippy.addr))
    
    var r = initRand(2024)
    let moveDir = getClippyMoveDirection(clippy.pos, things, r)
    
    # Clippy should move toward the agent (priority target)
    check moveDir.x > 0  # Should move right toward agent
    check moveDir.y == 0  # Agent is on same y level
    
  test "Clippy Manhattan Distance Calculation":
    # Test Manhattan distance function
    let pos1 = ivec2(5, 5)
    let pos2 = ivec2(10, 8)
    let dist = manhattanDistance(pos1, pos2)
    
    check dist == 8  # |10-5| + |8-5| = 5 + 3 = 8
    
  test "Find Nearest Agent":
    var things: seq[pointer] = @[]
    
    # Add multiple agents at different distances
    var agent1 = (
      kind: 0,  # Agent
      pos: ivec2(13, 10),
      id: 1
    )
    things.add(cast[pointer](agent1.addr))
    
    var agent2 = (
      kind: 0,  # Agent
      pos: ivec2(11, 11),
      id: 2
    )
    things.add(cast[pointer](agent2.addr))
    
    var nonAgent = (
      kind: 2,  # Wall
      pos: ivec2(10, 11),
      id: 3
    )
    things.add(cast[pointer](nonAgent.addr))
    
    let clippyPos = ivec2(10, 10)
    let nearest = findNearestAgent(clippyPos, things, 5)
    
    # Should find agent2 as it's closer (distance 2 vs 3)
    check nearest == ivec2(11, 11)
    
  test "Find Nearest Altar":
    var things: seq[pointer] = @[]
    
    # Add altars at different distances
    var altar1 = (
      kind: 4,  # Altar
      pos: ivec2(15, 10),
      id: 1
    )
    things.add(cast[pointer](altar1.addr))
    
    var altar2 = (
      kind: 4,  # Altar  
      pos: ivec2(11, 12),
      id: 2
    )
    things.add(cast[pointer](altar2.addr))
    
    let clippyPos = ivec2(10, 10)
    let nearest = findNearestAltar(clippyPos, things, 10)
    
    # Should find altar2 as it's closer (distance 3 vs 5)
    check nearest == ivec2(11, 12)
    
  test "Avoidance Direction Calculation":
    # Test that Clippy calculates correct avoidance direction
    let clippyPos = ivec2(10, 10)
    let otherClippies = @[
      ivec2(12, 10),  # Clippy to the right
      ivec2(11, 10)   # Another to the right
    ]
    
    let avoidDir = getAvoidanceDirection(clippyPos, otherClippies)
    
    # Should move left (away from clippies on the right)
    check avoidDir.x < 0
    check avoidDir.y == 0

# Run the tests
when isMainModule:
  echo "\n" & repeat("=", 60)
  echo "Running Clippy Behavior Tests"
  echo repeat("=", 60)