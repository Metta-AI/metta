import
  vmath,
  mettascope/[heatmap, replays]

type
  TestMap = object
    width: int
    height: int
    walkable: seq[bool]

proc setupTestReplay(mapSize: (int, int), numAgents: int, maxSteps: int): Replay =
  ## Create a test replay with the given parameters.
  result = Replay(
    version: 2,
    numAgents: numAgents,
    maxSteps: maxSteps,
    mapSize: mapSize,
    fileName: "test",
    objects: @[],
    agents: @[],
  )

proc createTestAgent(id: int, locations: seq[IVec2]): Entity =
  ## Create a test agent with the given ID and location history.
  result = Entity(
    id: id,
    typeName: "agent",
    agentId: id,
    isAgent: true,
    location: locations,
    orientation: newSeq[int](locations.len),
    inventory: newSeq[seq[ItemAmount]](locations.len),
    inventoryMax: 10,
    color: newSeq[int](locations.len),
    actionId: newSeq[int](locations.len),
    actionParameter: newSeq[int](locations.len),
    actionSuccess: newSeq[bool](locations.len),
    currentReward: newSeq[float](locations.len),
    totalReward: newSeq[float](locations.len),
    isFrozen: newSeq[bool](locations.len),
    frozenProgress: newSeq[int](locations.len),
    frozenTime: 0,
    visionSize: 11,
  )

  # Initialize arrays with default values
  for i in 0 ..< locations.len:
    result.orientation[i] = 0
    result.inventory[i] = @[]
    result.color[i] = id
    result.actionId[i] = 0
    result.actionParameter[i] = 0
    result.actionSuccess[i] = true
    result.currentReward[i] = 0.0
    result.totalReward[i] = 0.0
    result.isFrozen[i] = false
    result.frozenProgress[i] = 0

block basic_tests:
  block heatmap_initialization:
    echo "Testing heatmap initialization"

    # Test with empty replay
    let replay = setupTestReplay((10, 10), 0, 5)
    let heatmap = newHeatmap(replay)
    heatmap.initialize(replay)

    # Should have correct dimensions
    doAssert heatmap.width == 10, "heatmap width should match map width"
    doAssert heatmap.height == 10, "heatmap height should match map height"
    doAssert heatmap.maxSteps == 5, "heatmap maxSteps should match replay maxSteps"
    doAssert heatmap.data.len == 5, "heatmap data should have maxSteps arrays"

    # All heat values should be 0 with no agents
    for step in 0 ..< 5:
      for y in 0 ..< 10:
        for x in 0 ..< 10:
          doAssert heatmap.getHeat(step, x, y) == 0, "heat at (" & $x & "," & $y & ") step " & $step & " should be 0"

    # Max heat should be 0 for all steps
    for step in 0 ..< 5:
      doAssert heatmap.getMaxHeat(step) == 0, "max heat for step " & $step & " should be 0"

  block single_agent_movement:
    echo "Testing single agent movement"

    let replay = setupTestReplay((5, 5), 1, 3)
    let agent = createTestAgent(0, @[ivec2(0, 0), ivec2(1, 0), ivec2(2, 0)])
    replay.agents.add(agent)

    let heatmap = newHeatmap(replay)
    heatmap.initialize(replay)

    # Step 0: agent at (0,0) should have heat 1
    doAssert heatmap.getHeat(0, 0, 0) == 1, "agent position should have heat 1"
    doAssert heatmap.getHeat(0, 1, 0) == 0, "non-agent position should have heat 0"
    doAssert heatmap.getMaxHeat(0) == 1, "max heat for step 0 should be 1"

    # Step 1: agent at (1,0) should have heat 1, cumulative from step 0
    doAssert heatmap.getHeat(1, 0, 0) == 1, "previous position should retain heat"
    doAssert heatmap.getHeat(1, 1, 0) == 1, "current position should have heat 1"
    doAssert heatmap.getHeat(1, 2, 0) == 0, "non-agent position should have heat 0"
    doAssert heatmap.getMaxHeat(1) == 1, "max heat for step 1 should be 1"

    # Step 2: agent at (2,0) should have heat 1, cumulative from previous steps
    doAssert heatmap.getHeat(2, 0, 0) == 1, "step 0 position should retain heat"
    doAssert heatmap.getHeat(2, 1, 0) == 1, "step 1 position should retain heat"
    doAssert heatmap.getHeat(2, 2, 0) == 1, "current position should have heat 1"
    doAssert heatmap.getMaxHeat(2) == 1, "max heat for step 2 should be 1"

  block multiple_agents_same_tile:
    echo "Testing multiple agents on same tile"

    let replay = setupTestReplay((3, 3), 2, 2)
    let agent1 = createTestAgent(0, @[ivec2(1, 1), ivec2(1, 1)])
    let agent2 = createTestAgent(1, @[ivec2(1, 1), ivec2(1, 1)])
    replay.agents.add(agent1)
    replay.agents.add(agent2)

    let heatmap = newHeatmap(replay)
    heatmap.initialize(replay)

    # Step 0: 2 agents at (1,1) -> heat = 2
    # Step 1: agents still at (1,1) but no increment when standing still -> heat = 2
    doAssert heatmap.getHeat(0, 1, 1) == 2, "position with 2 agents should have heat 2"
    doAssert heatmap.getHeat(1, 1, 1) == 2, "standing still should not accumulate additional heat"
    doAssert heatmap.getMaxHeat(0) == 2, "max heat should be 2"
    doAssert heatmap.getMaxHeat(1) == 2, "max heat should remain 2"

  block cumulative_heat:
    echo "Testing cumulative heat behavior"

    let replay = setupTestReplay((4, 4), 1, 3)
    let agent = createTestAgent(0, @[ivec2(0, 0), ivec2(0, 0), ivec2(1, 1)])
    replay.agents.add(agent)

    let heatmap = newHeatmap(replay)
    heatmap.initialize(replay)

    # Step 0: agent at (0,0) for first time
    doAssert heatmap.getHeat(0, 0, 0) == 1, "first visit should have heat 1"

    # Step 1: agent at (0,0) again, but no increment when standing still
    doAssert heatmap.getHeat(1, 0, 0) == 1, "standing still should not accumulate heat"

    # Step 2: agent moved to (1,1), (0,0) retains heat 1, (1,1) gets heat 1
    doAssert heatmap.getHeat(2, 0, 0) == 1, "(0,0) should retain heat 1"
    doAssert heatmap.getHeat(2, 1, 1) == 1, "(1,1) should have heat 1"
    doAssert heatmap.getMaxHeat(2) == 1, "max heat should be 1"

  block boundary_conditions:
    echo "Testing boundary conditions"

    let replay = setupTestReplay((3, 3), 1, 2)
    let agent = createTestAgent(0, @[ivec2(0, 0), ivec2(2, 2)])
    replay.agents.add(agent)

    let heatmap = newHeatmap(replay)
    heatmap.initialize(replay)

    # Test corners
    doAssert heatmap.getHeat(0, 0, 0) == 1, "corner (0,0) should work"
    doAssert heatmap.getHeat(1, 2, 2) == 1, "corner (2,2) should work"

    # Test out of bounds access returns 0
    doAssert heatmap.getHeat(0, -1, 0) == 0, "out of bounds x should return 0"
    doAssert heatmap.getHeat(0, 0, -1) == 0, "out of bounds y should return 0"
    doAssert heatmap.getHeat(0, 3, 0) == 0, "out of bounds x should return 0"
    doAssert heatmap.getHeat(0, 0, 3) == 0, "out of bounds y should return 0"
    doAssert heatmap.getMaxHeat(-1) == 0, "invalid step should return 0"
    doAssert heatmap.getMaxHeat(10) == 0, "out of bounds step should return 0"

  block realtime_update:
    echo "Testing realtime update functionality"

    let replay = setupTestReplay((3, 3), 1, 1) # Start with 1 step
    let agent = createTestAgent(0, @[ivec2(0, 0)])
    replay.agents.add(agent)

    let heatmap = newHeatmap(replay)
    heatmap.initialize(replay)

    # Initially 1 step
    doAssert heatmap.maxSteps == 1, "should start with 1 step"
    doAssert heatmap.getHeat(0, 0, 0) == 1, "initial position should have heat"

    # Update for step 1 (extend the heatmap)
    let newAgent = createTestAgent(0, @[ivec2(0, 0), ivec2(1, 1)])
    replay.agents[0] = newAgent
    replay.maxSteps = 2
    heatmap.update(1, replay)

    doAssert heatmap.maxSteps == 2, "should extend to 2 steps"
    doAssert heatmap.getHeat(0, 0, 0) == 1, "original step should be preserved"
    doAssert heatmap.getHeat(1, 0, 0) == 1, "previous position should carry over"
    doAssert heatmap.getHeat(1, 1, 1) == 1, "new position should have heat"
    doAssert heatmap.getMaxHeat(1) == 1, "max heat should be 1"

echo "All heatmap tests passed!"
