## Collectives panel displays information about each collective (team/faction).
## Shows inventory and building counts by type.

import
  std/[strformat, tables, algorithm, json],
  silky, vmath,
  common, panels, replays, aoepanel

type
  CollectiveStats* = object
    collectiveId*: int
    name*: string
    collectiveInventory*: Table[string, int]  ## itemName -> count (collective's own inventory at current step)
    buildingsByType*: Table[string, int]  ## typeName -> count

proc getCollectiveInitialInventory(collectiveId: int): Table[string, int] =
  ## Get the initial inventory of a collective from the config.
  result = initTable[string, int]()
  let collectiveConfig = getCollectiveConfig(collectiveId)
  if collectiveConfig.isNil or collectiveConfig.kind != JObject:
    return
  if "inventory" notin collectiveConfig:
    return
  let inventoryConfig = collectiveConfig["inventory"]
  if inventoryConfig.kind != JObject or "initial" notin inventoryConfig:
    return
  let initial = inventoryConfig["initial"]
  if initial.kind != JObject:
    return
  for key, value in initial.pairs:
    if value.kind == JInt:
      result[key] = value.getInt

proc getCollectiveLiveInventory(collectiveName: string, currentStep: int): Table[string, int] =
  ## Get the live inventory of a collective at the current step.
  ## Uses the most recent snapshot at or before the current step.
  result = initTable[string, int]()
  if replay.isNil:
    return
  if collectiveName notin replay.collectiveInventory:
    return
  let snapshots = replay.collectiveInventory[collectiveName]
  # Find the most recent snapshot at or before currentStep
  for i in countdown(snapshots.len - 1, 0):
    if snapshots[i].step <= currentStep:
      return snapshots[i].inventory
  # No snapshot found before currentStep, return empty

proc getCollectiveStats*(): seq[CollectiveStats] =
  ## Collect statistics for all collectives from the replay.
  result = @[]
  if replay.isNil:
    return
  let numCollectives = getNumCollectives()
  # Initialize stats for each collective.
  for i in 0 ..< numCollectives:
    let collectiveName = getCollectiveName(i)
    # Try to get live inventory first, fall back to initial
    var inv = getCollectiveLiveInventory(collectiveName, step)
    if inv.len == 0:
      inv = getCollectiveInitialInventory(i)
    var stats = CollectiveStats(
      collectiveId: i,
      name: collectiveName,
      collectiveInventory: inv,
      buildingsByType: initTable[string, int]()
    )
    result.add(stats)
  # Iterate all objects and aggregate stats by collective.
  # Use index-based iteration to avoid issues with seq being modified elsewhere.
  let numObjects = replay.objects.len
  for i in 0 ..< numObjects:
    let obj = replay.objects[i]
    if obj.collectiveId < 0 or obj.collectiveId >= numCollectives:
      continue
    # Skip agents as they are not buildings.
    if obj.isAgent:
      continue
    # Count buildings by type.
    if result[obj.collectiveId].buildingsByType.hasKey(obj.typeName):
      result[obj.collectiveId].buildingsByType[obj.typeName] += 1
    else:
      result[obj.collectiveId].buildingsByType[obj.typeName] = 1

proc drawCollectivesPanel*(panel: Panel, frameId: string, contentPos: Vec2, contentSize: Vec2) =
  ## Draw the collectives panel showing stats for each collective.
  frame(frameId, contentPos, contentSize):
    if replay.isNil:
      text("Replay not loaded")
      return
    let numCollectives = getNumCollectives()
    if numCollectives == 0:
      text("No collectives configured")
      return
    let allStats = getCollectiveStats()
    for stats in allStats:
      # Header with collective name.
      let displayName = if stats.name.len > 0: stats.name else: &"Collective {stats.collectiveId}"
      h1text(displayName)
      # Collective's own inventory.
      if stats.collectiveInventory.len > 0:
        text("  Inventory:")
        var sortedItems: seq[(string, int)] = @[]
        for itemName, count in stats.collectiveInventory.pairs:
          sortedItems.add((itemName, count))
        sortedItems.sort(proc(a, b: (string, int)): int = cmp(b[1], a[1]))
        for (itemName, count) in sortedItems:
          text(&"    {itemName}: {count}")
      # Building counts by type.
      if stats.buildingsByType.len > 0:
        text("  Buildings:")
        var sortedTypes: seq[(string, int)] = @[]
        for typeName, count in stats.buildingsByType.pairs:
          sortedTypes.add((typeName, count))
        sortedTypes.sort(proc(a, b: (string, int)): int = cmp(b[1], a[1]))
        for (typeName, count) in sortedTypes:
          text(&"    {typeName}: {count}")
      sk.advance(vec2(0, theme.spacing.float32))
