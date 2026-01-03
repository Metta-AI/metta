## Commons panel displays information about each commons (team/faction).
## Shows inventory and building counts by type.

import
  std/[strformat, tables, algorithm, json],
  silky, vmath,
  common, panels, replays, aoepanel

type
  CommonsStats* = object
    commonsId*: int
    name*: string
    commonsInventory*: Table[string, int]  ## itemName -> count (commons' own inventory at current step)
    buildingsByType*: Table[string, int]  ## typeName -> count

proc getCommonsInitialInventory(commonsId: int): Table[string, int] =
  ## Get the initial inventory of a commons from the config.
  result = initTable[string, int]()
  if replay.isNil or replay.mgConfig.isNil:
    return
  if "game" notin replay.mgConfig or "commons" notin replay.mgConfig["game"]:
    return
  let commonsArr = replay.mgConfig["game"]["commons"]
  if commonsArr.kind != JArray or commonsId < 0 or commonsId >= commonsArr.len:
    return
  let commonsConfig = commonsArr[commonsId]
  if commonsConfig.kind != JObject:
    return
  if "inventory" notin commonsConfig:
    return
  let inventoryConfig = commonsConfig["inventory"]
  if inventoryConfig.kind != JObject or "initial" notin inventoryConfig:
    return
  let initial = inventoryConfig["initial"]
  if initial.kind != JObject:
    return
  for key, value in initial.pairs:
    if value.kind == JInt:
      result[key] = value.getInt

proc getCommonsLiveInventory(commonsName: string, currentStep: int): Table[string, int] =
  ## Get the live inventory of a commons at the current step.
  ## Uses the most recent snapshot at or before the current step.
  result = initTable[string, int]()
  if replay.isNil:
    return
  if commonsName notin replay.commonsInventory:
    return
  let snapshots = replay.commonsInventory[commonsName]
  # Find the most recent snapshot at or before currentStep
  for i in countdown(snapshots.len - 1, 0):
    if snapshots[i].step <= currentStep:
      return snapshots[i].inventory
  # No snapshot found before currentStep, return empty

proc getCommonsStats*(): seq[CommonsStats] =
  ## Collect statistics for all commons from the replay.
  result = @[]
  if replay.isNil:
    return
  let numCommons = getNumCommons()
  # Initialize stats for each commons.
  for i in 0 ..< numCommons:
    let commonsName = getCommonsName(i)
    # Try to get live inventory first, fall back to initial
    var inv = getCommonsLiveInventory(commonsName, step)
    if inv.len == 0:
      inv = getCommonsInitialInventory(i)
    var stats = CommonsStats(
      commonsId: i,
      name: commonsName,
      commonsInventory: inv,
      buildingsByType: initTable[string, int]()
    )
    result.add(stats)
  # Iterate all objects and aggregate stats by commons.
  # Use index-based iteration to avoid issues with seq being modified elsewhere.
  let numObjects = replay.objects.len
  for i in 0 ..< numObjects:
    let obj = replay.objects[i]
    if obj.commonsId < 0 or obj.commonsId >= numCommons:
      continue
    # Skip agents as they are not buildings.
    if obj.isAgent:
      continue
    # Count buildings by type.
    if result[obj.commonsId].buildingsByType.hasKey(obj.typeName):
      result[obj.commonsId].buildingsByType[obj.typeName] += 1
    else:
      result[obj.commonsId].buildingsByType[obj.typeName] = 1

proc drawCommonsPanel*(panel: Panel, frameId: string, contentPos: Vec2, contentSize: Vec2) =
  ## Draw the commons panel showing stats for each commons.
  frame(frameId, contentPos, contentSize):
    if replay.isNil:
      text("Replay not loaded")
      return
    let numCommons = getNumCommons()
    if numCommons == 0:
      text("No commons configured")
      return
    let allStats = getCommonsStats()
    for stats in allStats:
      # Header with commons name.
      let displayName = if stats.name.len > 0: stats.name else: &"Commons {stats.commonsId}"
      h1text(displayName)
      # Commons' own inventory.
      if stats.commonsInventory.len > 0:
        text("  Inventory:")
        var sortedItems: seq[(string, int)] = @[]
        for itemName, count in stats.commonsInventory.pairs:
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
