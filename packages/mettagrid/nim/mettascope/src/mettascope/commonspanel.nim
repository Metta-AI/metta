## Commons panel displays information about each commons (team/faction).
## Shows inventory, member counts by type, and other aggregate stats.

import
  std/[strformat, tables, algorithm],
  silky, vmath,
  common, panels, replays, aoepanel

type
  CommonsStats* = object
    commonsId*: int
    name*: string
    inventory*: Table[int, int]  ## itemId -> total count
    buildingsByType*: Table[string, int]  ## typeName -> count

proc getCommonsStats*(): seq[CommonsStats] =
  ## Collect statistics for all commons from the replay.
  result = @[]
  if replay.isNil:
    return
  let numCommons = getNumCommons()
  # Initialize stats for each commons.
  for i in 0 ..< numCommons:
    var stats = CommonsStats(
      commonsId: i,
      name: getCommonsName(i),
      inventory: initTable[int, int](),
      buildingsByType: initTable[string, int]()
    )
    result.add(stats)
  # Iterate all objects and aggregate stats by commons.
  for obj in replay.objects:
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
    # Aggregate inventory.
    let currentInventory = obj.inventory.at(step)
    for item in currentInventory:
      if result[obj.commonsId].inventory.hasKey(item.itemId):
        result[obj.commonsId].inventory[item.itemId] += item.count
      else:
        result[obj.commonsId].inventory[item.itemId] = item.count

proc getItemName(itemId: int): string =
  ## Get item name by ID from the replay.
  if replay.isNil:
    return "item#" & $itemId
  if itemId >= 0 and itemId < replay.itemNames.len:
    replay.itemNames[itemId]
  else:
    "item#" & $itemId

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
      # Building counts by type.
      if stats.buildingsByType.len > 0:
        text("  Buildings:")
        var sortedTypes: seq[(string, int)] = @[]
        for typeName, count in stats.buildingsByType.pairs:
          sortedTypes.add((typeName, count))
        sortedTypes.sort(proc(a, b: (string, int)): int = cmp(b[1], a[1]))
        for (typeName, count) in sortedTypes:
          text(&"    {typeName}: {count}")
      else:
        text("  No buildings")
      # Inventory.
      if stats.inventory.len > 0:
        text("  Inventory:")
        var sortedItems: seq[(int, int)] = @[]
        for itemId, count in stats.inventory.pairs:
          sortedItems.add((itemId, count))
        sortedItems.sort(proc(a, b: (int, int)): int = cmp(b[1], a[1]))
        for (itemId, count) in sortedItems:
          let itemName = getItemName(itemId)
          text(&"    {itemName}: {count}")
      else:
        text("  No inventory")
      sk.advance(vec2(0, theme.spacing.float32))

