import
  std/[os, json, algorithm, tables, sets, strutils, strformat],
  vmath, silky, windy,
  common, panels, replays

const InventoryScale = 0.5f

type
  ResourceLimitGroup* = object
    name*: string
    limit*: int
    resources*: seq[string]
    modifiers*: Table[string, int]

proc parseResourceLimits(mgConfig: JsonNode): seq[ResourceLimitGroup] =
  ## Parse resource_limits from the agent config.
  result = @[]
  if mgConfig.isNil:
    return
  if "game" notin mgConfig or "agent" notin mgConfig["game"]:
    return
  let agentConfig = mgConfig["game"]["agent"]
  if "resource_limits" notin agentConfig:
    return
  let resourceLimits = agentConfig["resource_limits"]
  for groupName, groupConfig in resourceLimits.pairs:
    var group = ResourceLimitGroup(name: groupName)
    if "limit" in groupConfig:
      group.limit = groupConfig["limit"].getInt
    if "resources" in groupConfig:
      for r in groupConfig["resources"]:
        group.resources.add(r.getStr)
    if "modifiers" in groupConfig:
      group.modifiers = initTable[string, int]()
      for k, v in groupConfig["modifiers"].pairs:
        group.modifiers[k] = v.getInt
    result.add(group)

proc computeEffectiveLimit(group: ResourceLimitGroup, inventory: seq[ItemAmount], itemNames: seq[string]): int =
  ## Compute effective limit based on base limit + modifiers from inventory.
  result = group.limit
  for modItem, bonus in group.modifiers.pairs:
    for itemAmount in inventory:
      if itemAmount.itemId >= 0 and itemAmount.itemId < itemNames.len:
        if itemNames[itemAmount.itemId] == modItem:
          result += bonus * itemAmount.count

proc getItemName(itemAmount: ItemAmount): string =
  ## Safely resolve an item name from the replay data.
  if replay.isNil:
    return "item#" & $itemAmount.itemId
  if itemAmount.itemId >= 0 and itemAmount.itemId < replay.itemNames.len:
    replay.itemNames[itemAmount.itemId]
  else:
    "item#" & $itemAmount.itemId

proc formatItem(itemAmount: ItemAmount): string =
  ## Render a compact "name x count" string.
  let name = getItemName(itemAmount)
  name & " x" & $itemAmount.count

proc showItem(itemAmount: ItemAmount) =
  icon("resources/" & replay.config.game.resourceNames[itemAmount.itemId])
  text("x" & $itemAmount.count)

proc getHeartCount(outputs: seq[ItemAmount]): int =
  ## Returns total hearts produced by this protocol.
  let heartId = replay.itemNames.find("heart")
  if heartId == -1:
    return 0
  for output in outputs:
    if output.itemId == heartId:
      return output.count
  return 0

proc protocolCmp(a, b: Protocol): int =
  ## Sort protocols: heart-producing ones first (most hearts first), then others.
  let
    aHearts = getHeartCount(a.outputs)
    bHearts = getHeartCount(b.outputs)
  if aHearts > 0 and bHearts == 0:
    return -1
  if aHearts == 0 and bHearts > 0:
    return 1
  cmp(bHearts, aHearts)

proc drawObjectInfo*(panel: Panel, frameId: string, contentPos: Vec2, contentSize: Vec2) =
  ## Draws the object info panel using silky widgets.
  frame(frameId, contentPos, contentSize):
    if selection.isNil:
      text("No selection")
      return

    if replay.isNil:
      text("Replay not loaded")
      return

    let cur = selection

    button("Open Config"):
      if cur.isNil:
        return
      let cfgText =
        if replay.isNil or replay.mgConfig.isNil:
          "No replay config found."
        else:
          let typeName = cur.typeName
          if typeName == "agent":
            let agentConfig = replay.mgConfig["game"]["agent"]
            agentConfig.pretty
          else:
            if "game" notin replay.mgConfig or "objects" notin replay.mgConfig["game"]:
              "No object config found."
            elif typeName notin replay.mgConfig["game"]["objects"]:
              "Object config not found for type: " & typeName
            else:
              let objConfig = replay.mgConfig["game"]["objects"][typeName]
              objConfig.pretty
      openTempTextFile(cur.typeName & "_config.json", cfgText)

    # Basic identity
    h1text(cur.typeName)
    text(&"  Object ID: {cur.id}")

    if cur.isAgent:
      # Agent-specific info.
      let reward = cur.totalReward.at
      text(&"  Agent ID: {cur.agentId}")
      text(&"  Total reward: {formatFloat(reward, ffDecimal, 2)}")
      let vibeId = cur.vibeId.at
      if vibeId >= 0 and vibeId < replay.config.game.vibeNames.len:
        let vibeName = getVibeName(vibeId)
        text("  Vibe: " & vibeName)
    else:
      # Assembler-specific info.
      let cooldown = cur.cooldownRemaining.at
      if cooldown > 0:
        text(&"  Cooldown remaining: {cooldown}")
      if cur.cooldownDuration > 0:
        text(&"  Cooldown duration: {cur.cooldownDuration}")
      if cur.isClipped.at:
        text("  Is clipped")
      if cur.isClipImmune.at:
        text("  Clip immune")
      if cur.usesCount.at > 0:
        text(&"  Uses: {cur.usesCount.at}" &
          (if cur.maxUses > 0: "/" & $cur.maxUses else: ""))
      elif cur.maxUses > 0:
        text(&"  Max uses: {cur.maxUses}")
      if cur.allowPartialUsage:
        text("  Allows partial usage")

    sk.advance(vec2(0, theme.spacing.float32))

    let currentInventory = cur.inventory.at
    text("Inventory")
    if currentInventory.len == 0:
      text("  Empty")
    else:
      if cur.isAgent:
        let resourceLimitGroups = parseResourceLimits(replay.mgConfig)

        var itemByName = initTable[string, ItemAmount]()
        for itemAmount in currentInventory:
          if itemAmount.itemId >= 0 and itemAmount.itemId < replay.itemNames.len:
            itemByName[replay.itemNames[itemAmount.itemId]] = itemAmount

        var shownItems = initOrderedSet[string]()

        for group in resourceLimitGroups:
          var usedAmount = 0
          var groupItems: seq[ItemAmount] = @[]
          for resourceName in group.resources:
            if resourceName in itemByName:
              let itemAmount = itemByName[resourceName]
              usedAmount += itemAmount.count
              groupItems.add(itemAmount)
              shownItems.incl(resourceName)

          if groupItems.len > 0:
            let effectiveLimit = computeEffectiveLimit(group, currentInventory, replay.itemNames)
            text(&"  {group.name}: {usedAmount}/{effectiveLimit}")
            for itemAmount in groupItems:
              if itemAmount.itemId != replay.itemNames.find("energy"):
                text("    " & formatItem(itemAmount))

        var ungroupedItems: seq[ItemAmount] = @[]
        for itemAmount in currentInventory:
          if itemAmount.itemId >= 0 and itemAmount.itemId < replay.itemNames.len:
            let itemName = replay.itemNames[itemAmount.itemId]
            if itemName notin shownItems:
              ungroupedItems.add(itemAmount)

        if ungroupedItems.len > 0:
          text("  Other:")
          for itemAmount in ungroupedItems:
            text("    " & formatItem(itemAmount))
      else:
        for itemAmount in currentInventory:
          text("  " & formatItem(itemAmount))

    sk.advance(vec2(0, theme.spacing.float32))

    # Protocols
    if cur.protocols.len > 0:
      text("Protocols")
      var sortedProtocols = cur.protocols
      sortedProtocols.sort(protocolCmp)

      for protocol in sortedProtocols:
        let protocol = protocol
        group(vec2(4, 4), LeftToRight):
          if protocol.vibes.len > 0:
            #var vibeLine = "  Vibes: "
            # Group the vibes by type.
            var vibeGroups: Table[string, int]
            for vibe in protocol.vibes:
              let vibeName = getVibeName(vibe)
              if vibeName notin vibeGroups:
                vibeGroups[vibeName] = 1
              else:
                vibeGroups[vibeName] = vibeGroups[vibeName] + 1
            for vibeName, numVibes in vibeGroups:
              icon("vibe/" & vibeName)
              text("x" & $numVibes)

            icon("ui/add")

          if protocol.inputs.len > 0:
            for i, resource in protocol.inputs:
              icon("resources/" & replay.config.game.resourceNames[resource.itemId])
              text("x" & $resource.count)

            icon("ui/right-arrow")

          if protocol.outputs.len > 0:
            for i, resource in protocol.outputs:
              icon("resources/" & replay.config.game.resourceNames[resource.itemId])
              text("x" & $resource.count)


proc selectObject*(obj: Entity) =
  selection = obj
