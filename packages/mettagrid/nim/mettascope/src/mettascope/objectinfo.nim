import
  std/[json, algorithm, tables, sets, strutils, strformat],
  vmath, silky, windy, chroma,
  common, panels, replays

type
  ResourceLimitGroup* = object
    name*: string
    limit*: int
    resources*: seq[string]
    modifiers*: Table[string, int]

  DeltaSource* = object
    sourceName*: string
    sourcePos*: IVec2
    delta*: int

const
  DeltaPosColor = rgbx(100, 255, 100, 255)
  DeltaNegColor = rgbx(255, 100, 100, 255)

proc textWithDelta(baseText: string, delta: int) =
  ## Draw text with an optional colored delta suffix.
  let startPos = sk.at
  text(baseText)
  if delta != 0:
    let deltaStr = if delta > 0: " (+" & $delta & ")" else: " (" & $delta & ")"
    let color = if delta > 0: DeltaPosColor else: DeltaNegColor
    let baseWidth = sk.getTextSize("Default", baseText).x
    discard sk.drawText("Default", deltaStr, startPos + vec2(baseWidth, 0), color)

proc parseInventoryLimits(mgConfig: JsonNode): seq[ResourceLimitGroup] =
  ## Parse inventory.limits from the agent config.
  result = @[]
  if mgConfig.isNil:
    return
  if "game" notin mgConfig or "agent" notin mgConfig["game"]:
    return
  let agentConfig = mgConfig["game"]["agent"]
  if "inventory" notin agentConfig:
    return
  let invConfig = agentConfig["inventory"]
  if "limits" notin invConfig:
    return
  let limits = invConfig["limits"]
  for groupName, groupConfig in limits.pairs:
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
  ## Render "name: count" string for an inventory item.
  let name = getItemName(itemAmount)
  name & ": " & $itemAmount.count

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

proc getCollectiveName(collectiveId: int): string =
  ## Get the collective name by ID from the mg_config.
  ## Iterates to the nth key in the collectives dict.
  if replay.isNil or replay.mgConfig.isNil:
    return ""
  if "game" notin replay.mgConfig or "collectives" notin replay.mgConfig["game"]:
    return ""
  let collectives = replay.mgConfig["game"]["collectives"]
  if collectiveId < 0 or collectives.kind != JObject:
    return ""
  var idx = 0
  for name, _ in collectives.pairs:
    if idx == collectiveId:
      return name
    inc idx
  return ""

proc getAoeConfigs(typeName: string): JsonNode =
  ## Get the AOE configs array for an object type. Returns nil if no AOEs.
  if replay.isNil or replay.mgConfig.isNil:
    return nil
  if "game" notin replay.mgConfig:
    return nil
  let game = replay.mgConfig["game"]
  if "objects" notin game:
    return nil
  let objects = game["objects"]
  if typeName notin objects:
    return nil
  let objConfig = objects[typeName]
  if "aoes" notin objConfig or objConfig["aoes"].kind != JArray or objConfig["aoes"].len == 0:
    return nil
  return objConfig["aoes"]

proc checkAlignmentFilter(agentCollectiveId: int, sourceCollectiveId: int, alignment: string): bool =
  ## Check if an alignment filter passes for the given agent and source collective.
  case alignment:
  of "aligned":
    return agentCollectiveId >= 0
  of "unaligned":
    return agentCollectiveId < 0
  of "same_collective":
    return agentCollectiveId >= 0 and sourceCollectiveId >= 0 and agentCollectiveId == sourceCollectiveId
  of "different_collective":
    # Both must have collective, and they must be different
    return agentCollectiveId >= 0 and sourceCollectiveId >= 0 and agentCollectiveId != sourceCollectiveId
  of "not_same_collective":
    # Not same collective means: unaligned OR different collective
    return agentCollectiveId < 0 or agentCollectiveId != sourceCollectiveId
  else:
    return true

proc isAgentAffectedByAoeFilters(agentCollectiveId: int, sourceCollectiveId: int, filters: JsonNode): bool =
  ## Check if an agent is affected by an AOE based on filter array.
  if filters.isNil or filters.kind != JArray or filters.len == 0:
    return true
  for filter in filters:
    if filter.kind != JObject:
      continue
    if "filter_type" notin filter:
      continue
    let filterType = filter["filter_type"].getStr
    case filterType:
    of "alignment":
      let alignment = if "alignment" in filter: filter["alignment"].getStr else: ""
      let target = if "target" in filter: filter["target"].getStr else: "target"
      # For AOE, actor is the source, target is the affected entity
      let checkCollectiveId = if target == "actor": sourceCollectiveId else: agentCollectiveId
      if not checkAlignmentFilter(checkCollectiveId, sourceCollectiveId, alignment):
        return false
    of "vibe", "resource":
      # These filters can't be fully checked without more context, assume they pass
      discard
    else:
      discard
  return true

type
  AoeSourceEffect* = object
    source*: Entity
    aoeConfig*: JsonNode
    aoeRange*: int
    filters*: JsonNode

proc getAoeSourcesAffectingAgent(agent: Entity): seq[AoeSourceEffect] =
  ## Find all AOE sources that affect this agent.
  result = @[]
  if replay.isNil:
    return
  let agentPos = agent.location.at(step).xy
  let numObjects = replay.objects.len
  for i in 0 ..< numObjects:
    let obj = replay.objects[i]
    if obj.isAgent:
      continue
    let aoeConfigs = getAoeConfigs(obj.typeName)
    if aoeConfigs.isNil:
      continue
    # Check distance once for this object (Chebyshev/square)
    let sourcePos = obj.location.at(step).xy
    let dx = abs(agentPos.x - sourcePos.x)
    let dy = abs(agentPos.y - sourcePos.y)
    let distance = max(dx, dy)
    # Check each AOE config from this source
    for aoeConfig in aoeConfigs:
      # Get AOE range
      var aoeRange = 0
      if "range" in aoeConfig:
        if aoeConfig["range"].kind == JInt:
          aoeRange = aoeConfig["range"].getInt
        elif aoeConfig["range"].kind == JFloat:
          aoeRange = aoeConfig["range"].getFloat.int
      if aoeRange <= 0:
        continue
      if distance > aoeRange:
        continue
      # Get filter settings
      var filters: JsonNode = nil
      if "filters" in aoeConfig and aoeConfig["filters"].kind == JArray:
        filters = aoeConfig["filters"]
      # Check if agent is affected by this source
      if not isAgentAffectedByAoeFilters(agent.collectiveId, obj.collectiveId, filters):
        continue
      result.add(AoeSourceEffect(
        source: obj,
        aoeConfig: aoeConfig,
        aoeRange: aoeRange,
        filters: filters
      ))

proc computeResourceDeltas(agent: Entity): (Table[string, int], Table[string, seq[DeltaSource]]) =
  ## Compute total resource deltas and per-resource sources for an agent from AOE effects.
  var totalDeltas = initTable[string, int]()
  var deltaSources = initTable[string, seq[DeltaSource]]()
  if not agent.isAgent:
    return (totalDeltas, deltaSources)
  let aoeSources = getAoeSourcesAffectingAgent(agent)
  for effect in aoeSources:
    let aoeConfig = effect.aoeConfig
    if "resource_deltas" notin aoeConfig or aoeConfig["resource_deltas"].kind != JObject:
      continue
    for key, value in aoeConfig["resource_deltas"].pairs:
      var delta = 0
      if value.kind == JInt:
        delta = value.getInt
      elif value.kind == JFloat:
        delta = value.getFloat.int
      if delta == 0:
        continue
      if key notin totalDeltas:
        totalDeltas[key] = 0
        deltaSources[key] = @[]
      totalDeltas[key] = totalDeltas[key] + delta
      deltaSources[key].add(DeltaSource(
        sourceName: effect.source.typeName,
        sourcePos: effect.source.location.at(step).xy,
        delta: delta
      ))
  return (totalDeltas, deltaSources)

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

    # Compute resource deltas for agents.
    var resourceDeltas: Table[string, int]
    var deltaSources: Table[string, seq[DeltaSource]]
    if cur.isAgent:
      (resourceDeltas, deltaSources) = computeResourceDeltas(cur)

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

    # Display alignment (collective) for alignable objects.
    if cur.collectiveId >= 0:
      let collectiveName = getCollectiveName(cur.collectiveId)
      if collectiveName.len > 0:
        text(&"  Collective: {collectiveName} ({cur.collectiveId})")
      else:
        text(&"  Collective: ({cur.collectiveId})")

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
    if cur.isAgent:
      let resourceLimitGroups = parseInventoryLimits(replay.mgConfig)

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

        let effectiveLimit = computeEffectiveLimit(group, currentInventory, replay.itemNames)
        text(&"{group.name}: {usedAmount} / {effectiveLimit}")
        for itemAmount in groupItems:
          let itemName = getItemName(itemAmount)
          let delta = resourceDeltas.getOrDefault(itemName, 0)
          textWithDelta(&"  {itemName}: {itemAmount.count}", delta)

      var ungroupedItems: seq[ItemAmount] = @[]
      for itemAmount in currentInventory:
        if itemAmount.itemId >= 0 and itemAmount.itemId < replay.itemNames.len:
          let itemName = replay.itemNames[itemAmount.itemId]
          if itemName notin shownItems:
            ungroupedItems.add(itemAmount)

      if ungroupedItems.len > 0:
        text("Other:")
        for itemAmount in ungroupedItems:
          let itemName = getItemName(itemAmount)
          let delta = resourceDeltas.getOrDefault(itemName, 0)
          textWithDelta("  " & formatItem(itemAmount), delta)
    else:
      text("Inventory")
      if currentInventory.len == 0:
        text("  Empty")
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

    # AOE Effects - show on source objects
    let aoeConfigs = getAoeConfigs(cur.typeName)
    if aoeConfigs != nil:
      sk.advance(vec2(0, theme.spacing.float32))
      text("AOE Source")
      var aoeIdx = 0
      for aoeConfig in aoeConfigs:
        if aoeIdx > 0:
          sk.advance(vec2(0, theme.spacing.float32 / 2))
        aoeIdx += 1
        # Get AOE range
        var aoeRange = 0
        if "range" in aoeConfig:
          if aoeConfig["range"].kind == JInt:
            aoeRange = aoeConfig["range"].getInt
          elif aoeConfig["range"].kind == JFloat:
            aoeRange = aoeConfig["range"].getFloat.int
        text(&"  Range: {aoeRange}")
        # Get resource deltas
        if "resource_deltas" in aoeConfig and aoeConfig["resource_deltas"].kind == JObject:
          text("  Per-tick effects:")
          for key, value in aoeConfig["resource_deltas"].pairs:
            var delta = 0
            if value.kind == JInt:
              delta = value.getInt
            elif value.kind == JFloat:
              delta = value.getFloat.int
            let sign = if delta >= 0: "+" else: ""
            text(&"    {key}: {sign}{delta}")
        # Get and display filter settings
        if "filters" in aoeConfig and aoeConfig["filters"].kind == JArray:
          for filter in aoeConfig["filters"]:
            if filter.kind != JObject or "filter_type" notin filter:
              continue
            let filterType = filter["filter_type"].getStr
            case filterType:
            of "alignment":
              let alignment = if "alignment" in filter: filter["alignment"].getStr else: ""
              let target = if "target" in filter: filter["target"].getStr else: "target"
              text(&"  Filter: {alignment} ({target})")
            of "vibe":
              let vibe = if "vibe" in filter: filter["vibe"].getStr else: ""
              let target = if "target" in filter: filter["target"].getStr else: "target"
              text(&"  Filter: vibe={vibe} ({target})")
            of "resource":
              let target = if "target" in filter: filter["target"].getStr else: "target"
              text(&"  Filter: resource ({target})")
            else:
              text(&"  Filter: {filterType}")

    # Deltas section - show expected next-tick resource changes for agents
    if cur.isAgent and resourceDeltas.len > 0:
      sk.advance(vec2(0, theme.spacing.float32))
      text("Deltas:")
      for resourceName, delta in resourceDeltas.pairs:
        let deltaStr = if delta > 0: "+" & $delta else: $delta
        let color = if delta > 0: DeltaPosColor else: DeltaNegColor
        # Build source list for this resource
        var sourceList = ""
        if resourceName in deltaSources:
          var sourceNames: seq[string] = @[]
          for src in deltaSources[resourceName]:
            sourceNames.add(src.sourceName)
          sourceList = " (" & sourceNames.join(", ") & ")"
        # Draw the delta line with colored value
        let baseText = &"  {resourceName}: "
        let startPos = sk.at
        text(baseText)
        let baseWidth = sk.getTextSize("Default", baseText).x
        discard sk.drawText("Default", deltaStr, startPos + vec2(baseWidth, 0), color)
        # Draw source list in normal color if present
        if sourceList.len > 0:
          let deltaWidth = sk.getTextSize("Default", deltaStr).x
          discard sk.drawText("Default", sourceList, startPos + vec2(baseWidth + deltaWidth, 0), rgbx(180, 180, 180, 255))


proc selectObject*(obj: Entity) =
  selection = obj
