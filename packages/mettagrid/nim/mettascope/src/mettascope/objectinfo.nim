import
  std/[json, algorithm, tables, sets, strutils, strformat],
  vmath, silky, windy,
  common, panels, replays

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

proc formatItemWithLimit(itemAmount: ItemAmount, limit: int): string =
  ## Render "name: count / limit" string if limit is set, otherwise "name: count".
  let name = getItemName(itemAmount)
  if limit > 0:
    name & ": " & $itemAmount.count & " / " & $limit
  else:
    name & ": " & $itemAmount.count

proc getInventoryLimitFromConfig(configNode: JsonNode, resourceName: string): int =
  ## Get inventory limit for a resource from an inventory config node. Returns 0 if not set.
  if configNode.isNil or configNode.kind != JObject:
    return 0
  # Check specific limits first
  if "limits" in configNode and configNode["limits"].kind == JObject:
    let limits = configNode["limits"]
    # Check for direct resource limit
    if resourceName in limits:
      let limitConfig = limits[resourceName]
      if "limit" in limitConfig:
        if limitConfig["limit"].kind == JInt:
          return limitConfig["limit"].getInt
        elif limitConfig["limit"].kind == JFloat:
          return limitConfig["limit"].getFloat.int
    # Check for resource group limits
    for groupName, groupConfig in limits.pairs:
      if "resources" in groupConfig and groupConfig["resources"].kind == JArray:
        for r in groupConfig["resources"]:
          if r.getStr == resourceName:
            if "limit" in groupConfig:
              if groupConfig["limit"].kind == JInt:
                return groupConfig["limit"].getInt
              elif groupConfig["limit"].kind == JFloat:
                return groupConfig["limit"].getFloat.int
  # Check default_limit
  if "default_limit" in configNode:
    if configNode["default_limit"].kind == JInt:
      return configNode["default_limit"].getInt
    elif configNode["default_limit"].kind == JFloat:
      return configNode["default_limit"].getFloat.int
  return 0

proc getObjectInventoryLimit(typeName: string, resourceName: string): int =
  ## Get the inventory limit for a resource on an object type. Returns 0 if not set.
  if replay.isNil or replay.mgConfig.isNil:
    return 0
  if "game" notin replay.mgConfig:
    return 0
  let game = replay.mgConfig["game"]
  if "objects" notin game:
    return 0
  let objects = game["objects"]
  if typeName notin objects:
    return 0
  let objConfig = objects[typeName]
  if "inventory" notin objConfig:
    return 0
  return getInventoryLimitFromConfig(objConfig["inventory"], resourceName)

proc getAgentInventoryLimit(resourceName: string): int =
  ## Get the inventory limit for a resource on an agent. Returns 0 if not set.
  if replay.isNil or replay.mgConfig.isNil:
    return 0
  if "game" notin replay.mgConfig:
    return 0
  let game = replay.mgConfig["game"]
  if "agent" notin game:
    return 0
  let agentConfig = game["agent"]
  if "inventory" notin agentConfig:
    return 0
  return getInventoryLimitFromConfig(agentConfig["inventory"], resourceName)

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

proc getCommonsName(commonsId: int): string =
  ## Get the commons name by ID from the mg_config.
  if replay.isNil or replay.mgConfig.isNil:
    return ""
  if "game" notin replay.mgConfig or "commons" notin replay.mgConfig["game"]:
    return ""
  let commonsArr = replay.mgConfig["game"]["commons"]
  if commonsArr.kind != JArray or commonsId < 0 or commonsId >= commonsArr.len:
    return ""
  let commonsConfig = commonsArr[commonsId]
  if commonsConfig.kind == JObject and "name" in commonsConfig:
    return commonsConfig["name"].getStr
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

proc isAgentAffectedByAoe(agentCommonsId: int, sourceCommonsId: int, membersOnly: bool, ignoreMembers: bool): bool =
  ## Check if an agent is affected by an AOE based on commons filtering.
  if membersOnly:
    # Only affect agents with matching commons
    return agentCommonsId >= 0 and agentCommonsId == sourceCommonsId
  if ignoreMembers:
    # Ignore agents with matching commons
    return agentCommonsId < 0 or agentCommonsId != sourceCommonsId
  # No filter, affect all agents
  return true

type
  AoeSourceEffect* = object
    source*: Entity
    aoeConfig*: JsonNode
    aoeRange*: int
    membersOnly*: bool
    ignoreMembers*: bool

proc getAoeSourcesAffectingAgent(agent: Entity): seq[AoeSourceEffect] =
  ## Find all AOE sources that affect this agent.
  result = @[]
  if replay.isNil:
    return
  let agentPos = agent.location.at(step).xy
  for obj in replay.objects:
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
      var membersOnly = false
      var ignoreMembers = false
      if "members_only" in aoeConfig and aoeConfig["members_only"].kind == JBool:
        membersOnly = aoeConfig["members_only"].getBool
      if "ignore_members" in aoeConfig and aoeConfig["ignore_members"].kind == JBool:
        ignoreMembers = aoeConfig["ignore_members"].getBool
      # Check if agent is affected by this source
      if not isAgentAffectedByAoe(agent.commonsId, obj.commonsId, membersOnly, ignoreMembers):
        continue
      result.add(AoeSourceEffect(
        source: obj,
        aoeConfig: aoeConfig,
        aoeRange: aoeRange,
        membersOnly: membersOnly,
        ignoreMembers: ignoreMembers
      ))

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

    # Display alignment (commons) for alignable objects.
    if cur.commonsId >= 0:
      let commonsName = getCommonsName(cur.commonsId)
      if commonsName.len > 0:
        text(&"  Alignment: {commonsName} ({cur.commonsId})")
      else:
        text(&"  Alignment: ({cur.commonsId})")

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
            text(&"  {group.name}: {usedAmount} / {effectiveLimit}")
            for itemAmount in groupItems:
              if itemAmount.itemId != replay.itemNames.find("energy"):
                let itemName = getItemName(itemAmount)
                text(&"    {itemName}: {itemAmount.count}")

        var ungroupedItems: seq[ItemAmount] = @[]
        for itemAmount in currentInventory:
          if itemAmount.itemId >= 0 and itemAmount.itemId < replay.itemNames.len:
            let itemName = replay.itemNames[itemAmount.itemId]
            if itemName notin shownItems:
              ungroupedItems.add(itemAmount)

        if ungroupedItems.len > 0:
          text("  Other:")
          for itemAmount in ungroupedItems:
            let resourceName = getItemName(itemAmount)
            let limit = getAgentInventoryLimit(resourceName)
            text("    " & formatItemWithLimit(itemAmount, limit))
      else:
        for itemAmount in currentInventory:
          let resourceName = getItemName(itemAmount)
          let limit = getObjectInventoryLimit(cur.typeName, resourceName)
          text("  " & formatItemWithLimit(itemAmount, limit))

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
        # Get filter settings
        var membersOnly = false
        var ignoreMembers = false
        if "members_only" in aoeConfig and aoeConfig["members_only"].kind == JBool:
          membersOnly = aoeConfig["members_only"].getBool
        if "ignore_members" in aoeConfig and aoeConfig["ignore_members"].kind == JBool:
          ignoreMembers = aoeConfig["ignore_members"].getBool
        if membersOnly:
          text("  Filter: Members only")
        elif ignoreMembers:
          text("  Filter: Ignore members")

    # AOE Effects - show incoming effects on agents
    if cur.isAgent:
      let aoeSources = getAoeSourcesAffectingAgent(cur)
      if aoeSources.len > 0:
        sk.advance(vec2(0, theme.spacing.float32))
        text("Incoming AOE Effects")

        # Aggregate all resource deltas from all sources
        var totalDeltas = initTable[string, int]()
        for effect in aoeSources:
          let aoeConfig = effect.aoeConfig
          if "resource_deltas" in aoeConfig and aoeConfig["resource_deltas"].kind == JObject:
            for key, value in aoeConfig["resource_deltas"].pairs:
              var delta = 0
              if value.kind == JInt:
                delta = value.getInt
              elif value.kind == JFloat:
                delta = value.getFloat.int
              if key notin totalDeltas:
                totalDeltas[key] = 0
              totalDeltas[key] = totalDeltas[key] + delta

        # Show totals with current -> new values
        let inv = cur.inventory.at(step)
        for resourceName, delta in totalDeltas.pairs:
          let resourceIdx = replay.itemNames.find(resourceName)
          var currentAmount = 0
          if resourceIdx >= 0:
            for item in inv:
              if item.itemId == resourceIdx:
                currentAmount = item.count
                break
          let newAmount = currentAmount + delta
          let sign = if delta >= 0: "+" else: ""
          text(&"  {resourceName}: {currentAmount} → {newAmount} ({sign}{delta})")

        # List sources
        sk.advance(vec2(0, theme.spacing.float32 / 2))
        text("  Sources:")
        for effect in aoeSources:
          let sourcePos = effect.source.location.at(step).xy
          let commonsName = getCommonsName(effect.source.commonsId)
          let commonsInfo = if commonsName.len > 0: " [" & commonsName & "]" else: ""
          text(&"    {effect.source.typeName}{commonsInfo} @ ({sourcePos.x}, {sourcePos.y})")


proc selectObject*(obj: Entity) =
  selection = obj
