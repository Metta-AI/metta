import
  std/[os, json, algorithm, tables, sets],
  fidget2,
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

find "/UI/Main/**/ObjectInfo/OpenConfig":
  onClick:
    if selection.isNil:
      return
    let text =
      if replay.isNil or replay.mgConfig.isNil:
        "No replay config found."
      else:
        let typeName = selection.typeName
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
    let typeName = selection.typeName
    openTempTextFile(typeName & "_config.json", text)

proc updateObjectInfo*() =
  ## Updates the object info panel to display the current selection.
  if selection.isNil:
    return

  let x = panels.objectInfoTemplate.copy()

  let
    params = x.find("Params")
    param = x.find("Params/Param").copy()
    vibeArea = x.find("VibeArea")
    inventoryArea = x.find("InventoryArea")
    inventoryLabel = x.find("InventoryArea/label")
    inventory = x.find("InventoryArea/Inventory")
    inventoryRowTemplate = x.find("InventoryArea/Inventory").copy()
    item = x.find("InventoryArea/Inventory/Item").copy()
    recipeArea = x.find("RecipeArea")
    recipe = x.find("RecipeArea/Recipe").copy()
    recipeVibes = recipe.find("Vibes")
    recipeInput = recipe.find("Inputs")
    recipeOutput = recipe.find("Outputs")

  params.removeChildren()
  vibeArea.hide()
  inventory.removeChildren()
  recipeArea.removeChildren()
  recipeVibes.removeChildren()
  recipeInput.removeChildren()
  recipeOutput.removeChildren()

  proc addParam(name: string, value: string) =
    let p = param.copy()
    p.find("**/Name").text = name
    p.find("**/Value").text = value
    params.addChild(p)

  addParam("Type", selection.typeName)

  if selection.isAgent:
    addParam("Agent ID", $selection.agentId)
    addParam("Reward", $selection.totalReward.at)

    if replay.config.game.vibeNames.len > 0:
      let vibeId = selection.vibeId.at
      let vibeName = getVibeName(vibeId)

      vibeArea.find("**/Icon").fills[0].imageRef = "../../vibe" / vibeName
      vibeArea.show()

  if selection.cooldownRemaining.at > 0:
    addParam("Cooldown Remaining", $selection.cooldownRemaining.at)
  if selection.cooldownDuration > 0:
    addParam("Cooldown Duration", $selection.cooldownDuration)
  if selection.isClipped.at:
    addParam("Is Clipped", $selection.isClipped.at)
  if selection.isClipImmune.at:
    addParam("Is Clip Immune", $selection.isClipImmune.at)
  if selection.usesCount.at > 0:
    addParam("Uses Count", $selection.usesCount.at)
  if selection.maxUses > 0:
    addParam("Max Uses", $selection.maxUses)
  if selection.allowPartialUsage:
    addParam("Allow Partial Usage", $selection.allowPartialUsage)

  proc scaleNode(node: Node, scale: float32) =
    ## Recursively scale a node and all its children.
    node.size = node.size * scale
    for child in node.children:
      scaleNode(child, scale)

  proc addResource(area: Node, itemAmount: ItemAmount) =
    let i = item.copy()
    i.find("**/Image").fills[0].imageRef = "../../" & replay.itemImages[
        itemAmount.itemId]
    i.find("**/Amount").text = $itemAmount.count
    area.addChild(i)

  proc addSmallResource(area: Node, itemAmount: ItemAmount) =
    let i = item.copy()
    i.find("**/Image").fills[0].imageRef = "../../" & replay.itemImages[
        itemAmount.itemId]
    i.find("**/Amount").text = $itemAmount.count
    scaleNode(i, InventoryScale)
    area.addChild(i)

  proc addVibe(area: Node, vibe: string, count: int = 1) =
    let v = item.copy()
    v.find("**/Image").fills[0].imageRef = "../../vibe" / vibe
    v.find("**/Amount").text =
      if count > 1:
        $count
      else:
        ""
    area.addChild(v)

  proc addInventoryRow(name: string, used: int, limit: int, items: seq[ItemAmount]) =
    ## Creates a row with used/limit label first, then resource items.
    let row = inventoryRowTemplate.copy()
    row.removeChildren()
    scaleNode(row, InventoryScale)

    let label = item.copy()
    label.find("**/Image").visible = false
    label.find("**/Amount").text = $used & "/" & $limit
    row.addChild(label)

    for itemAmount in items:
      row.addSmallResource(itemAmount)

    inventoryArea.addChild(row)

  proc addUngroupedRow(items: seq[ItemAmount]) =
    ## Creates a row for ungrouped items with no used/limit label.
    let row = inventoryRowTemplate.copy()
    row.removeChildren()
    scaleNode(row, InventoryScale)

    for itemAmount in items:
      row.addSmallResource(itemAmount)

    inventoryArea.addChild(row)

  # Render inventory grouped by resource limits
  let currentInventory = selection.inventory.at
  if currentInventory.len == 0:
    inventoryArea.remove()
  elif selection.isAgent:
    # Hide the default "Inventory" label - we'll use row labels instead
    inventoryLabel.text = ""

    # Remove default inventory container - we'll add rows
    inventory.remove()

    # Get resource limit groups from config
    let resourceLimitGroups = parseResourceLimits(replay.mgConfig)

    # Build a lookup from item name to item amount
    var itemByName = initTable[string, ItemAmount]()
    for itemAmount in currentInventory:
      if itemAmount.itemId >= 0 and itemAmount.itemId < replay.itemNames.len:
        itemByName[replay.itemNames[itemAmount.itemId]] = itemAmount

    # Track shown items
    var shownItems = initOrderedSet[string]()

    # Render each group as a row
    for group in resourceLimitGroups:
      var usedAmount = 0
      var groupItems: seq[ItemAmount] = @[]

      # Collect items for this group
      for resourceName in group.resources:
        if resourceName in itemByName:
          let itemAmount = itemByName[resourceName]
          usedAmount += itemAmount.count
          groupItems.add(itemAmount)
          shownItems.incl(resourceName)

      # Only show group if it has items
      if groupItems.len > 0:
        let effectiveLimit = computeEffectiveLimit(group, currentInventory, replay.itemNames)
        addInventoryRow(group.name, usedAmount, effectiveLimit, groupItems)

    # Add row for ungrouped items
    var ungroupedItems: seq[ItemAmount] = @[]
    for itemAmount in currentInventory:
      if itemAmount.itemId >= 0 and itemAmount.itemId < replay.itemNames.len:
        let itemName = replay.itemNames[itemAmount.itemId]
        if itemName notin shownItems:
          ungroupedItems.add(itemAmount)

    if ungroupedItems.len > 0:
      addUngroupedRow(ungroupedItems)
  else:
    # Non-agent objects: render all items in one row
    for itemAmount in currentInventory:
      inventory.addResource(itemAmount)

  proc getHeartCount(outputs: seq[ItemAmount], itemNames: seq[string]): int =
    ## Returns total hearts produced by this protocol.
    for output in outputs:
      if output.itemId == 5:  # "heart" item
        return output.count
    return 0

  proc addProtocol(protocol: Protocol) =
    var protocolNode = recipe.copy()
    # Count the vibes.
    var vibeCounts: Table[int, int]
    for vibe in protocol.vibes:
      vibeCounts[vibe] = vibeCounts.getOrDefault(vibe, 0) + 1
    for vibe, count in vibeCounts:
      protocolNode.find("**/Vibes").addVibe(vibe.getVibeName(), count)
    for resource in protocol.inputs:
      protocolNode.find("**/Inputs").addResource(resource)
    for resource in protocol.outputs:
      protocolNode.find("**/Outputs").addResource(resource)
    recipeArea.addChild(protocolNode)

  # Sort protocols: heart-producing ones first (most hearts first), then others.
  var sortedProtocols = selection.protocols
  sortedProtocols.sort(proc(a, b: Protocol): int =
    let aHearts = getHeartCount(a.outputs, replay.itemNames)
    let bHearts = getHeartCount(b.outputs, replay.itemNames)

    # non-heart recipes can go in any order after heart recipes.
    if aHearts > 0 and bHearts == 0:
      -1
    elif aHearts == 0 and bHearts > 0:
      1
    else:
      cmp(bHearts, aHearts)
  )

  for protocol in sortedProtocols:
    addProtocol(protocol)

  x.position = vec2(0, 0)
  objectInfoPanel.node.removeChildren()
  objectInfoPanel.node.addChild(x)

proc selectObject*(obj: Entity) =
  selection = obj
  updateObjectInfo()
