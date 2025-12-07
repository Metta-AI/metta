import
  std/[os, json, algorithm, tables],
  fidget2,
  common, panels, replays

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
    inventory = x.find("InventoryArea/Inventory")
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

  proc addResource(area: Node, itemAmount: ItemAmount) =
    let i = item.copy()
    i.find("**/Image").fills[0].imageRef = "../../" & replay.itemImages[
        itemAmount.itemId]
    i.find("**/Amount").text = $itemAmount.count
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

  if selection.inventory.at.len == 0:
    inventoryArea.remove()
  else:
    for itemAmount in selection.inventory.at:
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
