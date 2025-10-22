import
  std/[os, json, tables],
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
    if not dirExists("tmp"):
      createDir("tmp")
    let typeName = selection.typeName
    writeFile("tmp/" & typeName & "_config.json", text)
    when defined(windows):
      discard execShellCmd("notepad tmp/" & typeName & "_config.json")
    elif defined(macosx):
      discard execShellCmd("open -a TextEdit tmp/" & typeName & "_config.json")
    else:
      discard execShellCmd("xdg-open tmp/" & typeName & "_config.json")

proc updateObjectInfo*() =
  ## Updates the object info panel to display the current selection.
  if selection.isNil:
    return

  let x = panels.objectInfoTemplate.copy()

  let
    params = x.find("Params")
    param = x.find("Params/Param").copy()
    inventoryArea = x.find("InventoryArea")
    inventory = x.find("InventoryArea/Inventory")
    item = x.find("InventoryArea/Inventory/Item").copy()
    recipeArea = x.find("RecipeArea")
    recipe = x.find("RecipeArea/Recipe").copy()
    recipeInput = recipe.find("Inputs")
    recipeOutput = recipe.find("Outputs")

  params.removeChildren()
  inventory.removeChildren()
  recipeArea.removeChildren()
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
      if vibeId < replay.config.game.vibeNames.len:
        let vibeName = replay.config.game.vibeNames[vibeId]
        addParam("Vibe", vibeName)

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
    i.find("**/Image").fills[0].imageRef = "../../" & replay.itemImages[itemAmount.itemId]
    i.find("**/Amount").text = $itemAmount.count
    area.addChild(i)

  proc resourceTableToSeq(table: Table[string, int]): seq[ItemAmount] =
    ## Converts a resource table to a sequence of item amounts.
    for name, count in table:
      let itemId = replay.itemNames.find(name)
      result.add(ItemAmount(itemId: itemId, count: count))

  proc addVibe(area: Node, vibe: string) =
    let v = item.copy()
    v.find("**/Image").fills[0].imageRef = "../../vibe" / vibe
    v.find("**/Amount").text = ""
    area.addChild(v)

  if selection.inventory.at.len == 0:
    inventoryArea.remove()
  else:
    for itemAmount in selection.inventory.at:
      inventory.addResource(itemAmount)

  proc addRecipe(
    vibes: seq[string],
    inputs: seq[ItemAmount],
    outputs: seq[ItemAmount],
  ) =
    var recipeNode = recipe.copy()
    for vibe in vibes:
      recipeNode.find("**/Vibes").addVibe(vibe)
    for resource in inputs:
      recipeNode.find("**/Inputs").addResource(resource)
    for resource in outputs:
      recipeNode.find("**/Outputs").addResource(resource)
    recipeArea.addChild(recipeNode)

  for name, obj in replay.config.game.objects:
    if name == selection.typeName:
      for recipe in obj.recipes:
        addRecipe(
          recipe.pattern,
          recipe.protocol.inputResources.resourceTableToSeq,
          recipe.protocol.outputResources.resourceTableToSeq
        )

  x.position = vec2(0, 0)
  objectInfoPanel.node.removeChildren()
  objectInfoPanel.node.addChild(x)

proc selectObject*(obj: Entity) =
  selection = obj
  updateObjectInfo()
