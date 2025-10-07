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
        let typeName = replay.typeNames[selection.typeId]
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
    if not existsDir("tmp"):
      createDir("tmp")
    let typeName = replay.typeNames[selection.typeId]
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

  addParam("Type", replay.typeNames[selection.typeId])

  if selection.isAgent:
    addParam("Agent ID", $selection.agentId)
    addParam("Reward", $selection.totalReward.at)

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

  proc addRecipe(recipeInfo: RecipeInfo) =
    var recipeNode = recipe.copy()
    for resource in recipeInfo.inputs:
      recipeNode.find("**/Inputs").addResource(resource)
    for resource in recipeInfo.outputs:
      recipeNode.find("**/Outputs").addResource(resource)
    recipeArea.addChild(recipeNode)

  if selection.inventory.at.len == 0:
    inventoryArea.remove()
  else:
    for itemAmount in selection.inventory.at:
      inventory.addResource(itemAmount)

  if selection.inputResources.len > 0 or selection.outputResources.len > 0:
    var recipeInfo = RecipeInfo()
    recipeInfo.inputs = selection.inputResources
    recipeInfo.outputs = selection.outputResources
    addRecipe(recipeInfo)

  var mergedRecipes: Table[string, RecipeInfo]
  for recipe in selection.recipes:
    let key = $recipe.inputs & "->" & $recipe.outputs
    mergedRecipes[key] = recipe

  for recipe in mergedRecipes.values:
    addRecipe(recipe)

  if mergedRecipes.len == 0 and
    selection.inputResources.len == 0 and
    selection.outputResources.len == 0:
      recipeArea.remove()

  x.position = vec2(0, 0)
  objectInfoPanel.node.removeChildren()
  objectInfoPanel.node.addChild(x)

proc selectObject*(obj: Entity) =
  selection = obj
  updateObjectInfo()
