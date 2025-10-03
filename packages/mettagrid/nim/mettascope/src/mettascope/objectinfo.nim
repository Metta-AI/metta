import
  fidget2,
  common, panels, replays

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
    recipe = x.find("Recipe")
    recipeInput = x.find("Recipe/RecipeArea/RecipeInput")
    recipeOutput = x.find("Recipe/RecipeArea/RecipeOutput")
    actions = x.find("Actions")

  params.removeChildren()
  inventory.removeChildren()
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

  proc addResource(area: Node, itemAmount: ItemAmount) =
    let i = item.copy()
    i.find("**/Image").fills[0].imageRef = "../../" & replay.itemImages[itemAmount.itemId]
    i.find("**/Amount").text = $itemAmount.count
    area.addChild(i)

  if selection.inventory.at.len == 0:
    inventoryArea.remove()
  else:
    for itemAmount in selection.inventory.at:
      inventory.addResource(itemAmount)

  if selection.inputResources.len == 0 and selection.outputResources.len == 0:
    recipe.remove()
    discard
  else:
    for inputResource in selection.inputResources:
      recipeInput.addResource(inputResource)
    for outputResource in selection.outputResources:
      recipeOutput.addResource(outputResource)

  actions.remove()

  x.position = vec2(0, 0)
  objectInfoPanel.node.removeChildren()
  objectInfoPanel.node.addChild(x)

proc selectObject*(obj: Entity) =
  selection = obj
  updateObjectInfo()
