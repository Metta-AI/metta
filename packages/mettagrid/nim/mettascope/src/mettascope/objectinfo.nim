import
  fidget2,
  common, panels, replays

proc updateObjectInfo*() =
  echo "updateObjectInfo"
  echo "selection: ", selection != nil
  echo "objectInfoPanel.node: ", objectInfoPanel.node
  echo "objectInfoTemplate: ", panels.objectInfoTemplate
  let x = panels.objectInfoTemplate.copy()

  let
    params = x.find("**/Params")
    param = x.find("**/Param").copy()
    inventoryArea = x.find("**/InventoryArea")
    inventory = x.find("**/Inventory")
    item = x.find("**/Item").copy()
    recipe = x.find("**/Recipe")
    actions = x.find("**/Actions")

  params.removeChildren()
  inventory.removeChildren()

  proc addParam(name: string, value: string) =
    let p = param.copy()
    p.find("**/Name").text = name
    p.find("**/Value").text = value
    params.addChild(p)

  addParam("Type", replay.typeNames[selection.typeId])

  if selection.isAgent:
    addParam("Agent ID", $selection.agentId)
    addParam("Reward", $selection.totalReward.at)

  if selection.inventory.at.len == 0:
    inventoryArea.remove()
  else:
    proc addInventory(itemAmount: ItemAmount) =
      let i = item.copy()
      i.find("**/Image").fills[0].imageRef = "../../" & replay.itemImages[itemAmount.itemId]
      i.find("**/Amount").text = $itemAmount.count
      inventory.addChild(i)

    for itemAmount in selection.inventory.at:
      addInventory(itemAmount)

  recipe.remove()
  actions.remove()

  x.position = vec2(0, 0)
  objectInfoPanel.node.removeChildren()
  objectInfoPanel.node.addChild(x)

proc selectObject*(obj: Entity) =
  selection = obj
  updateObjectInfo()
