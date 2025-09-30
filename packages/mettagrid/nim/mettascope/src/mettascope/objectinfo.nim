import
  fidget2,
  common, panels, replays

proc updateObjectInfo() =
  echo "updateObjectInfo"
  echo "selection: ", selection != nil
  echo "objectInfoPanel.node: ", objectInfoPanel.node
  echo "objectInfoTemplate: ", panels.objectInfoTemplate
  let x = panels.objectInfoTemplate.copy()
  x.position = vec2(0, 0)
  objectInfoPanel.node.addChild(x)

proc selectObject*(obj: Entity) =
  selection = obj
  updateObjectInfo()
