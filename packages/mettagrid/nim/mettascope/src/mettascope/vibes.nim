## Vibe panel allows you to set vibe frequency for the agent.
## Vibe are emoji like symbols that the agent can use to communicate with
## the world and other agents.

import
  std/os,
  fidget2,
  common, panels, replays, actions

find "/UI/Main/**/VibePanel":
  find "**/Button":
    onClick:
      let row = thisNode.parent.childIndex
      let column = thisNode.childIndex
      let vibeId = row * 10 + column
      if selection.isNil:
        # TODO: Maybe gray out the vibe buttons?
        echo "no selection, can't send vibe action"
        return
      let vibeActionId = replay.actionNames.find("change_glyph_" & $vibeId)
      sendAction(selection.agentId, vibeActionId, -1)

proc updateVibePanel*() =
  ## Updates the vibe panel to display the current vibe frequency for the agent.
  if replay.isNil:
    return
  let panel = panels.vibeTemplate.copy()
  panel.position = vec2(0, 0)
  let rowTemplate = panel.find("Row")
  let buttonTemplate = rowTemplate.find("Button").copy()
  rowTemplate.removeChildren()
  panel.removeChildren()
  var row: Node
  for id, vibe in replay.config.game.vibeNames:
    if id mod 10 == 0:
      row = rowTemplate.copy()
      panel.addChild(row)
    let button = buttonTemplate.copy()
    var path = "../../vibe" / vibe
    button.find("**/Icon").fills[0].imageRef = path
    row.addChild(button)
  vibePanel.node.removeChildren()
  vibePanel.node.addChild(panel)
