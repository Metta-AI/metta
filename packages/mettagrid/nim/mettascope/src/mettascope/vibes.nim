## Vibe panel allows you to set vibe frequency for the agent.
## Vibe are emoji like symbols that the agent can use to communicate with
## the world and other agents.

import
  std/[os, tables],
  fidget2, windy,
  common, panels, replays, actions, pathfinding

find "/UI/Main/**/VibePanel":
  find "**/Button":
    onClick:
      let row = thisNode.parent.childIndex
      let column = thisNode.childIndex
      let vibeId = row * 10 + column
      if selection.isNil or not selection.isAgent:
        echo "no agent selected, can't send vibe action"
        return

      # Check if vibeId is valid (within bounds of vibeNames)
      if vibeId < 0 or vibeId >= replay.config.game.vibeNames.len:
        echo "invalid vibe id: ", vibeId, " (max: ", replay.config.game.vibeNames.len - 1, ")"
        return

      # Get the vibe name and find the corresponding action
      let vibeName = replay.config.game.vibeNames[vibeId]
      let vibeActionId = replay.actionNames.find("change_vibe_" & vibeName)
      if vibeActionId == -1:
        echo "vibe action not found: change_vibe_", vibeName
        return

      let shiftDown = window.buttonDown[KeyLeftShift] or window.buttonDown[KeyRightShift]

      if shiftDown:
        # Queue the vibe action as an objective.
        let objective = Objective(kind: Vibe, vibeActionId: vibeActionId, repeat: false)
        if not agentObjectives.hasKey(selection.agentId) or agentObjectives[
            selection.agentId].len == 0:
          agentObjectives[selection.agentId] = @[objective]
          # Append vibe action directly to path queue.
          agentPaths[selection.agentId] = @[
            PathAction(kind: Vibe, vibeActionId: vibeActionId)
          ]
        else:
          agentObjectives[selection.agentId].add(objective)
          # Push the vibe action to the end of the current path.
          if agentPaths.hasKey(selection.agentId):
            agentPaths[selection.agentId].add(
              PathAction(kind: Vibe, vibeActionId: vibeActionId)
            )
          else:
            agentPaths[selection.agentId] = @[
              PathAction(kind: Vibe, vibeActionId: vibeActionId)
            ]
      else:
        # Execute immediately.
        sendAction(selection.agentId, replay.actionNames[vibeActionId])

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
