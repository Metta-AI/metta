## Vibe panel allows you to set vibe frequency for the agent.
## Vibe are emoji like symbols that the agent can use to communicate with
## the world and other agents.

import
  std/[os, tables],
  silky, windy,
  common, panels, replays, actions, pathfinding


var vibes = @[
  "vibe/alembic",
  "vibe/angry",
  "vibe/anxious",
  "vibe/assembler",
  "vibe/asterisk",
  "vibe/backpack",
  "vibe/beaming",
  "vibe/black-circle",
  "vibe/black-heart",
  "vibe/blue-circle",
  "vibe/blue-diamond",
  "vibe/blue-heart",
  "vibe/bow",
  "vibe/broken-heart",
  "vibe/brown-circle",
  "vibe/brown-heart",
  "vibe/brown-square",
  "vibe/carbon",
  "vibe/carbon_a",
  "vibe/carbon_b",
  "vibe/carrot",
  "vibe/charger",
  "vibe/chart-down",
  "vibe/chart-up",
  "vibe/chest",
  "vibe/clown",
  "vibe/coin",
  "vibe/compass",
  "vibe/confused",
  "vibe/corn",
  "vibe/crying-cat",
  "vibe/crying",
  "vibe/dagger",
  "vibe/default",
  "vibe/diamond",
  "vibe/divide",
  "vibe/down-left",
  "vibe/down-right",
  "vibe/down",
  "vibe/drooling",
  "vibe/eight",
  "vibe/factory",
  "vibe/fearful",
  "vibe/fire",
  "vibe/five",
  "vibe/four",
  "vibe/fuel",
  "vibe/gear",
  "vibe/germanium",
  "vibe/germanium_a",
  "vibe/germanium_b",
  "vibe/ghost",
  "vibe/green-circle",
  "vibe/green-heart",
  "vibe/grinning-big-eyes",
  "vibe/grinning-smiling-eyes",
  "vibe/grinning",
  "vibe/growing-heart",
  "vibe/halo",
  "vibe/hammer",
  "vibe/hash",
  "vibe/heart-arrow",
  "vibe/heart-decoration",
  "vibe/heart-exclamation",
  "vibe/heart-eyes",
  "vibe/heart-ribbon",
  "vibe/heart",
  "vibe/heart_a",
  "vibe/heart_b",
  "vibe/hundred",
  "vibe/kiss",
  "vibe/left",
  "vibe/light-shade",
  "vibe/lightning",
  "vibe/love-letter",
  "vibe/medium-shade",
  "vibe/minus",
  "vibe/moai",
  "vibe/money",
  "vibe/monocle",
  "vibe/mountain",
  "vibe/multiply",
  "vibe/nine",
  "vibe/numbers",
  "vibe/oil",
  "vibe/one",
  "vibe/orange-circle",
  "vibe/orange-heart",
  "vibe/orange-square",
  "vibe/oxygen",
  "vibe/oxygen_a",
  "vibe/oxygen_b",
  "vibe/package",
  "vibe/paperclip",
  "vibe/pin",
  "vibe/plug",
  "vibe/plus",
  "vibe/pouting",
  "vibe/purple-circle",
  "vibe/purple-heart",
  "vibe/purple-square",
  "vibe/pushpin",
  "vibe/red-circle",
  "vibe/red-heart",
  "vibe/red-triangle",
  "vibe/revolving-hearts",
  "vibe/right",
  "vibe/rock",
  "vibe/rocket",
  "vibe/rofl",
  "vibe/rolling-eyes",
  "vibe/rotate-clockwise",
  "vibe/rotate",
  "vibe/savoring",
  "vibe/seahorse",
  "vibe/seven",
  "vibe/shield",
  "vibe/silicon",
  "vibe/silicon_a",
  "vibe/silicon_b",
  "vibe/six",
  "vibe/skull-crossbones",
  "vibe/sleepy",
  "vibe/small-blue-diamond",
  "vibe/smiling",
  "vibe/smirking",
  "vibe/sobbing",
  "vibe/sparkle",
  "vibe/sparkling-heart",
  "vibe/squinting",
  "vibe/star-struck",
  "vibe/swearing",
  "vibe/swords",
  "vibe/target",
  "vibe/tears-of-joy",
  "vibe/ten",
  "vibe/test-tube",
  "vibe/three",
  "vibe/tree",
  "vibe/two-hearts",
  "vibe/two",
  "vibe/up-left",
  "vibe/up-right",
  "vibe/up",
  "vibe/wall",
  "vibe/water",
  "vibe/wave",
  "vibe/wheat",
  "vibe/white-circle",
  "vibe/white-heart",
  "vibe/white-square",
  "vibe/wood",
  "vibe/wrench",
  "vibe/yawning",
  "vibe/yellow-circle",
  "vibe/yellow-heart",
  "vibe/yellow-square",
  "vibe/zero",
]

proc drawVibes*(panel: Panel, frameId: string, contentPos: Vec2, contentSize: Vec2) =
  let m = 12.0f
  frame(frameId, contentPos, contentSize):
    sk.at = sk.pos + vec2(m, m) * 2
    for i, vibe in vibes:
      if i > 0 and i mod 10 == 0:
        sk.at.x = sk.pos.x + m * 2
        sk.at.y += 32 + m
      iconButton(vibe):
        echo vibe

# find "/UI/Main/**/VibePanel":
#   find "**/Button":
#     onClick:
#       let row = thisNode.parent.childIndex
#       let column = thisNode.childIndex
#       let vibeId = row * 10 + column
#       if selection.isNil or not selection.isAgent:
#         echo "no agent selected, can't send vibe action"
#         return

#       # Get the vibe name and find the corresponding action
#       let vibeName = getVibeName(vibeId)

#       let vibeActionId = replay.actionNames.find("change_vibe_" & vibeName)
#       if vibeActionId == -1:
#         echo "vibe action not found: change_vibe_", vibeName
#         return

#       let shiftDown = window.buttonDown[KeyLeftShift] or window.buttonDown[KeyRightShift]

#       if shiftDown:
#         # Queue the vibe action as an objective.
#         let objective = Objective(kind: Vibe, vibeActionId: vibeActionId, repeat: false)
#         if not agentObjectives.hasKey(selection.agentId) or agentObjectives[
#             selection.agentId].len == 0:
#           agentObjectives[selection.agentId] = @[objective]
#           # Append vibe action directly to path queue.
#           agentPaths[selection.agentId] = @[
#             PathAction(kind: Vibe, vibeActionId: vibeActionId)
#           ]
#         else:
#           agentObjectives[selection.agentId].add(objective)
#           # Push the vibe action to the end of the current path.
#           if agentPaths.hasKey(selection.agentId):
#             agentPaths[selection.agentId].add(
#               PathAction(kind: Vibe, vibeActionId: vibeActionId)
#             )
#           else:
#             agentPaths[selection.agentId] = @[
#               PathAction(kind: Vibe, vibeActionId: vibeActionId)
#             ]
#       else:
#         # Execute immediately.
#         sendAction(selection.agentId, replay.actionNames[vibeActionId])

# proc updateVibePanel*() =
#   ## Updates the vibe panel to display the current vibe frequency for the agent.
#   if replay.isNil:
#     return
#   let panel = panels.vibeTemplate.copy()
#   panel.position = vec2(0, 0)
#   let rowTemplate = panel.find("Row")
#   let buttonTemplate = rowTemplate.find("Button").copy()
#   rowTemplate.removeChildren()
#   panel.removeChildren()
#   let elementsPerRow = 10
#   var row: Node
#   for id, vibe in replay.config.game.vibeNames:
#     if id mod elementsPerRow == 0:
#       row = rowTemplate.copy()
#       panel.addChild(row)
#     let button = buttonTemplate.copy()
#     var path = "../../vibe" / vibe
#     button.find("**/Icon").fills[0].imageRef = path
#     if playMode == Historical:
#       # Gray out the vibes for historical mode.
#       button.find("**/Icon").fills[0].opacity = 0.4
#     row.addChild(button)
#   vibePanel.node.removeChildren()
#   vibePanel.node.addChild(panel)
