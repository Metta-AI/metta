## Vibe panel allows you to set vibe frequency for the agent.
## Vibe are emoji like symbols that the agent can use to communicate with
## the world and other agents.

import
  std/[os, tables, strutils],
  silky, windy,
  common, panels, replays, actions, pathfinding

proc getVibes(): seq[string] =
  for vibe in replay.config.game.vibeNames:
    result.add("vibe/" & vibe)

proc drawVibes*(panel: Panel, frameId: string, contentPos: Vec2, contentSize: Vec2) =
  let m = 12.0f
  frame(frameId, contentPos, contentSize):
    sk.at = sk.pos + vec2(m, m) * 2
    for i, vibe in getVibes():
      if i > 0 and i mod 10 == 0:
        sk.at.x = sk.pos.x + m * 2
        sk.at.y += 32 + m
      iconButton(vibe):
        let vibeName = vibe.split("/")[1]
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
