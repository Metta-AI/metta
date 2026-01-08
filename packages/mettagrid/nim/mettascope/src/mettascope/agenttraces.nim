import
  std/[times],
  boxy, chroma, windy,
  common, panels, replays, worldmap

let
  traceWidth = 0.54 / 2
  traceHeight = 2.0


proc drawAgentTraces*(panel: Panel) =

  panel.scrollArea = rect(
    0,
    0,
    replay.maxSteps.float32 * traceWidth,
    replay.agents.len.float32 * traceHeight
  )

  panel.beginPanAndZoom()

  # Follow selected agent when lockFocus is enabled
  if settings.lockFocus and selection != nil and selection.isAgent:
    let rectW = panel.rect.w.float32
    let rectH = panel.rect.h.float32
    if rectW > 0 and rectH > 0:
      let z = panel.zoom * panel.zoom
      let centerX = step.float32 * traceWidth + traceWidth / 2
      let centerY = selection.agentId.float32 * traceHeight + traceHeight / 2
      panel.pos.x = rectW / 2.0f - centerX * z
      panel.pos.y = rectH / 2.0f - centerY * z

  if panel.hasMouse and window.buttonDown[MouseLeft]:
    let mousePos = bxy.getTransform().inverse * window.mousePos.vec2

    if window.buttonPressed[MouseLeft]:
      echo "Single press at position: ", mousePos
      settings.lockFocus = false
      let
        newStep = floor(mousePos.x() / traceWidth).int
        agentId = floor(mousePos.y() / traceHeight).int
      if newStep >= 0 and newStep < replay.maxSteps and agentId >= 0 and
          agentId < replay.agents.len:
        step = newStep
        stepFloat = newStep.float32
        selection = replay.agents[agentId]
        centerAt(worldMapPanel, selection)

  # Handle double-click to toggle focus lock
  if window.buttonPressed[DoubleClick]:
    let mousePos = bxy.getTransform().inverse * window.mousePos.vec2
    echo "Double-click detected at position: ", mousePos
    settings.lockFocus = true

  # Selected agent
  if selection != nil and selection.isAgent:
    bxy.drawRect(rect(0, selection.agentId.float32 * traceHeight,
      panel.rect.w.float32, traceHeight.float32), color(0.3, 0.3, 0.3, 1.0))

  # Current step
  bxy.drawRect(
    rect(
      step.float32 * traceWidth,
      0.0,
      traceWidth,
      panel.rect.h.float32
    ),
    color(0.5, 0.5, 0.5, 0.5)
  )

  # Agents
  for obj in replay.agents:
    let j = obj.agentId
    for i in 0 ..< replay.maxSteps:
      let pos = vec2(
        i.float32 * traceWidth + traceWidth / 2,
        j.float32 * traceHeight + traceHeight / 2
      )
      if obj.isFrozen.len > 1 and obj.isFrozen[i]:
        bxy.drawImage("trace/frozen", pos, angle = 0, scale = 1/200)
      else:
        let actionId = obj.actionId.at(i)
        if actionId >= 0:
          if obj.actionSuccess.at(i):
            bxy.drawImage(
              replay.traceImages[actionId],
              pos,
              angle = 0,
              scale = 1/200.0
            )
          else:
            bxy.drawImage("trace/invalid", pos, angle = 0, scale = 1/200)

      let reward = obj.currentReward.at(i)
      if reward > 0:
        bxy.drawImage(
          "resources/reward",
          vec2(pos.x, pos.y + (traceHeight / 2) - 32/256),
          angle = 0,
          scale = 1/800
        )

      if settings.showResources and i > 0:
        let gainMap = obj.gainMap[i]
        for item in gainMap:
          for j in 0 ..< item.count:
            bxy.drawImage(
              replay.itemImages[item.itemId],
              vec2(pos.x, pos.y - (traceHeight / 2) + ((j + 1) * 32/256)),
              angle = 0,
              scale = 1/800
            )

  panel.endPanAndZoom()
