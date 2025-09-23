import
  std/[times],
  boxy, chroma, windy, fidget2/[common, hybridrender],
  common, panels, replays, worldmap

let traceWidth = 0.54 / 2
let traceHeight = 2.0

# Double-click detection variables
var lastClickTime: float64 = 0.0
var lastClickPos: Vec2 = vec2(0, 0)
const clickInterval = 0.3 # seconds
const clickDistance = 10.0 # pixels

proc drawAgentTraces*(panel: Panel) =
  let box = IRect(x: 0, y: 0, w: panel.rect.w, h: panel.rect.h)

  panel.scrollArea = rect(0, 0, replay.maxSteps.float32 * traceWidth,
      replay.agents.len.float32 * traceHeight)

  panel.beginPanAndZoom()

  if panel.hasMouse and window.buttonDown[MouseLeft]:
    let mousePos = bxy.getTransform().inverse * window.mousePos.vec2
    let isClick = dist(mousePos, lastClickPos) < clickDistance

    if window.buttonPressed[MouseLeft]:
      let currentTime = epochTime()
      if currentTime - lastClickTime < clickInterval and isClick:
        echo "Double-click detected at position: ", mousePos
        followSelection = true
      else:
        echo "Single press at position: ", mousePos
        followSelection = false
        let newStep = floor(mousePos.x() / traceWidth).int
        let agentId = floor(mousePos.y() / traceHeight).int
        if newStep >= 0 and newStep < replay.maxSteps and agentId >= 0 and
            agentId < replay.agents.len:
          step = newStep
          selection = replay.agents[agentId]
          centerAt(worldMapPanel, selection)

      lastClickTime = currentTime
      lastClickPos = mousePos
    elif window.buttonReleased[MouseLeft] and isClick:
      echo "Single release at position: ", mousePos

  # Selected agent
  if selection != nil and selection.isAgent:
    bxy.drawRect(rect(0, selection.agentId.float32 * traceHeight,
      panel.rect.w.float32, traceHeight.float32), color(0.3, 0.3, 0.3, 1.0))

  # Current step
  bxy.drawRect(rect(step.float32 * traceWidth, 0.0, traceWidth,
    panel.rect.h.float32), color(0.5, 0.5, 0.5, 0.5))

  # Agents
  for obj in replay.agents:
    let j = obj.agentId
    for i in 0..<replay.maxSteps:
      let pos = vec2(i.float32 * traceWidth + traceWidth / 2, j.float32 *
        traceHeight + traceHeight / 2)
      if obj.isFrozen.len > 1 and obj.isFrozen[i]:
        bxy.drawImage("trace/frozen", pos, angle = 0, scale = 1/200)
      else:
        let actionId = obj.actionId[i]
        if actionId >= 0:
          if obj.actionSuccess[i]:
            bxy.drawImage(replay.traceImages[actionId], pos, angle = 0,
              scale = 1/200.0)
          else:
            bxy.drawImage("trace/invalid", pos, angle = 0, scale = 1/200)

      let reward = obj.currentReward[i]
      if reward > 0:
        bxy.drawImage("resources/reward", vec2(pos.x, pos.y + (traceHeight /
            2) - 32/256), angle = 0, scale = 1/800)

      if settings.showResources and i > 0:
        let gainMap = obj.gainMap[i]
        for item in gainMap:
          for j in 0..<item.count:
            bxy.drawImage(replay.itemImages[item.itemId], vec2(pos.x, pos.y - (
                traceHeight / 2) + ((j + 1) * 32/256)), angle = 0, scale = 1/800)

  panel.endPanAndZoom()
