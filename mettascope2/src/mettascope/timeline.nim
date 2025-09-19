import
  std/[times, math, strformat],
  boxy, vmath, windy, fidget2, fidget2/[hybridrender, common],
  common, panels, sim, actions, utils

const
  BgColor = parseHtmlColor("#1D1D1D")
  TraceWidth = 0.54 / 2

var
  # Drag state.
  scrubberActive = false
  minimapActive = false
  # Double click detection.
  lastClickTimeT: float64 = 0.0
  lastClickPosT: Vec2 = vec2(0, 0)
  clickIntervalT = 0.3 # seconds
  clickDistanceT = 10.0 # pixels

  # Figma nodes within GlobalTimeline.
  nodesBound = false
  nodeTimeline: Node
  nodeTimelineReadout: Node
  nodeStepCounter: Node
  nodeScrubberBg: Node
  nodeScrubber: Node

proc bindTimelineNodes() =
  if nodesBound or globalTimelinePanel.isNil or globalTimelinePanel.node.isNil:
    return
  nodeTimeline = globalTimelinePanel.node
  nodeTimelineReadout = nodeTimeline.find("**/TimelineReadout")
  nodeStepCounter = nodeTimeline.find("**/#step-counter")
  nodeScrubberBg = nodeTimeline.find("**/ScrubberBg")
  nodeScrubber = nodeTimeline.find("**/Scrubber")
  nodesBound = true

proc playControls*() =
  if window.buttonPressed[KeySpace]:
    play = not play
    stepFloat = step.float32
  if window.buttonPressed[KeyMinus]:
    playSpeed *= 0.5
    playSpeed = clamp(playSpeed, 0.00001, 60.0)
    play = true
  if window.buttonPressed[KeyEqual]:
    playSpeed *= 2
    playSpeed = clamp(playSpeed, 0.00001, 60.0)
    play = true

  if play:
    stepFloat += playSpeed
    if stepFloat >= replay.maxSteps.float32:
      echo "Requesting Python form timeline"
      requestPython = true
      stepFloat = replay.maxSteps.float32 - 1
    step = stepFloat.int
    step = step.clamp(0, replay.maxSteps - 1)

  if window.buttonPressed[KeyLeftBracket]:
    step -= 1
    step = clamp(step, 0, replay.maxSteps - 1)
    stepFloat = step.float32
    echo "step: ", step
  if window.buttonPressed[KeyRightBracket]:
    step += 1
    step = clamp(step, 0, replay.maxSteps - 1)
    stepFloat = step.float32
    echo "step: ", step

proc getStepFromX(localX, panelWidth: float32): int =
  ## Maps a local X coordinate within the timeline panel to a replay step.
  if replay.isNil or replay.maxSteps <= 0:
    return 0
  var
    trackLeft = nodeScrubberBg.position.x
    scrubberWidth = nodeScrubberBg.size.x
  let rel = clamp(localX - trackLeft, 0f, scrubberWidth)
  let fullSteps = max(1, replay.maxSteps - 1)
  result = int(floor((rel / scrubberWidth) * float32(replay.maxSteps)))
  result = clamp(result, 0, replay.maxSteps - 1)

proc onScrubberChange(localX, panelWidth: float32) =
  ## Updates global step based on local X.
  let s = getStepFromX(localX, panelWidth)
  step = s
  stepFloat = step.float32

proc centerTracesOnStep(targetStep: int) =
  ## Recenters the Agent Traces panel on the given step.
  if agentTracesPanel.isNil:
    return
  let zoom2 = agentTracesPanel.zoom * agentTracesPanel.zoom
  let worldX = targetStep.float32 * TraceWidth
  agentTracesPanel.pos.x = agentTracesPanel.rect.w.float32 / 2.0f - zoom2 * worldX

proc onTraceMinimapChange(localX, panelWidth: float32) =
  ## Pans the traces viewport based on local X.
  let s = getStepFromX(localX, panelWidth)
  centerTracesOnStep(s)

proc drawViewportMinimap(panel: Panel) =
  ## Draws the current Agent Traces viewport rectangle on the timeline.
  if agentTracesPanel.isNil or replay.isNil or replay.maxSteps <= 1:
    return
  let fullSteps = max(1, replay.maxSteps - 1)
  let scrubberWidth = nodeScrubberBg.size.x
  let scrubberLeft = nodeScrubberBg.position.x
  let zoom2 = agentTracesPanel.zoom * agentTracesPanel.zoom
  if zoom2 <= 0.00001f:
    return
  # Leftmost visible world X in traces panel.
  let worldXLeft = (-agentTracesPanel.pos.x) / zoom2
  let visibleWorldW = agentTracesPanel.rect.w.float32 / zoom2
  let startStep = max(0.0f, worldXLeft / TraceWidth)
  let stepsVisible = max(1.0f, visibleWorldW / TraceWidth)
  let tracesX = (startStep / float32(fullSteps)) * scrubberWidth
  let tracesW = (stepsVisible / float32(fullSteps)) * scrubberWidth
  # Draw a light rectangle indicating the viewport coverage.
  bxy.drawRect(
    rect(
      scrubberLeft + tracesX - 1,
      0,
      tracesW + 2,
      64
    ),
    color(1, 1, 1, 0.12)
  )

proc drawFrozenMarkers(panel: Panel) =
  ## Draws tick marks for the start of frozen states along the scrubber.
  if replay.isNil or replay.maxSteps <= 1:
    return
  let scrubberWidth = nodeScrubberBg.size.x
  let trackLeft = nodeScrubberBg.position.x
  let trackTop = nodeScrubberBg.position.y
  let fullSteps = max(1, replay.maxSteps - 1)
  for agent in replay.agents:
    var prevFrozen = false
    for j in 0 ..< replay.maxSteps:
      let isFrozen = (agent.isFrozen.len > j) and agent.isFrozen[j]
      if isFrozen and (not prevFrozen):
        let x = trackLeft + (float32(j) / float32(fullSteps)) * scrubberWidth
        # Draw icon above the scrubber and a small tick on the bar.
        bxy.drawImage("agents/frozen", vec2(x, trackTop - 22), angle = 0, scale = 1/10)
        bxy.drawRect(rect(x - 1, trackTop - 10, 2, 8), color(1, 1, 1, 1))
      prevFrozen = isFrozen

proc trackRect(panel: Panel): Rect =
  ## Track rectangle in panel-local coordinates.
  return Rect(x: nodeScrubberBg.position.x, y: nodeScrubberBg.position.y,
              w: nodeScrubberBg.size.x, h: nodeScrubberBg.size.y)

proc updateScrubber() =
  ## Timeline should act like a video playbar:
  ## - Scrubber is a filled bar from left to current step.
  ## - Step number is centered over the current end.
  if replay.isNil or not nodesBound:
    return
  # Step counter text and bubble position (move the entire readout frame)
  nodeStepCounter.text = $step
  nodeStepCounter.dirty = true
  # Position the bubble centered over the current end of the fill.
  let bubbleW = nodeTimelineReadout.size.x
  let trackLeft = nodeScrubberBg.position.x
  let trackWidth = nodeScrubberBg.size.x
  var bubbleX = trackLeft
  if replay.maxSteps > 1:
    let fullSteps = max(1, replay.maxSteps - 1)
    let progress = clamp(step.float32 / fullSteps.float32, 0f, 1f)
    let endX = trackLeft + progress * trackWidth
    bubbleX = endX - bubbleW * 0.5f
  # Clamp inside the track.
  bubbleX = clamp(bubbleX, trackLeft, trackLeft + trackWidth - bubbleW)
  if abs(nodeTimelineReadout.position.x - bubbleX) > 0.1f:
    nodeTimelineReadout.position = vec2(bubbleX, nodeTimelineReadout.position.y)
    nodeTimelineReadout.dirty = true
  # Scrubber fill width
  let trackLeft2 = nodeScrubberBg.position.x
  let trackWidth2 = nodeScrubberBg.size.x
  var fillW = 0f
  if replay.maxSteps > 1:
    let fullSteps = max(1, replay.maxSteps - 1)
    let progress = clamp(step.float32 / fullSteps.float32, 0f, 1f)
    fillW = progress * trackWidth2
  # Ensure the scrubber fill is visible at step 0: minimum width ~1px.
  fillW = max(fillW, 1f)
  # Anchor left and scale width using node scale to support VECTOR nodes.
  var changed = false
  if abs(nodeScrubber.position.x - trackLeft2) > 0.1f:
    nodeScrubber.position = vec2(trackLeft2, nodeScrubber.position.y)
    changed = true
  let baseW = (if nodeScrubber.origSize.x > 0: nodeScrubber.origSize.x else: max(1f, nodeScrubber.size.x))
  let scaleX = max(fillW / baseW, 0.001f)
  if abs(nodeScrubber.scale.x - scaleX) > 0.001f or abs(nodeScrubber.scale.y - 1f) > 0.001f:
    nodeScrubber.scale = vec2(scaleX, 1f)
    changed = true
  if changed:
    nodeScrubber.dirty = true

proc drawTimeline*(panel: Panel) =
  ## Draws the global timeline with scrubber, minimap, and markers, and handles input.
  if replay.isNil:
    return

  bindTimelineNodes()

  # Update the panel rect to match the node.
  panel.rect.x = panel.node.position.x.int32
  panel.rect.y = panel.node.position.y.int32
  panel.rect.w = panel.node.size.x.int32
  panel.rect.h = panel.node.size.y.int32

  # Handle mouse interactions.
  updateMouse(panel)
  let localMouse = window.mousePos.vec2 - panel.rect.xy.vec2

  if panel.hasMouse:
    if window.buttonPressed[MouseLeft]:
      let tr = trackRect(panel)
      let inTrack = (localMouse.x >= tr.x and localMouse.x <= tr.x + tr.w and
                     localMouse.y >= tr.y and localMouse.y <= tr.y + tr.h)
      mouseCaptured = true
      mouseCapturedPanel = panel
      scrubberActive = inTrack
      minimapActive = false
      if scrubberActive:
        onScrubberChange(localMouse.x, panel.rect.w.float32)
      else:
        # Click within the timeline panel but outside the track: block only.
        discard

    if mouseCaptured and mouseCapturedPanel == panel and window.buttonDown[MouseLeft]:
      if scrubberActive:
        onScrubberChange(localMouse.x, panel.rect.w.float32)

    if window.buttonReleased[MouseLeft]:
      scrubberActive = false
      minimapActive = false
      if mouseCaptured and mouseCapturedPanel == panel:
        mouseCaptured = false
        mouseCapturedPanel = nil

    # Double-click detection to center traces on step.
    if window.buttonPressed[MouseLeft]:
      let currentTime = epochTime()
      let isClick = dist(localMouse, lastClickPosT) < clickDistanceT
      if currentTime - lastClickTimeT < clickIntervalT and isClick:
        let s = getStepFromX(localMouse.x, panel.rect.w.float32)
        centerTracesOnStep(s)
      lastClickTimeT = currentTime
      lastClickPosT = localMouse

  # Draw trace viewport minimap window and frozen markers as an overlay in panel space.
  panel.beginDraw()
  drawViewportMinimap(panel)
  drawFrozenMarkers(panel)
  panel.endDraw()

  updateScrubber()
