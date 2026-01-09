# This example shows a draggable panel UI like in a large editor like VS Code or Blender.

import
  std/[sequtils, strformat],
  bumpy, chroma, windy, boxy, silky,
  common, windowstate

type
  AreaLayout* = enum
    Horizontal
    Vertical

  Area* = ref object
    layout*: AreaLayout
    areas*: seq[Area]
    panels*: seq[Panel]
    split*: float32
    selectedPanelNum*: int
    rect*: Rect # Calculated during draw

  PanelDraw = proc(panel: Panel, frameId: string, contentPos: Vec2, contentSize: Vec2)

  Panel* = ref object
    name*: string
    parentArea*: Area
    draw*: PanelDraw

  ZoomInfo* = ref object
    ## Used to track the zoom state of a world map and others.
    rect*: IRect
    pos*: Vec2
    vel*: Vec2
    zoom*: float32 = 10
    zoomVel*: float32
    minZoom*: float32 = 0.5
    maxZoom*: float32 = 50
    scrollArea*: Rect
    hasMouse*: bool = false
    dragging*: bool = false

  AreaScan* = enum
    Header
    Body
    North
    South
    East
    West

const
  AreaHeaderHeight = 32.0
  AreaMargin = 6.0
  BackgroundColor = parseHtmlColor("#222222").rgbx

var
  worldMapZoomInfo*: ZoomInfo
  viewStateChanged*: bool = false  ## Set when pan/zoom changes, consumed by state saver
  panelRegistry*: seq[(string, PanelDraw)]  ## Registry of panel names to draw functions

proc clampMapPan*(zoomInfo: ZoomInfo) =
  ## Clamp pan so the world map remains at least partially visible.
  if replay.isNil:
    return

  let zoomScale = zoomInfo.zoom * zoomInfo.zoom
  if zoomScale <= 0:
    return

  # Map bounds in world units (tiles), assuming tiles span [i-0.5, i+0.5].
  let
    mapMinX = -0.5f
    mapMinY = -0.5f
    mapMaxX = replay.mapSize[0].float32 - 0.5f
    mapMaxY = replay.mapSize[1].float32 - 0.5f

  let
    mapWidth = mapMaxX - mapMinX
    mapHeight = mapMaxY - mapMinY

  # View half-size in world units.
  let
    rectW = zoomInfo.rect.w.float32
    rectH = zoomInfo.rect.h.float32
    viewHalfW = rectW / (2.0f * zoomScale)
    viewHalfH = rectH / (2.0f * zoomScale)

  # Current view center in world units given screen-space pan.
  var
    cx = (rectW / 2.0f - zoomInfo.pos.x) / zoomScale
    cy = (rectH / 2.0f - zoomInfo.pos.y) / zoomScale

  # Require a minimum number of on-screen pixels of the map to remain visible.
  # Scale with panel size so small panels are not over-clamped.
  let minVisiblePixels = min(500.0f, min(rectW, rectH) * 0.5f)
  let minVisibleWorld = minVisiblePixels / zoomScale

  # Do not require more visibility than half the map size.
  let
    maxVisibleUnitsX = min(minVisibleWorld, mapWidth / 2.0f)
    maxVisibleUnitsY = min(minVisibleWorld, mapHeight / 2.0f)

  # Clamp the center so some of the map stays visible horizontally and vertically.
  let
    minCenterX = mapMinX + maxVisibleUnitsX - viewHalfW
    maxCenterX = mapMaxX - maxVisibleUnitsX + viewHalfW
    minCenterY = mapMinY + maxVisibleUnitsY - viewHalfH
    maxCenterY = mapMaxY - maxVisibleUnitsY + viewHalfH

  cx = cx.clamp(minCenterX, maxCenterX)
  cy = cy.clamp(minCenterY, maxCenterY)

  # Recompute screen-space pan from clamped world-space center.
  zoomInfo.pos.x = rectW / 2.0f - cx * zoomScale
  zoomInfo.pos.y = rectH / 2.0f - cy * zoomScale

proc beginPanAndZoom*(zoomInfo: ZoomInfo) =
  ## Pan and zoom the map.

  bxy.saveTransform()

  if zoomInfo.hasMouse:
    if window.buttonPressed[MouseLeft]:
      zoomInfo.dragging = true
    if not window.buttonDown[MouseLeft] and zoomInfo.dragging:
      zoomInfo.dragging = false

  if zoomInfo.dragging:
    if window.buttonDown[MouseLeft] or window.buttonDown[MouseMiddle]:

      zoomInfo.vel = window.mouseDelta.vec2
      settings.lockFocus = false
    else:
      zoomInfo.vel *= 0.9

    if zoomInfo.vel.x != 0 or zoomInfo.vel.y != 0:
      viewStateChanged = true
    zoomInfo.pos += zoomInfo.vel

  if zoomInfo.hasMouse:
    if window.scrollDelta.y != 0:
      # Apply zoom at focal point (mouse position or agent position if pinned).
      let localMousePos = window.mousePos.vec2 - zoomInfo.rect.xy.vec2
      let zoomSensitivity = 0.005

      let oldMat = translate(vec2(zoomInfo.pos.x, zoomInfo.pos.y)) *
        scale(vec2(zoomInfo.zoom*zoomInfo.zoom, zoomInfo.zoom*zoomInfo.zoom))

      # Use agent position as focal point if lockFocus is enabled and agent is selected
      let focalPoint = if settings.lockFocus and selection != nil:
        # Convert agent's world position to screen space
        let agentWorldPos = vec2(selection.location.at(step).x.float32, selection.location.at(step).y.float32)
        oldMat * agentWorldPos
      else:
        localMousePos

      let oldWorldPoint = oldMat.inverse() * focalPoint

      # Apply zoom with multiplicative scaling.
      # keeps zoom consistent when zoomed far out or zoomed far in.
      let zoomFactor = pow(1.0 - zoomSensitivity, window.scrollDelta.y)
      zoomInfo.zoom *= zoomFactor
      zoomInfo.zoom = clamp(zoomInfo.zoom, zoomInfo.minZoom, zoomInfo.maxZoom)
      viewStateChanged = true

      let newMat = translate(vec2(zoomInfo.pos.x, zoomInfo.pos.y)) *
        scale(vec2(zoomInfo.zoom*zoomInfo.zoom, zoomInfo.zoom*zoomInfo.zoom))
      let newWorldPoint = newMat.inverse() * focalPoint

      # Adjust pan position to keep the same world point under the focal point.
      zoomInfo.pos += (newWorldPoint - oldWorldPoint) * (zoomInfo.zoom*zoomInfo.zoom)

  clampMapPan(zoomInfo)

  bxy.translate(zoomInfo.pos)
  let zoomScale = zoomInfo.zoom * zoomInfo.zoom
  bxy.scale(vec2(zoomScale, zoomScale))

proc endPanAndZoom*(zoomInfo: ZoomInfo) =
  bxy.restoreTransform()

proc snapToPixels(rect: Rect): Rect =
  rect(rect.x.int.float32, rect.y.int.float32, rect.w.int.float32, rect.h.int.float32)

var
  rootArea*: Area
  dragArea: Area # For resizing splits
  dragPanel: Panel # For moving panels
  dropHighlight: Rect
  showDropHighlight: bool

  maybeDragStartPos: Vec2
  maybeDragPanel: Panel

proc movePanels*(area: Area, panels: seq[Panel])

proc clear*(area: Area) =
  ## Clear the area.
  for i in 0 ..< area.panels.len:
    area.panels[i].parentArea = nil
  for i in 0 ..< area.areas.len:
    area.areas[i].clear()
  area.panels.setLen(0)
  area.areas.setLen(0)

proc removeBlankAreas*(area: Area) =
  ## Remove blank areas recursively.
  if area.areas.len > 0:
    assert area.areas.len == 2
    if area.areas[0].panels.len == 0 and area.areas[0].areas.len == 0:
      if area.areas[1].panels.len > 0:
        area.movePanels(area.areas[1].panels)
        area.areas.setLen(0)
      elif area.areas[1].areas.len > 0:
        let oldAreas = area.areas
        area.areas = area.areas[1].areas
        area.split = oldAreas[1].split
        area.layout = oldAreas[1].layout
      else:
        discard
    elif area.areas[1].panels.len == 0 and area.areas[1].areas.len == 0:
      if area.areas[0].panels.len > 0:
        area.movePanels(area.areas[0].panels)
        area.areas.setLen(0)
      elif area.areas[0].areas.len > 0:
        let oldAreas = area.areas
        area.areas = area.areas[0].areas
        area.split = oldAreas[0].split
        area.layout = oldAreas[0].layout
      else:
        discard

    for i in 0 ..< area.areas.len:
      removeBlankAreas(area.areas[i])

proc addPanel*(area: Area, name: string, draw: PanelDraw) =
  ## Add a panel to the area.
  let panel = Panel(name: name, parentArea: area, draw: draw)
  area.panels.add(panel)

proc movePanel*(area: Area, panel: Panel) =
  ## Move a panel to this area.
  let idx = panel.parentArea.panels.find(panel)
  if idx != -1:
    panel.parentArea.panels.delete(idx)
  area.panels.add(panel)
  panel.parentArea = area

proc insertPanel*(area: Area, panel: Panel, index: int) =
  ## Insert a panel into this area at a specific index.
  let idx = panel.parentArea.panels.find(panel)
  var finalIndex = index

  # If moving within the same area, adjust index if we're moving forward
  if panel.parentArea == area and idx != -1:
    if idx < index:
      finalIndex = index - 1

  if idx != -1:
    panel.parentArea.panels.delete(idx)

  # Clamp index to be safe
  finalIndex = clamp(finalIndex, 0, area.panels.len)

  area.panels.insert(panel, finalIndex)
  panel.parentArea = area
  # Update selection to the new panel position
  area.selectedPanelNum = finalIndex

proc getTabInsertInfo(area: Area, mousePos: Vec2): (int, Rect) =
  ## Get the insert information for a tab.
  var x = area.rect.x + 4
  let headerH = AreaHeaderHeight

  # If no panels, insert at 0
  if area.panels.len == 0:
    return (0, rect(x, area.rect.y + 2, 4, headerH - 4))

  var bestIndex = 0
  var minDist = float32.high
  var bestX = x

  # Check before first tab (index 0)
  let dist0 = abs(mousePos.x - x)
  minDist = dist0
  bestX = x
  bestIndex = 0

  for i in 0 ..< area.panels.len:
    let panel = area.panels[i]
    let textSize = sk.getTextSize("Default", panel.name)
    let tabW = textSize.x + 16

    # The gap after this tab (index i + 1)
    let gapX = x + tabW + 2
    let dist = abs(mousePos.x - gapX)
    if dist < minDist:
      minDist = dist
      bestIndex = i + 1
      bestX = gapX

    x += tabW + 2

  return (bestIndex, rect(bestX - 2, area.rect.y + 2, 4, headerH - 4))

proc movePanels*(area: Area, panels: seq[Panel]) =
  ## Move multiple panels to this area.
  var panelList = panels # Copy
  for panel in panelList:
    area.movePanel(panel)

proc split*(area: Area, layout: AreaLayout) =
  ## Split the area.
  let
    area1 = Area(rect: area.rect) # inherit rect initially
    area2 = Area(rect: area.rect)
  area.layout = layout
  area.split = 0.5
  area.areas.add(area1)
  area.areas.add(area2)

proc scan*(area: Area): (Area, AreaScan, Rect) =
  ## Scan the area to find the target under mouse.
  let mousePos = window.mousePos.vec2
  var
    targetArea: Area
    areaScan: AreaScan
    resRect: Rect

  proc visit(area: Area) =
    if not mousePos.overlaps(area.rect):
      return

    if area.areas.len > 0:
      for i in 0 ..< area.areas.len:
        visit(area.areas[i])
    else:
      let
        headerRect = rect(
          area.rect.xy,
          vec2(area.rect.w, AreaHeaderHeight)
        )
        bodyRect = rect(
          area.rect.xy + vec2(0, AreaHeaderHeight),
          vec2(area.rect.w, area.rect.h - AreaHeaderHeight)
        )
        northRect = rect(
          area.rect.xy + vec2(0, AreaHeaderHeight),
          vec2(area.rect.w, area.rect.h * 0.2)
        )
        southRect = rect(
          area.rect.xy + vec2(0, area.rect.h * 0.8),
          vec2(area.rect.w, area.rect.h * 0.2)
        )
        eastRect = rect(
          area.rect.xy + vec2(area.rect.w * 0.8, 0) + vec2(0, AreaHeaderHeight),
          vec2(area.rect.w * 0.2, area.rect.h - AreaHeaderHeight)
        )
        westRect = rect(
          area.rect.xy + vec2(0, 0) + vec2(0, AreaHeaderHeight),
          vec2(area.rect.w * 0.2, area.rect.h - AreaHeaderHeight)
        )

      if mousePos.overlaps(headerRect):
        areaScan = Header
        resRect = headerRect
      elif mousePos.overlaps(northRect):
        areaScan = North
        resRect = northRect
      elif mousePos.overlaps(southRect):
        areaScan = South
        resRect = southRect
      elif mousePos.overlaps(eastRect):
        areaScan = East
        resRect = eastRect
      elif mousePos.overlaps(westRect):
        areaScan = West
        resRect = westRect
      elif mousePos.overlaps(bodyRect):
        areaScan = Body
        resRect = bodyRect

      targetArea = area

  visit(rootArea)
  return (targetArea, areaScan, resRect)

proc drawAreaRecursive(area: Area, r: Rect) =
  area.rect = r.snapToPixels()

  if area.areas.len > 0:
    let m = AreaMargin / 2
    if area.layout == Horizontal:
      # Top/Bottom
      let splitPos = r.h * area.split

      # Handle split resizing
      let splitRect = rect(r.x, r.y + splitPos - 2, r.w, 4)

      if dragArea == nil and window.mousePos.vec2.overlaps(splitRect):
        sk.cursor = Cursor(kind: ResizeUpDownCursor)
        if window.buttonPressed[MouseLeft]:
          dragArea = area

      let r1 = rect(r.x, r.y, r.w, splitPos - m)
      let r2 = rect(r.x, r.y + splitPos + m, r.w, r.h - splitPos - m)
      drawAreaRecursive(area.areas[0], r1)
      drawAreaRecursive(area.areas[1], r2)

    else:
      # Left/Right
      let splitPos = r.w * area.split

      let splitRect = rect(r.x + splitPos - 2, r.y, 4, r.h)

      if dragArea == nil and window.mousePos.vec2.overlaps(splitRect):
        sk.cursor = Cursor(kind: ResizeLeftRightCursor)
        if window.buttonPressed[MouseLeft]:
          dragArea = area

      let r1 = rect(r.x, r.y, splitPos - m, r.h)
      let r2 = rect(r.x + splitPos + m, r.y, r.w - splitPos - m, r.h)
      drawAreaRecursive(area.areas[0], r1)
      drawAreaRecursive(area.areas[1], r2)

  elif area.panels.len > 0:
    # Draw Panel
    if area.selectedPanelNum > area.panels.len - 1:
      area.selectedPanelNum = area.panels.len - 1

    # Draw Header
    let headerRect = rect(r.x, r.y, r.w, AreaHeaderHeight)
    sk.draw9Patch("panel.header.9patch", 3, headerRect.xy, headerRect.wh)

    # Draw Tabs
    var x = r.x + 4
    sk.pushClipRect(rect(r.x, r.y, r.w - 2, AreaHeaderHeight))
    let numPanels = area.panels.len
    for i in 0 ..< numPanels:
      let panel = area.panels[i]
      let textSize = sk.getTextSize("Default", panel.name)
      let tabW = textSize.x + 16
      let tabRect = rect(x, r.y + 4, tabW, AreaHeaderHeight - 4)

      let isSelected = i == area.selectedPanelNum
      let isHovered = window.mousePos.vec2.overlaps(tabRect)

      # Handle Tab Clicks and Dragging
      if isHovered:
        if window.buttonPressed[MouseLeft]:
          if area.selectedPanelNum != i:
            viewStateChanged = true  # Tab changed, save state
          area.selectedPanelNum = i
          # Only start dragging if the mouse moves 10 pixels.
          maybeDragStartPos = window.mousePos.vec2
          maybeDragPanel = panel
        elif window.buttonDown[MouseLeft] and dragPanel == panel:
          # Dragging started
          discard

      if window.buttonDown[MouseLeft]:
        if maybeDragPanel != nil and (maybeDragStartPos - window.mousePos.vec2).length() > 10:
          dragPanel = maybeDragPanel
          maybeDragStartPos = vec2(0, 0)
          maybeDragPanel = nil
      else:
        maybeDragStartPos = vec2(0, 0)
        maybeDragPanel = nil

      if isSelected:
        sk.draw9Patch("panel.tab.selected.9patch", 3, tabRect.xy, tabRect.wh, rgbx(255, 255, 255, 255))
      elif isHovered:
        sk.draw9Patch("panel.tab.hover.9patch", 3, tabRect.xy, tabRect.wh, rgbx(255, 255, 255, 255))
      else:
        sk.draw9Patch("panel.tab.9patch", 3, tabRect.xy, tabRect.wh)

      discard sk.drawText("Default", panel.name, vec2(x + 8, r.y + 4 + 2), rgbx(255, 255, 255, 255))

      x += tabW + 2
    sk.popClipRect()

    # Draw Content
    let contentRect = rect(r.x, r.y + AreaHeaderHeight, r.w, r.h - AreaHeaderHeight)
    let activePanel = area.panels[area.selectedPanelNum]
    let frameId = "panel:" & $cast[uint](activePanel)
    let contentPos = vec2(contentRect.x, contentRect.y)
    let contentSize = vec2(contentRect.w, contentRect.h)

    activePanel.draw(activePanel, frameId, contentPos, contentSize)

proc drawPanels*() =

  # Reset cursor
  sk.cursor = Cursor(kind: ArrowCursor)

  # Update Dragging Split
  if dragArea != nil:
    if not window.buttonDown[MouseLeft]:
      dragArea = nil
      viewStateChanged = true  # Split changed, save state
    else:
      if dragArea.layout == Horizontal:
        sk.cursor = Cursor(kind: ResizeUpDownCursor)
        dragArea.split = (window.mousePos.vec2.y - dragArea.rect.y) / dragArea.rect.h
      else:
        sk.cursor = Cursor(kind: ResizeLeftRightCursor)
        dragArea.split = (window.mousePos.vec2.x - dragArea.rect.x) / dragArea.rect.w
      dragArea.split = clamp(dragArea.split, 0.1, 0.9)

  # Update Dragging Panel
  showDropHighlight = false
  if dragPanel != nil:
    if not window.buttonDown[MouseLeft]:
      # Drop
      let (targetArea, areaScan, _) = rootArea.scan()
      if targetArea != nil:
        case areaScan:
          of Header:
            let (idx, _) = targetArea.getTabInsertInfo(window.mousePos.vec2)
            targetArea.insertPanel(dragPanel, idx)
          of Body:
            targetArea.movePanel(dragPanel)
          of North:
            targetArea.split(Horizontal)
            targetArea.areas[0].movePanel(dragPanel)
            targetArea.areas[1].movePanels(targetArea.panels)
          of South:
            targetArea.split(Horizontal)
            targetArea.areas[1].movePanel(dragPanel)
            targetArea.areas[0].movePanels(targetArea.panels)
          of East:
            targetArea.split(Vertical)
            targetArea.areas[1].movePanel(dragPanel)
            targetArea.areas[0].movePanels(targetArea.panels)
          of West:
            targetArea.split(Vertical)
            targetArea.areas[0].movePanel(dragPanel)
            targetArea.areas[1].movePanels(targetArea.panels)

        rootArea.removeBlankAreas()
        viewStateChanged = true  # Panel moved, save state
      dragPanel = nil
    else:
      # Dragging
      let (targetArea, areaScan, rect) = rootArea.scan()
      dropHighlight = rect
      showDropHighlight = true

      if targetArea != nil and areaScan == Header:
         let (_, highlightRect) = targetArea.getTabInsertInfo(window.mousePos.vec2)
         dropHighlight = highlightRect

  # Draw Areas
  drawAreaRecursive(rootArea, rect(0, 64, window.size.x.float32, window.size.y.float32 - 64 * 3))

  # Draw Drop Highlight
  if showDropHighlight and dragPanel != nil:
    sk.drawRect(dropHighlight.xy, dropHighlight.wh, rgbx(255, 255, 0, 100))

    # Draw dragging ghost
    let label = dragPanel.name
    let textSize = sk.getTextSize("Default", label)
    let size = textSize + vec2(16, 8)
    sk.draw9Patch("tooltip.9patch", 4, window.mousePos.vec2 + vec2(10, 10), size, rgbx(255, 255, 255, 200))
    discard sk.drawText("Default", label, window.mousePos.vec2 + vec2(18, 14), rgbx(255, 255, 255, 255))

proc captureAreaState*(area: Area): AreaState =
  ## Capture the current state of an area tree for serialization.
  result.layout = area.layout.ord
  result.split = area.split
  result.selectedPanelNum = area.selectedPanelNum
  # Copy seqs to avoid modification issues
  let panelsCopy = area.panels
  let areasCopy = area.areas
  for panel in panelsCopy:
    result.panelNames.add(panel.name)
  for subarea in areasCopy:
    result.areas.add(captureAreaState(subarea))

proc capturePanelState*(): AreaState =
  ## Capture the current panel layout state.
  if rootArea != nil:
    result = captureAreaState(rootArea)

proc registerPanel*(name: string, draw: PanelDraw) =
  ## Register a panel draw function by name for later reconstruction.
  panelRegistry.add((name, draw))

proc getPanelDraw(name: string): PanelDraw =
  ## Get a panel draw function by name.
  for (n, d) in panelRegistry:
    if n == name:
      return d
  return nil

proc rebuildAreaFromState*(state: AreaState): Area =
  ## Rebuild an area tree from saved state.
  result = Area()
  result.layout = AreaLayout(state.layout)
  result.split = state.split
  result.selectedPanelNum = state.selectedPanelNum

  if state.areas.len > 0:
    # Has sub-areas
    for subState in state.areas:
      result.areas.add(rebuildAreaFromState(subState))
  else:
    # Leaf area with panels
    for panelName in state.panelNames:
      let drawProc = getPanelDraw(panelName)
      if drawProc != nil:
        let panel = Panel(name: panelName, parentArea: result, draw: drawProc)
        result.panels.add(panel)
      else:
        echo "Warning: Unknown panel '", panelName, "' in saved state"

proc applyPanelState*(state: AreaState) =
  ## Apply saved panel state by rebuilding the entire panel tree.
  if state.areas.len > 0 or state.panelNames.len > 0:
    rootArea = rebuildAreaFromState(state)

proc captureZoomState*(): ZoomState =
  ## Capture the current zoom state with center in world coordinates.
  if worldMapZoomInfo != nil:
    result.zoom = worldMapZoomInfo.zoom
    # Calculate center in world coordinates from pan position
    let
      z = worldMapZoomInfo.zoom * worldMapZoomInfo.zoom
      rectW = worldMapZoomInfo.rect.w.float32
      rectH = worldMapZoomInfo.rect.h.float32
    if z > 0:
      result.centerX = (rectW / 2.0f - worldMapZoomInfo.pos.x) / z
      result.centerY = (rectH / 2.0f - worldMapZoomInfo.pos.y) / z
