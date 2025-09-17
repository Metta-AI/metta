import
  vmath, bumpy, windy, boxy, chroma, fidget2, fidget2/[hybridrender, common],
  common, math

const HeaderSize = 30

proc updateMouse*(panel: Panel) =
  let box = Rect(
    x: panel.rect.x.float32,
    y: panel.rect.y.float32,
    w: panel.rect.w.float32,
    h: panel.rect.h.float32
  )

  panel.hasMouse =
    (not mouseCaptured and window.mousePos.vec2.overlaps(box)) or
    (mouseCaptured and mouseCapturedPanel == panel)

proc clampMapPan*(panel: Panel) =
  ## Clamp pan so the world map remains at least partially visible.
  if replay.isNil:
    return

  let zoomScale = panel.zoom * panel.zoom
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
    rectW = panel.rect.w.float32
    rectH = panel.rect.h.float32
    viewHalfW = rectW / (2.0f * zoomScale)
    viewHalfH = rectH / (2.0f * zoomScale)

  # Current view center in world units given screen-space pan.
  var
    cx = (rectW / 2.0f - panel.pos.x) / zoomScale
    cy = (rectH / 2.0f - panel.pos.y) / zoomScale

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
  panel.pos.x = rectW / 2.0f - cx * zoomScale
  panel.pos.y = rectH / 2.0f - cy * zoomScale

proc beginPanAndZoom*(panel: Panel) =
  ## Pan and zoom the map.

  bxy.saveTransform()

  updateMouse(panel)

  if panel.hasMouse:
    if window.buttonPressed[MouseLeft]:
      mouseCaptured = true
      mouseCapturedPanel = panel

    if window.buttonDown[MouseLeft] or window.buttonDown[MouseMiddle]:
      panel.vel = window.mouseDelta.vec2
    else:
      panel.vel *= 0.9

    panel.pos += panel.vel

    if window.scrollDelta.y != 0:
      # Apply zoom at focal point (mouse position) with consistent sensitivity.
      let localMousePos = window.mousePos.vec2 - panel.rect.xy.vec2
      let zoomSensitivity = 0.005
      
      let oldMat = translate(vec2(panel.pos.x, panel.pos.y)) * scale(vec2(
          panel.zoom*panel.zoom, panel.zoom*panel.zoom))
      let oldWorldPoint = oldMat.inverse() * localMousePos
      
      # Apply zoom with multiplicative scaling.
      # keeps zoom consistent when zoomed far out or zoomed far in.
      let zoomFactor = pow(1.0 - zoomSensitivity, window.scrollDelta.y)
      panel.zoom *= zoomFactor
      panel.zoom = clamp(panel.zoom, panel.minZoom, panel.maxZoom)
      
      let newMat = translate(vec2(panel.pos.x, panel.pos.y)) * scale(vec2(
          panel.zoom*panel.zoom, panel.zoom*panel.zoom))
      let newWorldPoint = newMat.inverse() * localMousePos
      
      # Adjust pan position to keep the same world point under the mouse.
      panel.pos += (newWorldPoint - oldWorldPoint) * (panel.zoom*panel.zoom)

  clampMapPan(panel)

  bxy.translate(panel.pos)
  bxy.scale(vec2(panel.zoom*panel.zoom, panel.zoom*panel.zoom))

proc endPanAndZoom*(panel: Panel) =
  bxy.restoreTransform()

proc beginDraw*(panel: Panel) =
  bxy.pushLayer()
  bxy.saveTransform()

  bxy.translate(vec2(panel.rect.x.float32, panel.rect.y.float32))

proc endDraw*(panel: Panel) =

  bxy.restoreTransform()

  # Draw the mask.
  bxy.pushLayer()
  bxy.drawRect(
    rect = panel.rect.rect,
    color = color(1, 0, 0, 1.0)
  )
  bxy.popLayer(blendMode = MaskBlend)

  bxy.popLayer()

proc updatePanelsSizes*(area: Area) =
  # Update the sizes of the panels in the area and its subareas and subpanels.
  let newPos = vec2(area.rect.x.float32, area.rect.y.float32)
  if newPos != area.node.position:
    area.node.dirty = true
    area.node.position = newPos

  for num, panel in area.panels:
    if num == area.selectedPanelNum:
      panel.rect.x = area.rect.x
      panel.rect.y = area.rect.y + HeaderSize
      panel.rect.w = area.rect.w
      panel.rect.h = area.rect.h - HeaderSize
    else:
      panel.rect.x = 0
      panel.rect.y = 0
      panel.rect.w = 0
      panel.rect.h = 0

    let newPos = vec2(panel.rect.x.float32, panel.rect.y.float32)
    let newSize = vec2(panel.rect.w.float32, panel.rect.h.float32)
    if newPos != panel.node.position or newSize != panel.node.size:
      panel.node.dirty = true
      panel.node.position = newPos
      panel.node.size = newSize

  for subarea in area.areas:
    updatePanelsSizes(subarea)

proc select*(area: Area, name: string) =
  ## Selects the panel with the given name.
  for i, panel in area.panels:
    if panel.name == name:
      # Hide previous panel.
      area.panels[area.selectedPanelNum].node.visible = false
      let prevName = area.panels[area.selectedPanelNum].name
      area.node.children[area.selectedPanelNum].setVariant("State", "Default")

      # Show new panel.
      area.selectedPanelNum = i
      panel.node.visible = true
      area.node.children[area.selectedPanelNum].setVariant("State", "Selected")
      break

  for subarea in area.areas:
    select(subarea, name)

proc add*(area: Area, other: Area) =
  ## Adds an area to the current area.
  if area.panels.len > 0:
    raise newException(Exception, "Area already has panels, can't have both panels and areas")
  area.areas.add(other)

proc add*(area: Area, panel: Panel) =
  ## Adds a panel to the current area.
  if area.areas.len > 0:
    raise newException(Exception, "Area already has areas, can't have both panels and areas")
  area.panels.add(panel)
