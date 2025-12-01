import
  std/math,
  vmath, bumpy, windy, boxy, chroma,
  common, environment

proc updateMouse*(panel: Panel) =
  let scale = window.contentScale
  let panelRect = panel.rect.rect
  let logicalRect = Rect(
    x: panelRect.x / scale,
    y: panelRect.y / scale,
    w: panelRect.w / scale,
    h: panelRect.h / scale
  )

  let mousePos = logicalMousePos(window)
  let insideRect = mousePos.x >= logicalRect.x and mousePos.x <= logicalRect.x + logicalRect.w and
    mousePos.y >= logicalRect.y and mousePos.y <= logicalRect.y + logicalRect.h

  panel.hasMouse = panel.visible and ((not mouseCaptured and insideRect) or
    (mouseCaptured and mouseCapturedPanel == panel))

  # Handle focus changes
  if panel.hasMouse and window.buttonPressed[MouseLeft]:
    panel.focused = true

proc clampMapPan*(panel: Panel) =
  ## Clamp pan so the world map remains at least partially visible.
  let zoomScale = panel.zoom * panel.zoom
  if zoomScale <= 0:
    return

  let scale = window.contentScale.float32
  let panelRect = panel.rect.rect
  let rectW = panelRect.w / scale
  let rectH = panelRect.h / scale

  if rectW <= 0 or rectH <= 0:
    return

  let mapMinX = -0.5'f32
  let mapMinY = -0.5'f32
  let mapMaxX = MapWidth.float32 - 0.5'f32
  let mapMaxY = MapHeight.float32 - 0.5'f32
  let mapWidth = mapMaxX - mapMinX
  let mapHeight = mapMaxY - mapMinY

  let viewHalfW = rectW / (2.0'f32 * zoomScale)
  let viewHalfH = rectH / (2.0'f32 * zoomScale)

  var cx = (rectW / 2.0'f32 - panel.pos.x) / zoomScale
  var cy = (rectH / 2.0'f32 - panel.pos.y) / zoomScale

  let minVisiblePixels = min(500.0'f32, min(rectW, rectH) * 0.5'f32)
  let minVisibleWorld = minVisiblePixels / zoomScale
  let maxVisibleUnitsX = min(minVisibleWorld, mapWidth / 2.0'f32)
  let maxVisibleUnitsY = min(minVisibleWorld, mapHeight / 2.0'f32)

  let minCenterX = mapMinX + maxVisibleUnitsX - viewHalfW
  let maxCenterX = mapMaxX - maxVisibleUnitsX + viewHalfW
  let minCenterY = mapMinY + maxVisibleUnitsY - viewHalfH
  let maxCenterY = mapMaxY - maxVisibleUnitsY + viewHalfH

  cx = cx.clamp(minCenterX, maxCenterX)
  cy = cy.clamp(minCenterY, maxCenterY)

  panel.pos.x = rectW / 2.0'f32 - cx * zoomScale
  panel.pos.y = rectH / 2.0'f32 - cy * zoomScale

proc beginPanAndZoom*(panel: Panel) =
  ## Pan and zoom the map.

  bxy.saveTransform()

  updateMouse(panel)

  if panel.hasMouse:
    if window.buttonPressed[MouseLeft]:
      common.mouseCaptured = true
      common.mouseCapturedPanel = panel
      mouseDownPos = logicalMousePos(window)

    if window.buttonDown[MouseLeft] or window.buttonDown[MouseMiddle]:
      panel.vel = logicalMouseDelta(window)
    else:
      panel.vel *= 0.9

    panel.pos += panel.vel

    if window.scrollDelta.y != 0:
      let panelRect = panel.rect.rect
      let scale = window.contentScale.float32
      let rectOrigin = vec2(panelRect.x / scale, panelRect.y / scale)
      let localMouse = logicalMousePos(window) - rectOrigin

      let zoomSensitivity = when defined(emscripten): 0.002 else: 0.005
      let oldMat = translate(panel.pos) * scale(vec2(panel.zoom*panel.zoom, panel.zoom*panel.zoom))
      let oldWorldPoint = oldMat.inverse() * localMouse

      let zoomFactor64 = pow(1.0 - zoomSensitivity, window.scrollDelta.y.float64)
      let zoomFactor = zoomFactor64.float32
      panel.zoom = clamp(panel.zoom * zoomFactor, panel.minZoom, panel.maxZoom)

      let newMat = translate(panel.pos) * scale(vec2(panel.zoom*panel.zoom, panel.zoom*panel.zoom))
      let newWorldPoint = newMat.inverse() * localMouse
      panel.pos += (newWorldPoint - oldWorldPoint) * (panel.zoom * panel.zoom)

  clampMapPan(panel)

  let scale = window.contentScale.float32
  bxy.translate(panel.pos * scale)
  let zoomScale = panel.zoom * panel.zoom * scale
  bxy.scale(vec2(zoomScale, zoomScale))

proc endPanAndZoom*(panel: Panel) =
  bxy.restoreTransform()

proc beginDraw*(panel: Panel) =
  bxy.pushLayer()
  bxy.saveTransform()

  let panelRect = panel.rect.rect
  bxy.translate(vec2(panelRect.x, panelRect.y))

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
  # Simplified: all panels fill the area; no tabs/header
  for panel in area.panels:
    panel.rect = area.rect
  for subarea in area.areas:
    updatePanelsSizes(subarea)

## drawFrame removed with tab/header simplification
