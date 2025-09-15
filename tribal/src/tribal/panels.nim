import
  vmath, bumpy, windy, boxy, chroma,
  common

proc updateMouse*(panel: Panel) =
  let box = panel.rect.rect

  panel.hasMouse = panel.visible and ((not common.mouseCaptured and window.mousePos.vec2.overlaps(box)) or
    (common.mouseCaptured and common.mouseCapturedPanel == panel))
  
  # Handle focus changes
  if panel.hasMouse and window.buttonPressed[MouseLeft]:
    panel.focused = true

proc beginPanAndZoom*(panel: Panel) =
  ## Pan and zoom the map.

  bxy.saveTransform()

  updateMouse(panel)

  if panel.hasMouse:
    if window.buttonPressed[MouseLeft]:
      common.mouseCaptured = true
      common.mouseCapturedPanel = panel

    if window.buttonDown[MouseLeft] or window.buttonDown[MouseMiddle]:
      panel.vel = window.mouseDelta.vec2
    else:
      panel.vel *= 0.9

    panel.pos += panel.vel

    if window.scrollDelta.y != 0:
      when defined(emscripten):
        let scrollK = 0.0003
      else:
        let scrollK = 0.15
      panel.zoomVel = window.scrollDelta.y * scrollK
    else:
      panel.zoomVel *= 0.8

    let oldMat = translate(vec2(panel.pos.x, panel.pos.y)) * scale(vec2(
        panel.zoom*panel.zoom, panel.zoom*panel.zoom))
    panel.zoom += panel.zoomVel
    panel.zoom = clamp(panel.zoom, panel.minZoom, panel.maxZoom)
    let newMat = translate(vec2(panel.pos.x, panel.pos.y)) * scale(vec2(
        panel.zoom*panel.zoom, panel.zoom*panel.zoom))
    let newAt = newMat.inverse() * window.mousePos.vec2
    let oldAt = oldMat.inverse() * window.mousePos.vec2
    panel.pos -= (oldAt - newAt).xy * (panel.zoom*panel.zoom)

  bxy.translate(panel.pos)
  bxy.scale(vec2(panel.zoom*panel.zoom, panel.zoom*panel.zoom))

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
