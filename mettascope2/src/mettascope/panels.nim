import
  vmath, bumpy, windy, boxy, chroma, fidget2, fidget2/[hybridrender, common],
  common, utils

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
      when defined(emscripten):
        let scrollK = 0.0003
      else:
        let scrollK = 0.03
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

    #let area = panel.scrollArea * panel.zoom
    #let x = panel.rect.x / 2
    #let y = panel.rect.y / 2
    #panel.pos = vec2(
    #  clamp(panel.pos.x, area.x - x, area.x + area.w + x),
    #  clamp(panel.pos.y, area.y - y, area.y + area.h + y))

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

  echo "Updating area sizes: ", area.node.name
  area.node.position = vec2(0, 0)
  area.node.size = vec2(area.rect.w.float32, area.rect.h.float32)
  echo " pos: ", area.node.position, " size: ", area.node.size
  area.node.dirty = true

  for num, panel in area.panels:
    echo "Updating panel sizes: ", panel.name
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

    panel.node.position = vec2(0, HeaderSize)
    panel.node.size = vec2(panel.rect.w.float32, panel.rect.h.float32)
    echo " pos: ", panel.node.position, " size: ", panel.node.size
    panel.node.dirty = true

  echo "Updating subarea sizes"
  for subarea in area.areas:
    updatePanelsSizes(subarea)
