import
  vmath, bumpy, windy, boxy, chroma,
  common, utils

const HeaderSize = 30

proc rect*(rect: IRect): Rect =
  Rect(x: rect.x.float32, y: rect.y.float32, w: rect.w.float32,
      h: rect.h.float32)

proc updateMouse*(panel: Panel) =
  let box = Rect(
    x: panel.rect.x.float32,
    y: panel.rect.y.float32,
    w: panel.rect.w.float32,
    h: panel.rect.h.float32
  )

  panel.hasMouse = (not mouseCaptured and window.mousePos.vec2.overlaps(box)) or
    (mouseCaptured and mouseCapturedPanel == panel)

  if panel.hasMouse and window.buttonPressed[MouseLeft]:
    mouseCaptured = true
    mouseCapturedPanel = panel

proc beginPanAndZoom*(panel: Panel) =
  ## Pan and zoom the map.

  updateMouse(panel)
  bxy.saveTransform()

  let center = vec2(panel.rect.w.float32 / 2, panel.rect.h.float32 / 2)
  let pos = translate(panel.pos)
  let ofs = translate(center)
  if panel.hasMouse:
    if window.buttonDown[MouseLeft] or window.buttonDown[MouseMiddle]:
      panel.vel = window.mouseDelta.vec2
    else:
      panel.vel *= 0.9

    panel.pos += panel.vel

    if window.scrollDelta.y != 0:
      when defined(emscripten):
        let scrollK = 0.00008
      else:
        let scrollK = 0.008
      panel.zoomVel = window.scrollDelta.y * scrollK
    else:
      panel.zoomVel *= 0.8

    # Screen space to panel space (centered)
    let mousePos = window.mousePos.vec2 -
      (vec2(panel.rect.x.float32, panel.rect.y.float32) + center)

    # Panel space to scroll space (pan and zoom)
    let oldZoom = panel.zoom
    panel.zoom *= 1.0 + panel.zoomVel
    panel.zoom = clamp(panel.zoom, panel.minZoom, panel.maxZoom) # TODO from size

    let newZoom = panel.zoom
    let old = (ofs * scale(vec2(oldZoom))) * pos
    let mat = (ofs * scale(vec2(newZoom))) * pos
    let before = old * mousePos
    let after = mat * mousePos
    #let before = panel.pos + mousePos / oldZoom
    #let after = panel.pos + mousePos / newZoom
    panel.pos += before - after

    #let area = panel.scrollArea * panel.zoom
    #let x = panel.rect.x / 2
    #let y = panel.rect.y / 2
    #panel.pos = vec2(
    #  clamp(panel.pos.x, area.x - x, area.x + area.w + x),
    #  clamp(panel.pos.y, area.y - y, area.y + area.h + y))
    bxy.setTransform(mat)
  else:
    #bxy.setTransform((ofs * scale(vec2(panel.zoom)) * pos).inverse())
    bxy.setTransform(ofs * scale(vec2(panel.zoom)) * pos)

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

  for subarea in area.areas:
    updatePanelsSizes(subarea)

proc drawFrame*(area: Area) =
  # Draw the frame of the area.

  # Draw the header ribbon background.
  bxy.saveTransform()
  bxy.translate(vec2(area.rect.x.float32, area.rect.y.float32))
  bxy.drawRect(
    rect = Rect(
      x: 0,
      y: 0,
      w: area.rect.w.float32,
      h: HeaderSize.float32
    ),
    color = color(0, 0, 0, 1)
  )

  var x = 10.0
  for num, panel in area.panels:
    let width = measureText(panel.name, 16).x + 20
    let panelBox = Rect(
      x: x.float32,
      y: 2,
      w: width,
      h: HeaderSize.float32 - 4
    )
    var color = parseHtmlColor("#282D35")
    if num == area.selectedPanelNum:
      color = parseHtmlColor("#43526A")

    if window.boxyMouse.vec2.overlaps(panelBox):
      color = parseHtmlColor("#FF0000")
      if window.buttonPressed[MouseLeft]:
        area.selectedPanelNum = num

    bxy.drawRect(
      rect = panelBox,
      color = color
    )
    bxy.drawText(
      panel.name,
      translate(vec2(x.float32 + 5, 4)),
      typeface,
      panel.name,
      16,
      color(1, 1, 1, 1)
    )

    x += width + 10

  bxy.restoreTransform()

  for subarea in area.areas:
    drawFrame(subarea)
