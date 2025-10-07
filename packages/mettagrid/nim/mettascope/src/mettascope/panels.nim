# This example shows a draggable panel UI like in a large editor like VS Code or Blender.

import
  std/[random, sequtils],
  fidget2, bumpy, chroma, windy, boxy, fidget2/hybridrender,
  common

const
  AreaHeaderHeight = 28
  AreaMargin = 6

var
  areaTemplate: Node
  panelHeaderTemplate: Node
  panelTemplate: Node
  rootArea*: Area
  dropHighlight: Node
  dragArea: Area
  objectInfoTemplate*: Node
  envConfigTemplate*: Node

proc updateMouse*(panel: Panel) =
  let box = Rect(
    x: panel.rect.x.float32,
    y: panel.rect.y.float32,
    w: panel.rect.w.float32,
    h: panel.rect.h.float32
  )

  panel.hasMouse =
    (not mouseCaptured and window.logicalMousePos.overlaps(box)) or
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
      panel.vel = window.logicalMouseDelta
    else:
      panel.vel *= 0.9

    panel.pos += panel.vel

    if window.scrollDelta.y != 0:
      # Apply zoom at focal point (mouse position) with consistent sensitivity.
      let localMousePos = window.logicalMousePos - panel.rect.xy.vec2
      let zoomSensitivity = 0.005

      let oldMat = translate(vec2(panel.pos.x, panel.pos.y)) *
        scale(vec2(panel.zoom*panel.zoom, panel.zoom*panel.zoom))
      let oldWorldPoint = oldMat.inverse() * localMousePos

      # Apply zoom with multiplicative scaling.
      # keeps zoom consistent when zoomed far out or zoomed far in.
      let zoomFactor = pow(1.0 - zoomSensitivity, window.scrollDelta.y)
      panel.zoom *= zoomFactor
      panel.zoom = clamp(panel.zoom, panel.minZoom, panel.maxZoom)

      let newMat = translate(vec2(panel.pos.x, panel.pos.y)) *
        scale(vec2(panel.zoom*panel.zoom, panel.zoom*panel.zoom))
      let newWorldPoint = newMat.inverse() * localMousePos

      # Adjust pan position to keep the same world point under the mouse.
      panel.pos += (newWorldPoint - oldWorldPoint) * (panel.zoom*panel.zoom)

  clampMapPan(panel)

  bxy.translate(panel.pos * window.contentScale)
  let zoomScale = panel.zoom * panel.zoom * window.contentScale
  bxy.scale(vec2(zoomScale, zoomScale))

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

proc clear*(area: Area) =
  ## Clears the area and all its subareas and panels.
  for panel in area.panels:
    if panel.node != nil:
      panel.header.remove()
      panel.node.remove()
    panel.parentArea = nil
  for subarea in area.areas:
    subarea.clear()
  if area.node != nil:
    area.node.remove()
  area.panels.setLen(0)
  area.areas.setLen(0)

proc movePanels*(area: Area, panels: seq[Panel])

proc removeBlankAreas*(area: Area) =
  ## Removes all areas that have no panels or subareas.

  if area.areas.len > 0:
    assert area.areas.len == 2
    if area.areas[0].panels.len == 0 and area.areas[0].areas.len == 0:
      if area.areas[1].panels.len > 0:
        area.movePanels(area.areas[1].panels)
        area.areas[0].node.remove()
        area.areas[1].node.remove()
        area.areas.setLen(0)
      elif area.areas[1].areas.len > 0:
        let oldAreas = area.areas
        area.areas = area.areas[1].areas
        for subarea in area.areas:
          area.node.addChild(subarea.node)
        area.split = oldAreas[1].split
        area.layout = oldAreas[1].layout
        oldAreas[0].node.remove()
        oldAreas[1].node.remove()
      else:
        discard # Both areas are blank, do nothing.

    elif area.areas[1].panels.len == 0 and area.areas[1].areas.len == 0:
      if area.areas[0].panels.len > 0:
        area.movePanels(area.areas[0].panels)
        area.areas[1].node.remove()
        area.areas[0].node.remove()
        area.areas.setLen(0)
      elif area.areas[0].areas.len > 0:
        let oldAreas = area.areas
        area.areas = area.areas[0].areas
        for subarea in area.areas:
          area.node.addChild(subarea.node)
        area.split = oldAreas[0].split
        area.layout = oldAreas[0].layout
        oldAreas[1].node.remove()
        oldAreas[0].node.remove()
      else:
        discard # Both areas are blank, do nothing.

    for subarea in area.areas:
      removeBlankAreas(subarea)

proc refresh*(area: Area, depth = 0) =
  if area.areas.len > 0:
    # Layout according to the layout.
    let m = AreaMargin/2
    if area.layout == Horizontal:
      # Split horizontally (top/bottom)
      let splitPos = area.node.size.y * area.split
      area.areas[0].node.position = vec2(0, 0).floor()
      area.areas[0].node.size = vec2(area.node.size.x, splitPos - m)
      area.areas[1].node.position = vec2(0, splitPos + m).floor()
      area.areas[1].node.size = vec2(area.node.size.x, area.node.size.y - splitPos - m).ceil()
    else:
      # Split vertically (left/right)
      let splitPos = area.node.size.x * area.split
      area.areas[0].node.position = vec2(0, 0).floor()
      area.areas[0].node.size = vec2(splitPos - m, area.node.size.y).ceil()
      area.areas[1].node.position = vec2(splitPos + m, 0).floor()
      area.areas[1].node.size = vec2(area.node.size.x - splitPos - m, area.node.size.y).ceil()

  for subarea in area.areas:
    subarea.refresh(depth + 1)

  if area.panels.len > 0:
    if area.selectedPanelNum > area.panels.len - 1:
      area.selectedPanelNum = area.panels.len - 1
    # Set the state of the headers.
    for i, panel in area.panels:
      if i != area.selectedPanelNum:
        panel.header.setVariant("State", "Default")
        panel.node.visible = false
      else:
        panel.header.setVariant("State", "Selected")
        panel.node.visible = true
        panel.node.position = vec2(0, AreaHeaderHeight)
        panel.node.size = area.node.size - vec2(0, AreaHeaderHeight)

proc findPanelByHeader*(node: Node): Panel =
  ## Finds the panel that contains the given header node.
  proc visit(area: Area, node: Node): Panel =
    for panel in area.panels:
      if panel.header == node:
        return panel
    for subarea in area.areas:
      let panel = visit(subarea, node)
      if panel != nil:
        return panel
    return nil
  return visit(rootArea, node)

proc findAreaByNode*(node: Node): Area =
  ## Finds the area that contains the given node.
  proc visit(area: Area): Area =
    if area.node == node:
      return area
    for subarea in area.areas:
      let area = visit(subarea)
      if area != nil:
        return area
  return visit(rootArea)

proc addPanel*(area: Area, panelType: PanelType, name: string): Panel =
  ## Adds a panel to the given area.
  let panel = Panel(name: name)
  panel.panelType = panelType
  panel.header = panelHeaderTemplate.copy()
  panel.header.find("title").text = name
  panel.node = panelTemplate.copy()
  area.panels.add(panel)
  panel.parentArea = area
  area.node.find("Header").addChild(panel.header)
  area.node.addChild(panel.node)
  return panel

proc movePanel*(area: Area, panel: Panel) =
  ## Moves the panel to the given area.
  panel.parentArea.panels.delete(panel.parentArea.panels.find(panel))
  area.panels.add(panel)
  panel.parentArea = area
  area.node.find("Header").addChild(panel.header)
  area.node.addChild(panel.node)

proc movePanels*(area: Area, panels: seq[Panel]) =
  ## Moves the panels to the given area.
  var panelList = panels.toSeq()
  for panel in panelList:
    area.movePanel(panel)

proc split*(area: Area, layout: AreaLayout) =
  ## Splits the area into two subareas.
  let
    area1 = Area(node: areaTemplate.copy())
    area2 = Area(node: areaTemplate.copy())
  area.layout = layout
  area.split = 0.5
  area.areas.add(area1)
  area.areas.add(area2)
  area.node.addChild(area1.node)
  area.node.addChild(area2.node)

type
  AreaScan = enum
    Header
    Body
    North
    South
    East
    West

proc scan*(area: Area): (Area,AreaScan, Rect) =
  let mousePos = window.logicalMousePos
  var targetArea: Area
  var areaScan: AreaScan
  var rect: Rect
  proc visit(area: Area) =
    let areaRect = rect(area.node.absolutePosition, area.node.size)
    if mousePos.overlaps(areaRect):
      if area.areas.len > 0:
        for subarea in area.areas:
          visit(subarea)
      else:
        let
          headerRect = rect(
            area.node.absolutePosition,
            vec2(area.node.size.x, AreaHeaderHeight)
          )
          bodyRect = rect(
            area.node.absolutePosition + vec2(0, AreaHeaderHeight),
            vec2(area.node.size.x, area.node.size.y - AreaHeaderHeight)
          )
          northRect = rect(
            area.node.absolutePosition + vec2(0, AreaHeaderHeight),
            vec2(area.node.size.x, area.node.size.y * 0.2)
          )
          southRect = rect(
            area.node.absolutePosition + vec2(0, area.node.size.y * 0.8),
            vec2(area.node.size.x, area.node.size.y * 0.2)
          )
          eastRect = rect(
            area.node.absolutePosition + vec2(area.node.size.x * 0.8, 0) + vec2(0, AreaHeaderHeight),
            vec2(area.node.size.x * 0.2, area.node.size.y - AreaHeaderHeight)
          )
          westRect = rect(
            area.node.absolutePosition + vec2(0, 0) + vec2(0, AreaHeaderHeight),
            vec2(area.node.size.x * 0.2, area.node.size.y - AreaHeaderHeight)
          )
        if mousePos.overlaps(headerRect):
          areaScan = Header
          rect = headerRect
        elif mousePos.overlaps(northRect):
          areaScan = North
          rect = northRect
        elif mousePos.overlaps(southRect):
          areaScan = South
          rect = southRect
        elif mousePos.overlaps(eastRect):
          areaScan = East
          rect = eastRect
        elif mousePos.overlaps(westRect):
          areaScan = West
          rect = westRect
        elif mousePos.overlaps(bodyRect):
          areaScan = Body
          rect = bodyRect
        targetArea = area
  visit(rootArea)
  return (targetArea, areaScan, rect)

proc visiblePanels*(area: Area): seq[Panel] =
  ## Returns the visible panels in the area and subareas.
  proc visit(area: Area, panels: var seq[Panel]) =
    if area.panels.len > 0:
      let selectedPanel = area.panels[area.selectedPanelNum]
      panels.add(selectedPanel)
    for subarea in area.areas:
      visit(subarea, panels)
  visit(area, result)

find "/UI/Main":
  onLoad:

    areaTemplate = find("Area").copy()
    areaTemplate.findAll("**/Panel").remove()
    areaTemplate.findAll("**/PanelHeader").remove()
    panelHeaderTemplate = find("**/PanelHeader").copy()
    panelTemplate = find("**/Panel").copy()

    objectInfoTemplate = find("../ObjectInfo").copy()

    find("Area").remove()

    rootArea = Area(node: areaTemplate.copy())
    rootArea.node.position = vec2(0, 64)
    rootArea.node.size = vec2(
      thisFrame.size.x,
      thisFrame.size.y - 64 * 5
    )
    thisNode.addChild(rootArea.node)

    dropHighlight = find("/UI/DropHighlight")
    dropHighlight.remove()
    dropHighlight.position = vec2(100, 100)
    dropHighlight.size = vec2(500, 500)
    dropHighlight.visible = false
    thisNode.addChild(dropHighlight)

  onResize:
    rootArea.node.size = vec2(
      thisFrame.size.x,
      thisFrame.size.y - 64 * 3
    )
    rootArea.refresh()

  find "**/Area":
    onMouseMove:
      if thisNode == hoverNodes[0]:
        let area = findAreaByNode(thisNode)
        if area != nil:
          if area.layout == Horizontal:
            thisCursor = Cursor(kind: ResizeUpDownCursor)
          else:
            thisCursor = Cursor(kind: ResizeLeftRightCursor)
    onDragStart:
      let area = findAreaByNode(thisNode)
      if area != nil and area.areas.len > 0:
        dragArea = area
        dropHighlight.visible = true
    onDrag:
      let mousePos = window.logicalMousePos
      if dragArea != nil:
        if dragArea.layout == Horizontal:
          dropHighlight.position = vec2(dragArea.node.absolutePosition.x, mousePos.y)
          dropHighlight.size = vec2(dragArea.node.size.x, AreaMargin)
          thisCursor = Cursor(kind: ResizeUpDownCursor)
        else:
          dropHighlight.position = vec2(mousePos.x, dragArea.node.absolutePosition.y)
          dropHighlight.size = vec2(AreaMargin, dragArea.node.size.y)
          thisCursor = Cursor(kind: ResizeLeftRightCursor)
    onDragEnd:
      let mousePos = window.logicalMousePos
      if dragArea != nil:
        if dragArea.layout == Horizontal:
          dragArea.split = (mousePos.y - dragArea.node.absolutePosition.y) / dragArea.node.size.y
        else:
          dragArea.split = (mousePos.x - dragArea.node.absolutePosition.x) / dragArea.node.size.x
        dragArea.refresh()
      dragArea = nil
      dropHighlight.visible = false

  find "**/PanelHeader":
    onClick:
      let panel = findPanelByHeader(thisNode)
      if panel != nil:
        panel.parentArea.selectedPanelNum = thisNode.childIndex
        panel.parentArea.refresh()

    onDragStart:
      dropHighlight.visible = true

    onDrag:
      let (_, _, rect) = rootArea.scan()
      dropHighlight.position = rect.xy
      dropHighlight.size = rect.wh

    onDragEnd:
      dropHighlight.visible = false
      let (targetArea, areaScan, _) = rootArea.scan()
      if targetArea != nil:
        let panel = findPanelByHeader(thisNode)
        if panel != nil:
          case areaScan:
            of Header:
              targetArea.movePanel(panel)
            of Body:
              targetArea.movePanel(panel)
            of North:
              targetArea.split(Horizontal)
              targetArea.areas[0].movePanel(panel)
              targetArea.areas[1].movePanels(targetArea.panels)
            of South:
              targetArea.split(Horizontal)
              targetArea.areas[1].movePanel(panel)
              targetArea.areas[0].movePanels(targetArea.panels)
            of East:
              targetArea.split(Vertical)
              targetArea.areas[1].movePanel(panel)
              targetArea.areas[0].movePanels(targetArea.panels)
            of West:
              targetArea.split(Vertical)
              targetArea.areas[0].movePanel(panel)
              targetArea.areas[1].movePanels(targetArea.panels)

        rootArea.removeBlankAreas()
        rootArea.refresh()
