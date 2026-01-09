import
  boxy, chroma, vmath, windy,
  common, worldmap, panels

proc drawMinimap*(zoomInfo: ZoomInfo) =
  ## Draw the minimap with automatic fitting to panel size.
  let box = irect(0, 0, zoomInfo.rect.w, zoomInfo.rect.h)

  bxy.drawRect(
    rect = box.rect,
    color = color(0, 0, 0, 1.0)
  )

  if replay.isNil or replay.mapSize == (0, 0):
    return

  bxy.saveTransform()

  # Calculate transform to fit entire world in minimap panel.
  let rectW = zoomInfo.rect.w.float32
  let rectH = zoomInfo.rect.h.float32
  if rectW <= 0 or rectH <= 0:
    bxy.restoreTransform()
    return

  let
    mapMinX = -0.5f
    mapMinY = -0.5f
    mapMaxX = replay.mapSize[0].float32 - 0.5f
    mapMaxY = replay.mapSize[1].float32 - 0.5f
    mapW = max(0.001f, mapMaxX - mapMinX)
    mapH = max(0.001f, mapMaxY - mapMinY)

  let zoomScale = min(rectW / mapW, rectH / mapH)
  let
    cx = (mapMinX + mapMaxX) / 2.0f
    cy = (mapMinY + mapMaxY) / 2.0f
    posX = rectW / 2.0f - cx * zoomScale
    posY = rectH / 2.0f - cy * zoomScale

  bxy.translate(vec2(posX, posY))
  bxy.scale(vec2(zoomScale, zoomScale))

  drawWorldMini()

  bxy.restoreTransform()
