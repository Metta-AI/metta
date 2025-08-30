import
  boxy, chroma,
  common, panels

proc drawMinimap*(panel: Panel) =

  let box = IRect(x: 0, y: 0, w: panel.rect.w, h: panel.rect.h)

  bxy.drawRect(
    rect = box.rect,
    color = color(1, 0, 0, 1.0)
  )
