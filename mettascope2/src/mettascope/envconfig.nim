import
  boxy, chroma, fidget2/[common, hybridrender],
  common, panels

proc drawEnvConfig*(panel: Panel) =

  let box = IRect(x: 0, y: 0, w: panel.rect.w, h: panel.rect.h)

  bxy.drawRect(
    rect = box.rect,
    color = color(0, 1, 0, 1.0)
  )
