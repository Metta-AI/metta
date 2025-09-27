import
  boxy, chroma, fidget2/hybridrender,
  common, panels

proc drawAgentTable*(panel: Panel) =

  let box = IRect(x: 0, y: 0, w: panel.rect.w, h: panel.rect.h)

  bxy.drawRect(
    rect = box.rect,
    color = color(0, 1, 1, 1.0)
  )
