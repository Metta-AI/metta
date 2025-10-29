import
  boxy, vmath, windy, chroma,
  common

proc drawPanelBackground*(panel: Panel, bgColor: Color) =
  let panelRect = panel.rect.rect
  bxy.drawRect(
    rect = Rect(
      x: 0,
      y: 0,
      w: panelRect.w,
      h: panelRect.h
    ),
    color = bgColor
  )

proc drawIconButton*(
  image: string,
  pos: Vec2,
  size = vec2(32, 32),
): bool =
  let box = Rect(
    x: pos.x,
    y: pos.y,
    w: size.x,
    h: size.y
  )

  # Use current transform to convert mouse to local panel coords
  let mouseLocal = bxy.getTransform().inverse * window.mousePos.vec2
  if mouseLocal.overlaps(box):
    if window.buttonPressed[MouseLeft]:
      result = true
    bxy.drawRect(
      rect = box,
      color = color(1, 1, 1, 0.5)
    )

  bxy.drawImage(
    image,
    pos = pos
  )

 

proc drawSpeedButton*(x: float32, speed: float32, label: string): bool =
  if drawIconButton("ui/speed", pos = vec2(x, 16), size = vec2(20, 32)):
    playSpeed = speed
    play = true
    return true
  return false



 
proc drawFooter*(panel: Panel) =
  drawPanelBackground(panel, parseHtmlColor("#2D343D"))
  # Minimal controls: turtle (slow), play/pause, rabbit (fast)
  var x = (panel.rect.rect.w.float32 - (32f * 3 + 10f * 2)) / 2

  if drawIconButton(
    "ui/turtle",
    pos = vec2(x, 16)
  ):
    playSpeed = 0.5
    play = true
  x += 32 + 10

  if drawIconButton(
    if play: "ui/pause" else: "ui/play",
    pos = vec2(x, 16)
  ):
    if play:
      play = false
    else:
      playSpeed = DefaultPlaySpeed
      play = true
  x += 32 + 10

  if drawIconButton(
    "ui/rabbit",
    pos = vec2(x, 16)
  ):
    playSpeed = 0.015625
    play = true


 

proc drawResourceBar*(pos: Vec2, size: Vec2, current: int, maximum: int, bgColor: Color = color(0.2, 0.2, 0.2, 1), fillColor: Color = color(0.8, 0.8, 0.8, 1)) =
  ## Draw a resource bar
  let progress = if maximum > 0: current.float / maximum.float else: 0.0
  
  # Background
  bxy.drawRect(
    rect = Rect(x: pos.x, y: pos.y, w: size.x, h: size.y),
    color = bgColor
  )
  
  # Fill
  if progress > 0:
    bxy.drawRect(
      rect = Rect(x: pos.x, y: pos.y, w: size.x * progress, h: size.y),
      color = fillColor
    )

proc drawStatusIcon*(icon: string, pos: Vec2, active: bool = true) =
  ## Draw a status icon with optional dimming for inactive state
  let alpha = if active: 1.0 else: 0.3
  bxy.drawImage(
    icon,
    pos = pos,
    tint = color(1, 1, 1, alpha)
  )

proc drawInfoTooltip*(text: string, pos: Vec2, bgColor: Color = color(0, 0, 0, 0.8)) =
  ## Draw an information tooltip
  let textWidth = text.len.float * 8.0  # Approximate text width
  let textHeight = 16.0  # Approximate text height
  let padding = 8.0
  
  # Background
  bxy.drawRect(
    rect = Rect(
      x: pos.x - padding,
      y: pos.y - padding,
      w: textWidth + padding * 2,
      h: textHeight + padding * 2
    ),
    color = bgColor
  )
  
  # Text omitted in simplified UI to avoid extra font deps
  discard
