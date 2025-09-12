import
  boxy, vmath, windy, chroma,
  common, utils, controls

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

  if window.boxyMouse.vec2.overlaps(box):
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

proc drawIconToggle*(
  image: string,
  pos: Vec2,
  size = vec2(32, 32),
  value: var bool
): bool =
  let box = Rect(
    x: pos.x,
    y: pos.y,
    w: size.x,
    h: size.y
  )

  if window.boxyMouse.vec2.overlaps(box):
    if window.buttonPressed[MouseLeft]:
      value = not value
      result = true
    bxy.drawRect(
      rect = box,
      color = color(1, 1, 1, 0.5)
    )

  var alpha = 0.4
  if value:
    alpha = 1.0

  bxy.drawImage(
    image,
    pos = pos,
    tint = color(1, 1, 1, alpha)
  )

proc drawSpeedButton*(x: float32, speed: float32, label: string): bool =
  if drawIconButton("ui/speed", pos = vec2(x, 16), size = vec2(20, 32)):
    playSpeed = speed
    play = true
    return true
  return false



const
  HeaderBgColor = parseHtmlColor("#273646")

proc drawHeader*(panel: Panel) =
  drawPanelBackground(panel, HeaderBgColor)
  bxy.drawImage(
    "ui/logo",
    pos = vec2(0, 0),
  )
  bxy.drawImage(
    "ui/header-bg",
    pos = vec2(0, 0),
  )

  bxy.drawText(
    "Mettascope Arena Basic",
    translate(vec2(64+16, 16)),
    typeface,
    "Mettascope Arena Basic",
    24,
    color(1, 1, 1, 1)
  )

  discard drawIconButton(
    "ui/share",
    pos = vec2(panel.rect.rect.w - (16 + 32)*1, 16)
  )
  
  discard drawIconButton(
    "ui/help",
    pos = vec2(panel.rect.rect.w - (16 + 32)*2, 16)
  )


proc drawFooter*(panel: Panel) =
  drawPanelBackground(panel, parseHtmlColor("#2D343D"))
  
  var x = 16f
  discard drawIconButton(
    "ui/rewindToStart",
    pos = vec2(x, 16)
  )
  x += 32 + 5

  discard drawIconButton(
    "ui/stepBack",
    pos = vec2(x, 16)
  )
  x += 32 + 5

  if drawIconButton(
    if play: "ui/pause" else: "ui/play",
    pos = vec2(x, 16)
  ):
    play = not play
    echo if play: "Playing" else: "Paused"
  x += 32 + 5


  if drawIconButton(
    "ui/stepForward",
    pos = vec2(x, 16)
  ):
    simStep()  # Step the simulation once
  x += 32 + 5

  discard drawIconButton(
    "ui/rewindToEnd",
    pos = vec2(x, 16)
  )


  x = panel.rect.rect.w / 2 - 32

  if drawIconButton(
    "ui/turtle",
    pos = vec2(x, 16)
  ):
    playSpeed = 0.5
    play = true
  x += 32 + 3

  discard drawSpeedButton(x, 0.25, "1x")
  x += 20

  discard drawSpeedButton(x, 0.125, "2x")
  x += 20

  discard drawSpeedButton(x, 0.0625, "4x")
  x += 20

  discard drawSpeedButton(x, 0.03125, "8x")
  x += 20

  if drawIconButton(
    "ui/rabbit",
    pos = vec2(x, 16)
  ):
    playSpeed = 0.015625
    play = true


  x = panel.rect.rect.w - 16 - 32

  discard drawIconToggle(
    "ui/cloud",
    pos = vec2(x, 16),
    value = settings.showFogOfWar
  )
  x -= 32 + 5

  discard drawIconToggle(
    "ui/eye",
    pos = vec2(x, 16),
    value = settings.showVisualRange
  )
  x -= 32 + 5

  discard drawIconToggle(
    "ui/grid",
    pos = vec2(x, 16),
    value = settings.showGrid
  )
  x -= 32 + 5

  discard drawIconToggle(
    "ui/tack",
    pos = vec2(x, 16),
    value = settings.lockFocus
  )
  x -= 32 + 5


proc drawTimeline*(panel: Panel) =
  drawPanelBackground(panel, parseHtmlColor("#1D1D1D"))
  
  bxy.drawRect(
    rect = Rect(
      x: 16,
      y: 32,
      w: panel.rect.rect.w - 32,
      h: 16
    ),
    color = parseHtmlColor("#717171")
  )

  var progress = 0.37

  bxy.drawRect(
    rect = Rect(
      x: 16,
      y: 32,
      w: (panel.rect.rect.w - 32) * progress,
      h: 16
    ),
    color = color(1, 1, 1, 1)
  )

proc drawResourceBar*(pos: Vec2, size: Vec2, current: int, maximum: int, bgColor: Color = color(0.2, 0.2, 0.2, 1), fillColor: Color = color(0.8, 0.8, 0.8, 1)) =
  ## Draw a resource bar (health, hunger, etc.)
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
  
  # Text
  bxy.drawText(
    "",  # image key
    translate(pos) * scale(vec2(16, 16)),  # transform  
    typeface,
    text,
    16,  # size
    color(1, 1, 1, 1)
  )
