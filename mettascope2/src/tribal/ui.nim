import
  boxy, vmath, windy,
  common, panels, utils, simulation

proc drawPanelBackground*(panel: Panel, bgColor: Color) =
  ## Draw a solid color background for a panel
  bxy.drawRect(
    rect = Rect(
      x: 0,
      y: 0,
      w: panel.rect.w.float32,
      h: panel.rect.h.float32
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
  ## Draw a speed button and handle clicking
  if drawIconButton("ui/speed", pos = vec2(x, 16), size = vec2(20, 32)):
    playSpeed = speed
    play = true
    echo "Speed: " & label
    return true
  return false

# ============== HEADER UI ==============


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

  # Draw the title.
  bxy.drawText(
    "Mettascope Arena Basic",
    translate(vec2(64+16, 16)),
    typeface,
    "Mettascope Arena Basic",
    24,
    color(1, 1, 1, 1)
  )

  if drawIconButton(
    "ui/share",
    pos = vec2(panel.rect.w.float32 - (16 + 32)*1, 16)
  ):
    echo "Share"

  if drawIconButton(
    "ui/help",
    pos = vec2(panel.rect.w.float32 - (16 + 32)*2, 16)
  ):
    echo "Help"

# ============== FOOTER UI ==============

const
  FooterBgColor = parseHtmlColor("#2D343D")

proc drawFooter*(panel: Panel) =
  drawPanelBackground(panel, FooterBgColor)

  # Draw the left side buttons.
  var x = 16f
  if drawIconButton(
    "ui/rewindToStart",
    pos = vec2(x, 16)
  ):
    echo "Rewind to start"
  x += 32 + 5

  if drawIconButton(
    "ui/stepBack",
    pos = vec2(x, 16)
  ):
    echo "Step back"
  x += 32 + 5

  if drawIconButton(
    if play: "ui/pause" else: "ui/play",
    pos = vec2(x, 16)
  ):
    play = not play
    echo if play: "Playing" else: "Paused"
  x += 32 + 5

  # if drawIconButton(
  #   "ui/pause",
  #   pos = vec2(x, 16)
  # ):
  #   echo "Pause"
  # x += 32 + 5

  if drawIconButton(
    "ui/stepForward",
    pos = vec2(x, 16)
  ):
    echo "Step forward"
    simStep()  # Step the simulation once
  x += 32 + 5

  if drawIconButton(
    "ui/rewindToEnd",
    pos = vec2(x, 16)
  ):
    echo "Rewind to end"


  # Draw the middle buttons.
  x = panel.rect.w.float32 / 2 - 32

  if drawIconButton(
    "ui/turtle",
    pos = vec2(x, 16)
  ):
    playSpeed = 0.5
    play = true
    echo "Speed: Slow (0.5x)"
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
    echo "Speed: Fast (16x)"


  # Draw the right side buttons.
  x = panel.rect.w.float32 - 16 - 32

  if drawIconToggle(
    "ui/cloud",
    pos = vec2(x, 16),
    value = settings.showFogOfWar
  ):
    echo "Fog of war"
  x -= 32 + 5

  if drawIconToggle(
    "ui/eye",
    pos = vec2(x, 16),
    value = settings.showVisualRange
  ):
    echo "Visual range"
  x -= 32 + 5

  if drawIconToggle(
    "ui/grid",
    pos = vec2(x, 16),
    value = settings.showGrid
  ):
    echo "Grid"
  x -= 32 + 5

  if drawIconToggle(
    "ui/tack",
    pos = vec2(x, 16),
    value = settings.lockFocus
  ):
    echo "Focus"
  x -= 32 + 5

const
  TimelineBgColor = parseHtmlColor("#1D1D1D")

proc drawTimeline*(panel: Panel) =
  drawPanelBackground(panel, TimelineBgColor)

  # Draw the scrubber bg.
  bxy.drawRect(
    rect = Rect(
      x: 16,
      y: 32,
      w: panel.rect.w.float32 - 32,
      h: 16
    ),
    color = parseHtmlColor("#717171")
  )

  var progress = 0.37

  # Draw the progress bar.
  bxy.drawRect(
    rect = Rect(
      x: 16,
      y: 32,
      w: (panel.rect.w.float32 - 32) * progress,
      h: 16
    ),
    color = color(1, 1, 1, 1)
  )
