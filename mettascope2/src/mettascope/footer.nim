import
  std/[strformat],
  boxy, vmath, windy,
  common, panels, tribal, actions, utils, ui

const
  BgColor = parseHtmlColor("#2D343D")

proc drawFooter*(panel: Panel) =
  bxy.drawRect(
    rect = Rect(
      x: 0,
      y: 0,
      w: panel.rect.w.float32,
      h: panel.rect.h.float32
    ),
    color = BgColor
  )

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

  if drawIconButton(
    "ui/speed",
    pos = vec2(x, 16),
    size = vec2(20, 32)
  ):
    playSpeed = 0.25
    play = true
    echo "Speed: 1x"
  x += 20

  if drawIconButton(
    "ui/speed",
    pos = vec2(x, 16),
    size = vec2(20, 32)
  ):
    playSpeed = 0.125
    play = true
    echo "Speed: 2x"
  x += 20

  if drawIconButton(
    "ui/speed",
    pos = vec2(x, 16),
    size = vec2(20, 32)
  ):
    playSpeed = 0.0625
    play = true
    echo "Speed: 4x"
  x += 20

  if drawIconButton(
    "ui/speed",
    pos = vec2(x, 16),
    size = vec2(20, 32)
  ):
    playSpeed = 0.03125
    play = true
    echo "Speed: 8x"
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
