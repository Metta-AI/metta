import boxy, vmath, windy, chroma, pixie

# Settings for UI controls
type
  Settings* = object
    showFogOfWar*: bool
    showVisualRange*: bool
    showGrid*: bool
    lockFocus*: bool
    showObservations*: int

var 
  settings* = Settings(
    showFogOfWar: false,
    showVisualRange: false,
    showGrid: false,
    lockFocus: false,
    showObservations: -1
  )

# Playback control variables
var
  play* = false
  playSpeed* = 0.125  # Default 2x speed
  lastSimTime* = 0.0

proc boxyMouse*(window: Window, bxy: Boxy): Vec2 =
  return inverse(bxy.getTransform()) * window.mousePos.vec2

proc drawIconButton*(
  bxy: Boxy,
  window: Window,
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

  if window.boxyMouse(bxy).overlaps(box):
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
  bxy: Boxy,
  window: Window,
  image: string,
  pos: Vec2,
  value: var bool,
  size = vec2(32, 32),
): bool =
  let box = Rect(
    x: pos.x,
    y: pos.y,
    w: size.x,
    h: size.y
  )

  if window.boxyMouse(bxy).overlaps(box):
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

const
  HeaderBgColor* = parseHtmlColor("#273646")
  FooterBgColor* = parseHtmlColor("#2D343D")
  HeaderHeight* = 64
  FooterHeight* = 64

proc drawHeader*(bxy: Boxy, window: Window, typeface: Typeface, width: float32) =
  # Draw background
  bxy.drawRect(
    rect = Rect(x: 0, y: 0, w: width, h: HeaderHeight),
    color = HeaderBgColor
  )
  
  # Draw logo if available
  try:
    bxy.drawImage("ui/logo", pos = vec2(0, 0))
  except:
    discard
    
  try:
    bxy.drawImage("ui/header-bg", pos = vec2(0, 0))
  except:
    discard

  # Draw title
  var font = newFont(typeface)
  font.size = 24
  font.paint = color(1, 1, 1, 1)
  let titleText = "Tribal Grid"
  let arrangement = typeset(@[newSpan(titleText, font)], bounds = vec2(400, 100))
  let transform = translate(vec2(80, 20))
  let bounds = arrangement.computeBounds(transform).snapToPixels()
  let textImage = newImage(bounds.w.int, bounds.h.int)
  let imageSpace = translate(-bounds.xy) * transform
  textImage.fillText(arrangement, imageSpace)
  
  bxy.addImage("title_text", textImage)
  bxy.drawImage("title_text", bounds.xy)

  # Draw help button
  if drawIconButton(bxy, window, "ui/help", vec2(width - (16 + 32)*2, 16)):
    echo "Help"
    
  # Draw share button  
  if drawIconButton(bxy, window, "ui/share", vec2(width - (16 + 32)*1, 16)):
    echo "Share"

proc drawFooter*(bxy: Boxy, window: Window, width: float32, simStep: proc()) =
  # Draw background
  bxy.drawRect(
    rect = Rect(x: 0, y: 0, w: width, h: FooterHeight),
    color = FooterBgColor
  )

  # Left side - playback controls
  var x = 16'f32
  
  if drawIconButton(bxy, window, "ui/rewindToStart", vec2(x, 16)):
    echo "Rewind to start"
  x += 37

  if drawIconButton(bxy, window, "ui/stepBack", vec2(x, 16)):
    echo "Step back"
  x += 37

  if drawIconButton(bxy, window,
    if play: "ui/pause" else: "ui/play",
    vec2(x, 16)
  ):
    play = not play
    echo if play: "Playing" else: "Paused"
  x += 37

  if drawIconButton(bxy, window, "ui/stepForward", vec2(x, 16)):
    echo "Step forward"
    simStep()
  x += 37

  if drawIconButton(bxy, window, "ui/rewindToEnd", vec2(x, 16)):
    echo "Rewind to end"

  # Middle - speed controls
  x = width / 2 - 80
  
  if drawIconButton(bxy, window, "ui/turtle", vec2(x, 16)):
    playSpeed = 0.5
    play = true
    echo "Speed: Slow (0.5x)"
  x += 35
  
  # Speed bars (1x, 2x, 4x, 8x)
  if drawIconButton(bxy, window, "ui/speed", vec2(x, 16), size = vec2(20, 32)):
    playSpeed = 0.25
    play = true
    echo "Speed: 1x"
  x += 20
  
  if drawIconButton(bxy, window, "ui/speed", vec2(x, 16), size = vec2(20, 32)):
    playSpeed = 0.125
    play = true
    echo "Speed: 2x"
  x += 20
  
  if drawIconButton(bxy, window, "ui/speed", vec2(x, 16), size = vec2(20, 32)):
    playSpeed = 0.0625
    play = true
    echo "Speed: 4x"
  x += 20
  
  if drawIconButton(bxy, window, "ui/speed", vec2(x, 16), size = vec2(20, 32)):
    playSpeed = 0.03125
    play = true
    echo "Speed: 8x"
  x += 20
  
  if drawIconButton(bxy, window, "ui/rabbit", vec2(x, 16)):
    playSpeed = 0.015625
    play = true
    echo "Speed: Fast (16x)"

  # Right side - view toggles
  x = width - 16 - 32
  
  if drawIconToggle(bxy, window, "ui/cloud", vec2(x, 16), settings.showFogOfWar):
    echo "Fog of war: ", settings.showFogOfWar
  x -= 37

  if drawIconToggle(bxy, window, "ui/eye", vec2(x, 16), settings.showVisualRange):
    echo "Visual range: ", settings.showVisualRange
  x -= 37

  if drawIconToggle(bxy, window, "ui/grid", vec2(x, 16), settings.showGrid):
    echo "Grid: ", settings.showGrid
  x -= 37

  if drawIconToggle(bxy, window, "ui/tack", vec2(x, 16), settings.lockFocus):
    echo "Lock focus: ", settings.lockFocus