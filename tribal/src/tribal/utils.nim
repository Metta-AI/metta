import
  boxy, vmath, windy,
  common
import std/[times, strformat]


var
  typeface* = readTypeface("data/fonts/Inter-Regular.ttf")

proc drawText*(
  bxy: Boxy,
  imageKey: string,
  transform: Mat3,
  typeface: Typeface,
  text: string,
  size: float32,
  color: Color
) =
  ## Draw text on the screen.
  var font = newFont(typeface)
  font.size = size
  font.paint = color
  let
    arrangement = typeset(@[newSpan(text, font)], bounds = vec2(1280, 800))
    globalBounds = arrangement.computeBounds(transform).snapToPixels()
    textImage = newImage(globalBounds.w.int, globalBounds.h.int)
    imageSpace = translate(-globalBounds.xy) * transform
  textImage.fillText(arrangement, imageSpace)

  bxy.addImage(imageKey, textImage)
  bxy.drawImage(imageKey, globalBounds.xy)

proc measureText*(
  text: string,
  size: float32
): Vec2 =
  var font = newFont(typeface)
  font.size = size
  let arrangement = typeset(@[newSpan(text, font)], bounds = vec2(1280, 800))
  let transform = translate(vec2(0, 0))
  let bounds = arrangement.computeBounds(transform).snapToPixels()
  return vec2(bounds.w, bounds.h)



proc boxyMouse*(window: Window): Vec2 =
  inverse(bxy.getTransform()) * window.mousePos.vec2

var
  frameTimer*: float64 = 0.0
  lastFrameTime*: float64 = 0.0
  frameCount*: int = 0
  avgFrameTime*: float64 = 0.0

proc startFrameTiming*() =
  ## Start timing a frame for performance monitoring
  frameTimer = epochTime()

proc endFrameTiming*() =
  ## End frame timing and update performance stats
  lastFrameTime = epochTime() - frameTimer
  frameCount += 1

  # Rolling average over last 60 frames
  if frameCount >= 60:
    avgFrameTime = avgFrameTime * 0.95 + lastFrameTime * 0.05
  else:
    avgFrameTime = (avgFrameTime * (frameCount - 1).float + lastFrameTime) / frameCount.float

proc getFPS*(): float64 =
  ## Get current FPS based on average frame time
  if avgFrameTime > 0: 1.0 / avgFrameTime else: 0.0

proc getFrameTimeMS*(): float64 =
  ## Get current frame time in milliseconds
  avgFrameTime * 1000.0

proc debugLog*(msg: string) =
  ## Debug logging function (only outputs if debug mode enabled)
  if settings.enableLogging and settings.debugMode:
    let timestamp = fmt"{epochTime():.3f}"
    echo fmt"[{timestamp}] DEBUG: {msg}"

proc performanceLog*(operation: string, duration: float64) =
  ## Performance logging for operations
  if settings.enableLogging and settings.showPerformanceStats:
    let timestamp = fmt"{epochTime():.3f}"
    echo fmt"[{timestamp}] PERF: {operation} took {duration * 1000.0:.2f}ms"