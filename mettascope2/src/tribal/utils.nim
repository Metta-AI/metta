import
  boxy, vmath, windy,
  common


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
