import
  boxy, vmath, windy, fidget2/hybridrender

var
  typeface*: Typeface

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
  size: float32,
  typeface: Typeface
): Vec2 =
  var font = newFont(typeface)
  font.size = size
  let arrangement = typeset(@[newSpan(text, font)], bounds = vec2(1280, 800))
  let transform = translate(vec2(0, 0))
  let bounds = arrangement.computeBounds(transform).snapToPixels()
  return vec2(bounds.w, bounds.h)

proc drawBubbleLine*(bxy: Boxy, start: Vec2, stop: Vec2, color: Color) =
  ## Draw a line with circles.
  let
    dir = (stop - start).normalize
  for i in 0 ..< int(dist(start, stop) / 5):
    let pos = start + dir * i.float32 * 5
    # bxy.drawImage(
    #   "bubble",
    #   pos,
    #   angle = 0,
    #   scale = 0.25,
    #   tint = color
    # )

proc newSeq2D*[T](width: int, height: int): seq[seq[T]] =
  result = newSeq[seq[T]](width)
  for i in 0 ..< width:
    result[i] = newSeq[T](height)
