import
  boxy, vmath, windy, fidget2/hybridrender,
  utils


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
