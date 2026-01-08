import
  boxy, vmath, windy,
  common

proc boxyMouse*(window: Window): Vec2 =
  inverse(bxy.getTransform()) * window.mousePos.vec2
