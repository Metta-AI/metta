import
  boxy, windy, pixie,
  ../src/mettascope/[pixelator],
  opengl, boxy/shaders

let window = newWindow("Test Pixelator", ivec2(1280, 800))
makeContextCurrent(window)
loadExtensions()

let bxy = newBoxy()

var
  vel: Vec2
  pos: Vec2
  zoom: float32 = 1
  zoomVel: float32
  frame: int

let px = newPixelator("data/atlas.png", "data/atlas.json")

let spriteNames = @[
  "agents/agent.n",
  "agents/agent.s",
  "agents/agent.e",
  "agents/agent.w",
  "objects/chest",
  "objects/factory",
  "objects/water_well",
  "objects/oxygen_extractor",
  "objects/mine",
  "view/grid"
]

window.onFrame = proc() =
  bxy.beginFrame(window.size)

  glClearColor(0.1, 0.1, 0.1, 1.0)
  glClear(GL_COLOR_BUFFER_BIT)

  # Panning with middle mouse, inertial feel.
  if window.buttonDown[MouseMiddle] or window.buttonDown[MouseLeft]:
    vel = window.mouseDelta.vec2 + vel * 0.1
  else:
    vel *= 0.99

  pos += vel

  # Zoom with scroll, smooth approach.
  if window.scrollDelta.y != 0:
    zoomVel = window.scrollDelta.y * 0.005
  else:
    zoomVel *= 0.95

  var zoomOld = zoom
  zoom += zoomVel
  zoom = clamp(zoom, 0.025, 20.0)

  # Keep the point under the mouse anchored while zooming, using 2D transforms.
  let oldMat2D = translate(vec2(pos.x, pos.y)) * scale(vec2(zoomOld, zoomOld))
  let newMat2D = translate(vec2(pos.x, pos.y)) * scale(vec2(zoom, zoom))
  let newAt = newMat2D.inverse() * window.mousePos.vec2
  let oldAt = oldMat2D.inverse() * window.mousePos.vec2
  pos -= (oldAt - newAt).xy * zoom

  let projection = ortho(0.0f, window.size.x.float32, window.size.y.float32, 0.0f, -1.0f, 1.0f)
  let view = translate(vec3(pos.x, pos.y, 0.0f)) * scale(vec3(zoom, zoom, 1.0f))
  let mvp = projection * view

  # Queue a small grid of sprites to exercise the atlas.
  px.clear()
  let cols = 100
  let rows = 100
  let pitch = 96
  for y in 0 ..< rows:
    for x in 0 ..< cols:
      let name = spriteNames[(x + y) mod spriteNames.len]
      let spriteX = (x * pitch + 48).uint16
      let spriteY = (y * pitch + 48).uint16
      px.drawSprite(name, spriteX, spriteY)

  px.flush(mvp)

  bxy.endFrame()
  window.swapBuffers()
  inc frame

while not window.closeRequested:
  pollEvents()
