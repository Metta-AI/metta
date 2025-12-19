import
  std/[math],
  boxy, windy, opengl, vmath,
  ../src/mettascope/[heatmap, heatmapshader, replays]

let window = newWindow("Test Heatmap Shader", ivec2(800, 800))
makeContextCurrent(window)
loadExtensions()

let bxy = newBoxy()

var
  pos: Vec2
  zoom: float32 = 20.0
  frame: int
  currentStep: int = 0

proc centerView() =
  ## Center the view on the map.
  let mapWidth = 32.0 * zoom
  let mapHeight = 32.0 * zoom
  pos.x = (800.0 - mapWidth) / 2.0
  pos.y = (800.0 - mapHeight) / 2.0

# Create a mock replay with some agents moving around.
proc createTestReplay(): Replay =
  result = Replay()
  result.version = 3
  result.mapSize = (32, 32)
  result.maxSteps = 100
  result.numAgents = 4

  # Create 4 agents with different movement patterns.
  for i in 0 ..< 4:
    var agent = Entity()
    agent.id = i
    agent.agentId = i
    agent.isAgent = true
    agent.typeName = "agent"
    agent.location = newSeq[IVec2](result.maxSteps)

    for step in 0 ..< result.maxSteps:
      let angle = float(step) * 0.1 + float(i) * PI / 2.0
      let radius = 8.0 + float(i) * 2.0
      let cx = 16.0
      let cy = 16.0
      agent.location[step] = ivec2(
        int32(cx + cos(angle) * radius),
        int32(cy + sin(angle) * radius)
      )

    result.objects.add(agent)
    result.agents.add(agent)

let testReplay = createTestReplay()

# Create heatmap and shader.
var testHeatmap = newHeatmap(testReplay)
testHeatmap.initialize(testReplay)

var testShader = newHeatmapShader()
testShader.updateTexture(testHeatmap, 0)

centerView()

echo "Heatmap test ready."
echo "  Scroll to zoom, drag to pan."
echo "  Left/Right arrows to change step."
echo "  Space to animate, R to reset view."
echo "Map size: ", testReplay.mapSize[0], "x", testReplay.mapSize[1]
echo "Max steps: ", testReplay.maxSteps

var animating = false

window.onFrame = proc() =
  bxy.beginFrame(window.size)

  glClearColor(0.15, 0.15, 0.2, 1.0)
  glClear(GL_COLOR_BUFFER_BIT)

  # Panning with middle mouse or left mouse.
  if window.buttonDown[MouseMiddle] or window.buttonDown[MouseLeft]:
    pos += window.mouseDelta.vec2

  # Zoom with scroll, centered on mouse position.
  if window.scrollDelta.y != 0:
    let zoomOld = zoom
    let zoomFactor = 1.0 + window.scrollDelta.y * 0.05
    zoom = clamp(zoom * zoomFactor, 2.0, 100.0)

    # Keep the point under the mouse anchored while zooming.
    let mouseWorld = (window.mousePos.vec2 - pos) / zoomOld
    pos = window.mousePos.vec2 - mouseWorld * zoom

  # Animate if enabled.
  if animating:
    currentStep = (currentStep + 1) mod testReplay.maxSteps

  # Build MVP matrix.
  let projection = ortho(0.0f, window.size.x.float32, window.size.y.float32, 0.0f, -1.0f, 1.0f)
  let view = translate(vec3(pos.x, pos.y, 0.0f)) * scale(vec3(zoom, zoom, 1.0f))
  let mvp = projection * view

  # Update and draw heatmap.
  testShader.updateTexture(testHeatmap, currentStep)

  bxy.enterRawOpenGLMode()

  let maxHeat = testHeatmap.getMaxHeat(currentStep).float32
  let mapSize = vec2(testReplay.mapSize[0].float32, testReplay.mapSize[1].float32)

  if maxHeat > 0:
    testShader.draw(mvp, mapSize, maxHeat)

  # Draw grid lines for reference.
  glUseProgram(0)
  glMatrixMode(GL_PROJECTION)
  glLoadIdentity()
  glOrtho(0, window.size.x.float64, window.size.y.float64, 0, -1, 1)
  glMatrixMode(GL_MODELVIEW)
  glLoadIdentity()
  glTranslatef(pos.x, pos.y, 0)
  glScalef(zoom, zoom, 1)

  glLineWidth(1.0)
  glColor4f(0.3, 0.3, 0.3, 0.5)
  glBegin(GL_LINES)
  for i in 0 .. testReplay.mapSize[0]:
    glVertex2f(i.float32, 0)
    glVertex2f(i.float32, testReplay.mapSize[1].float32)
  for i in 0 .. testReplay.mapSize[1]:
    glVertex2f(0, i.float32)
    glVertex2f(testReplay.mapSize[0].float32, i.float32)
  glEnd()

  bxy.exitRawOpenGLMode()

  bxy.endFrame()
  window.swapBuffers()
  inc frame

window.onButtonPress = proc(button: Button) =
  if button == KeyRight:
    currentStep = min(currentStep + 1, testReplay.maxSteps - 1)
    echo "Step: ", currentStep, " MaxHeat: ", testHeatmap.getMaxHeat(currentStep)
  elif button == KeyLeft:
    currentStep = max(currentStep - 1, 0)
    echo "Step: ", currentStep, " MaxHeat: ", testHeatmap.getMaxHeat(currentStep)
  elif button == KeySpace:
    animating = not animating
    echo "Animating: ", animating
  elif button == KeyR:
    zoom = 20.0
    centerView()
    echo "View reset"

while not window.closeRequested:
  pollEvents()

