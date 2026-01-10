import
  std/[math],
  boxy, windy, opengl, vmath,
  mettascope/[heatmap, heatmapshader, replays, utils, common]

# run from the mettascope directory:
# nim r tests/manual_heatmap.nim

let window = newWindow("Test Heatmap Shader", ivec2(800, 800))
makeContextCurrent(window)
loadExtensions()

let bxy = newBoxy()

let debugTypeface = readTypeface("./data/fonts/Inter-Regular.ttf")

var
  pos: Vec2
  zoom: float32 = 20.0
  frame: int
  currentStep: int = 0
  showDebug: bool = false
  patternMode: bool = false

proc centerView() =
  ## Center the view on the map.
  let mapWidth = 32.0 * zoom
  let mapHeight = 32.0 * zoom
  pos.x = (800.0 - mapWidth) / 2.0
  pos.y = (800.0 - mapHeight) / 2.0

proc createTestReplay(): Replay =
  ## Create a mock replay with some agents moving around.
  result = Replay()
  result.version = 3
  result.mapSize = (31, 33)
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

    let cx = 15.0
    let cy = 16.0
    let size = 8.0 + float(i) * 2.0
    let perimeter = size * 8.0  # Each side is size*2 long, 4 sides total
    let speed = 1.0  # Units per step

    for step in 0 ..< result.maxSteps:
      # Calculate position along perimeter (constant speed for all agents)
      let position = (float(step) * speed) mod perimeter

      # Determine which side of the square we're on
      let sideLength = size * 2.0  # Each side is 2*size long
      let side = int(position / sideLength) mod 4
      let sidePos = position mod sideLength

      var x, y: float
      case side:
      of 0:  # Top edge: left to right
        x = cx - size + sidePos
        y = cy - size
      of 1:  # Right edge: top to bottom
        x = cx + size
        y = cy - size + sidePos
      of 2:  # Bottom edge: right to left
        x = cx + size - sidePos
        y = cy + size
      else:  # Left edge: bottom to top
        x = cx - size
        y = cy + size - sidePos

      agent.location[step] = ivec2(
        int32(x),
        int32(y)
      )

    result.objects.add(agent)
    result.agents.add(agent)

let testReplay = createTestReplay()

var testHeatmap = newHeatmap(testReplay)
testHeatmap.initialize(testReplay)

proc generateCoordinatePattern(heatmap: var Heatmap) =
  ## Generate heatmap data based on coordinates for verification.
  ## Uses abs(x) + abs(y) pattern to create a diagonal gradient.
  let maxCoordValue = (heatmap.width - 1) + (heatmap.height - 1)

  for step in 0 ..< heatmap.maxSteps:
    for y in 0 ..< heatmap.height:
      for x in 0 ..< heatmap.width:
        let value = abs(x) + abs(y)
        heatmap.data[step][y * heatmap.width + x] = value

    # Update max heat for this step to the actual max coordinate value
    heatmap.maxHeat[step] = maxCoordValue

initHeatmapShader()
updateTexture(testHeatmap, 0)

centerView()

echo "Heatmap test ready."
echo "  Scroll to zoom, drag to pan."
echo "  Left/Right arrows to change step."
echo "  Space to animate, R to reset view."
echo "  D to toggle debug overlay."
echo "  P to toggle coordinate pattern mode."
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

  if animating:
    currentStep = (currentStep + 1) mod testReplay.maxSteps

  # Build MVP matrix.
  let projection = ortho(0.0f, window.size.x.float32, window.size.y.float32, 0.0f, -1.0f, 1.0f)
  let view = translate(vec3(pos.x, pos.y, 0.0f)) * scale(vec3(zoom, zoom, 1.0f))
  let mvp = projection * view

  # Update and draw heatmap.
  updateTexture(testHeatmap, currentStep)

  bxy.enterRawOpenGLMode()

  let maxHeat = testHeatmap.getMaxHeat(currentStep).float32
  let mapSize = vec2(testReplay.mapSize[0].float32, testReplay.mapSize[1].float32)

  if maxHeat > 0:
    draw(testHeatmap, mvp, mapSize, maxHeat)

  if showDebug:
    bxy.exitRawOpenGLMode()

    # Show coordinates when zoomed in enough (individual cells are visible)
    let showCoords = zoom > 15.0

    # Calculate visible area in world coordinates to optimize rendering
    let viewMatrix = translate(vec3(pos.x, pos.y, 0.0f)) * scale(vec3(zoom, zoom, 1.0f))
    let invView = viewMatrix.inverse()

    # Transform screen corners to world coordinates to find visible area
    let topLeftScreen = vec2(0, 0)
    let bottomRightScreen = vec2(window.size.x.float32, window.size.y.float32)
    let topLeftWorld = (invView * vec4(topLeftScreen.x, topLeftScreen.y, 0.0, 1.0)).xy
    let bottomRightWorld = (invView * vec4(bottomRightScreen.x, bottomRightScreen.y, 0.0, 1.0)).xy

    # Convert to grid bounds with margin
    let margin = 2.0
    let minX = max(0, int(floor(topLeftWorld.x - margin)))
    let maxX = min(testReplay.mapSize[0] - 1, int(ceil(bottomRightWorld.x + margin)))
    let minY = max(0, int(floor(topLeftWorld.y - margin)))
    let maxY = min(testReplay.mapSize[1] - 1, int(ceil(bottomRightWorld.y + margin)))

    # Draw heat values and optionally coordinates on visible cells only
    for y in minY .. maxY:
      for x in minX .. maxX:
        let heatValue = testHeatmap.getHeat(currentStep, x, y)
        # Cell center in world coordinates
        let worldPos = vec2(x.float32 + 0.5, y.float32 + 0.5)
        # Transform to screen coordinates using view matrix only
        let screenPos = (viewMatrix * vec4(worldPos.x, worldPos.y, 0.0, 1.0)).xy

        # Draw heat value
        if heatValue > 0:
          let heatText = $heatValue
          drawText(bxy, "debug_heat_" & $x & "_" & $y, translate(screenPos - vec2(0, 6)), debugTypeface, heatText, 12.0, color(1, 1, 1, 0.9))

        # Draw coordinates when zoomed in
        if showCoords:
          let coordText = "(" & $x & "," & $y & ")"
          drawText(bxy, "debug_coord_" & $x & "_" & $y, translate(screenPos + vec2(0, 6)), debugTypeface, coordText, 10.0, color(1, 0.8, 0.8, 0.7))

    bxy.enterRawOpenGLMode()

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
  # Grid lines should align with heatmap cell boundaries
  # Heatmap spans from -0.5 to mapSize - 0.5, so grid lines are at half-integers
  for i in 0 .. testReplay.mapSize[0]:
    let x = i.float32 - 0.5
    glVertex2f(x, -0.5)
    glVertex2f(x, testReplay.mapSize[1].float32 - 0.5)
  for i in 0 .. testReplay.mapSize[1]:
    let y = i.float32 - 0.5
    glVertex2f(-0.5, y)
    glVertex2f(testReplay.mapSize[0].float32 - 0.5, y)
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
  elif button == KeyD:
    showDebug = not showDebug
    echo "Debug overlay: ", (if showDebug: "ON" else: "OFF")
  elif button == KeyP:
    patternMode = not patternMode
    echo "Pattern mode: ", (if patternMode: "ON" else: "OFF")
    # Regenerate heatmap with pattern or restore from replay
    if patternMode:
      generateCoordinatePattern(testHeatmap)
    else:
      testHeatmap.initialize(testReplay)
    # Force texture update by resetting current step
    testShader.currentStep = -1
    testShader.updateTexture(testHeatmap, currentStep)

while not window.closeRequested:
  pollEvents()

