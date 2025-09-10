## Tribal Process-Separated Viewer
## Standalone Nim executable that runs the environment and viewer
## Communicates with Python through file-based IPC

import std/[os, times, strutils, json, strformat, random]
import boxy, opengl, windy, vmath
import tribal/[environment, renderer, common, panels, controls, ui, external_actions]

# Communication files
const
  ActionsFile = "tribal_actions.json"
  StateFile = "tribal_state.json"
  ControlFile = "tribal_control.json"

# Global state
var
  window: Window
  bxy: Boxy
  rootArea: Area
  worldMapPanel: Panel
  minimapPanel: Panel  
  agentTablePanel: Panel
  agentTracesPanel: Panel
  globalTimelinePanel: Panel
  globalFooterPanel: Panel
  globalHeaderPanel: Panel
  
  env: Environment
  lastUpdateTime = 0.0
  communicationActive = false
  assetsLoaded = false  # Track whether we have PNG assets
  testMode = false  # Standalone test mode without Python communication

# Forward declaration for safe rendering
proc drawSafeRectangleEnvironment()

proc initViewer(): bool =
  ## Initialize the tribal viewer with proper OpenGL context
  try:
    echo "🎮 Initializing Tribal Process Viewer with graphics"
    
    # Create window with careful error handling
    try:
      window = newWindow("Tribal - Process Separated", ivec2(1280, 800))
      echo "✅ Window created successfully"
    except Exception as e:
      echo "❌ Failed to create window: ", e.msg
      return false
    
    try:
      makeContextCurrent(window)
      echo "✅ OpenGL context made current"
    except Exception as e:
      echo "❌ Failed to make context current: ", e.msg
      return false
    
    when not defined(emscripten):
      try:
        loadExtensions()
        echo "✅ OpenGL extensions loaded"
      except Exception as e:
        echo "❌ Failed to load OpenGL extensions: ", e.msg
        return false
    
    # Initialize boxy with error handling
    try:
      bxy = newBoxy()
      echo "✅ Boxy initialized"
    except Exception as e:
      echo "❌ Failed to initialize boxy: ", e.msg
      return false
    
    # Initialize UI panels
    try:
      rootArea = Area(layout: Horizontal)
      worldMapPanel = Panel(panelType: WorldMap, name: "World Map")
      minimapPanel = Panel(panelType: Minimap, name: "Minimap")
      agentTablePanel = Panel(panelType: AgentTable, name: "Agent Table")
      agentTracesPanel = Panel(panelType: AgentTraces, name: "Agent Traces")
      globalTimelinePanel = Panel(panelType: GlobalTimeline)
      globalFooterPanel = Panel(panelType: GlobalFooter)
      globalHeaderPanel = Panel(panelType: GlobalHeader)
      
      rootArea.areas.add(Area(layout: Horizontal))
      rootArea.panels.add(worldMapPanel)
      rootArea.panels.add(minimapPanel)
      rootArea.panels.add(agentTablePanel)
      rootArea.panels.add(agentTracesPanel)
      echo "✅ UI panels initialized"
    except Exception as e:
      echo "❌ Failed to initialize UI panels: ", e.msg
      return false
    
    echo "✅ Viewer initialized with full graphics support"
    return true
    
  except Exception as e:
    echo "❌ Failed to initialize viewer: ", e.msg
    return false

proc loadAssets(): bool =
  ## Load tribal assets with detailed error handling to isolate crash
  try:
    echo "🎨 Loading tribal assets with detailed debugging..."
    var loadedCount = 0
    var totalFiles = 0
    
    # Count total PNG files first
    echo "🔧 DEBUG: Starting file count..."
    for path in walkDirRec("data/"):
      if path.endsWith(".png"):
        inc totalFiles
    echo "✅ File count completed"
    
    echo "📁 Found ", totalFiles, " PNG files to load"
    
    if totalFiles == 0:
      echo "⚠️  No PNG files found - this might be the issue!"
      return false
    
    # Load assets with error handling for each file
    echo "🔧 DEBUG: Starting asset loading loop..."
    for path in walkDirRec("data/"):
      if path.endsWith(".png"):
        inc loadedCount
        echo "🔧 Loading asset ", loadedCount, "/", totalFiles, ": ", path
        
        try:
          echo "  📖 Reading image file..."
          let imageName = path.replace("data/", "").replace(".png", "")
          let image = readImage(path)
          echo "  ✅ Image read successfully"
          
          echo "  🖼️  Adding to boxy..."
          bxy.addImage(imageName, image)
          echo "  ✅ Added to boxy successfully"
          
        except Exception as e:
          echo "  ❌ Failed on ", path, ": ", e.msg
          # Continue loading other assets even if one fails
    
    echo "🎨 Asset loading complete! Loaded ", loadedCount, "/", totalFiles, " files"
    return true
    
  except Exception as e:
    echo "❌ Failed to load assets: ", e.msg
    return false

proc initEnvironment(): bool =
  ## Initialize the tribal environment
  try:
    echo "🌍 Initializing tribal environment"
    
    # Create default environment
    env = newEnvironment()
    
    # Set up for external control (no built-in AI)
    globalController = nil  # No controller - Python will provide actions
    
    echo "✅ Environment initialized with external control"
    return true
    
  except Exception as e:
    echo "❌ Failed to initialize environment: ", e.msg
    return false

proc writeStateToFile() =
  ## Write current environment state to file for Python
  try:
    var state = newJObject()
    
    # Get observations
    var obsArray = newJArray()
    for i in 0..<MapAgents:
      var agentObs = newJArray()
      for layer in 0..<ObservationLayers:
        var layerData = newJArray()
        for y in 0..<ObservationHeight:
          var rowData = newJArray()
          for x in 0..<ObservationWidth:
            rowData.add(newJInt(env.observations[i][layer][x][y].int))
          layerData.add(rowData)
        agentObs.add(layerData)
      obsArray.add(agentObs)
    
    # Get rewards  
    var rewardsArray = newJArray()
    for i in 0..<MapAgents:
      rewardsArray.add(newJFloat(env.agents[i].reward))
    
    # Get terminals/truncations
    var terminalsArray = newJArray()
    var truncationsArray = newJArray()
    for i in 0..<MapAgents:
      terminalsArray.add(newJBool(env.agents[i].frozen >= 999999))  # Agent is dead if frozen is very high
      truncationsArray.add(newJBool(env.currentStep >= env.config.maxSteps))
    
    # Environment info
    state["observations"] = obsArray
    state["rewards"] = rewardsArray
    state["terminals"] = terminalsArray
    state["truncations"] = truncationsArray
    state["current_step"] = newJInt(env.currentStep)
    state["max_steps"] = newJInt(env.config.maxSteps)
    state["episode_done"] = newJBool(env.currentStep >= env.config.maxSteps)
    state["timestamp"] = newJFloat(epochTime())
    
    writeFile(StateFile, $state)
    
  except Exception as e:
    echo "⚠️  Failed to write state file: ", e.msg

proc readActionsFromFile(): array[MapAgents, array[2, uint8]] =
  ## Read actions from file written by Python
  var actions: array[MapAgents, array[2, uint8]]
  
  # Default to NOOP actions
  for i in 0..<MapAgents:
    actions[i] = [0'u8, 0'u8]
  
  if not fileExists(ActionsFile):
    return actions
    
  try:
    let content = readFile(ActionsFile)
    let jsonData = parseJson(content)
    
    if jsonData.hasKey("actions"):
      let actionArray = jsonData["actions"]
      for i in 0..<min(actionArray.len, MapAgents):
        let agentAction = actionArray[i]
        if agentAction.len >= 2:
          actions[i][0] = agentAction[0].getInt().uint8
          actions[i][1] = agentAction[1].getInt().uint8
    
    # Remove file after reading to avoid stale actions
    removeFile(ActionsFile)
    
  except Exception as e:
    echo "⚠️  Failed to read actions file: ", e.msg
  
  return actions

proc checkCommunication(): bool =
  ## Check if Python is still communicating
  if not fileExists(ControlFile):
    return false
    
  try:
    let content = readFile(ControlFile)
    let control = parseJson(content)
    
    if control.hasKey("active") and control["active"].getBool():
      communicationActive = true
      return true
    elif control.hasKey("shutdown") and control["shutdown"].getBool():
      echo "🛑 Received shutdown signal from Python"
      return false
      
  except Exception as e:
    echo "⚠️  Failed to read control file: ", e.msg
  
  return communicationActive

proc renderFrame() =
  ## Render one frame of the environment
  try:
    bxy.beginFrame(window.size)
    
    # Draw environment
    if not isNil(env):
      drawSafeRectangleEnvironment()
    
    # Draw UI panels
    let headerColor = color(0.15, 0.15, 0.2, 1.0)
    bxy.drawRect(
      rect = rect(0, 0, window.size.x.float32, 64),
      color = headerColor
    )
    
    let footerColor = color(0.15, 0.15, 0.2, 1.0)
    bxy.drawRect(
      rect = rect(0, window.size.y.float32 - 64, window.size.x.float32, 64),
      color = footerColor
    )
    
    let timelineColor = color(0.1, 0.1, 0.15, 1.0)
    bxy.drawRect(
      rect = rect(0, window.size.y.float32 - 128, window.size.x.float32, 64),
      color = timelineColor
    )
    
    bxy.endFrame()
    window.swapBuffers()
    
  except Exception as e:
    echo "❌ Error in renderFrame: ", e.msg

proc drawSafeRectangleEnvironment() =
  ## Enhanced rectangle-based rendering with improved visuals
  # Calculate scale to fit full map in window (1280x800)
  # Map is 100x50 tiles, so we need cellSize that fits within window bounds
  let cellSize = 10.0  # Optimized to show full 100x50 map in 1280x800 window
  let mapPixelWidth = 100.0 * cellSize   # 1000 pixels
  let mapPixelHeight = 50.0 * cellSize   # 500 pixels
  let offsetX = (1280.0 - mapPixelWidth) / 2.0   # Center horizontally: 140 pixels
  let offsetY = (800.0 - mapPixelHeight) / 2.0   # Center vertically: 150 pixels
  
  # Draw terrain background first
  for x in 0 ..< MapWidth:
    for y in 0 ..< MapHeight:
      let drawX = offsetX + x.float32 * cellSize
      let drawY = offsetY + y.float32 * cellSize
      
      # Enhanced terrain colors with better contrast
      let terrainColor = case env.terrain[x][y]:
        of Empty: color(0.75, 0.65, 0.45, 1.0)  # Sandy ground
        of Water: color(0.1, 0.4, 0.9, 1.0)     # Bright blue water
        of Wheat: color(1.0, 0.9, 0.3, 1.0)     # Golden wheat
        of Tree: color(0.2, 0.7, 0.2, 1.0)      # Forest green
      
      # Draw terrain tile with border
      bxy.drawRect(
        rect = rect(drawX, drawY, cellSize, cellSize),
        color = terrainColor
      )
      
      # Add subtle border for grid visibility
      bxy.drawRect(
        rect = rect(drawX, drawY, cellSize, 1),
        color = color(0.0, 0.0, 0.0, 0.1)  # Dark border
      )
      bxy.drawRect(
        rect = rect(drawX, drawY, 1, cellSize),
        color = color(0.0, 0.0, 0.0, 0.1)  # Dark border
      )
  
  # Draw grid objects (buildings, etc.)
  for x in 0 ..< MapWidth:
    for y in 0 ..< MapHeight:
      if not isNil(env.grid[x][y]):
        let thing = env.grid[x][y]
        let drawX = offsetX + x.float32 * cellSize
        let drawY = offsetY + y.float32 * cellSize
        
        # Enhanced object colors with better contrast and shape variety
        let (objectColor, borderColor) = case thing.kind:
          of Agent: continue  # Agents drawn separately
          of Wall: (color(0.2, 0.2, 0.2, 1.0), color(0.1, 0.1, 0.1, 1.0))     # Dark walls
          of Altar: (color(0.9, 0.2, 0.9, 1.0), color(0.6, 0.1, 0.6, 1.0))    # Bright purple altars
          of Mine: (color(0.8, 0.4, 0.1, 1.0), color(0.5, 0.2, 0.0, 1.0))     # Copper mines
          of Converter: (color(0.0, 0.9, 0.9, 1.0), color(0.0, 0.6, 0.6, 1.0)) # Bright cyan converters
          of Forge: (color(1.0, 0.6, 0.2, 1.0), color(0.7, 0.3, 0.0, 1.0))    # Bright orange forges
          of Armory: (color(0.6, 0.6, 1.0, 1.0), color(0.3, 0.3, 0.7, 1.0))   # Light blue armory
          of ClayOven: (color(0.9, 0.5, 0.3, 1.0), color(0.6, 0.2, 0.1, 1.0)) # Terracotta color
          of WeavingLoom: (color(0.7, 0.9, 0.5, 1.0), color(0.4, 0.6, 0.2, 1.0)) # Bright green
          of Spawner: (color(1.0, 0.3, 0.3, 1.0), color(0.7, 0.0, 0.0, 1.0))  # Bright red spawners
          of Clippy: (color(0.8, 0.2, 0.8, 1.0), color(0.5, 0.0, 0.5, 1.0))   # Magenta clippies
        
        # Draw main object with border for better visibility
        let objSize = cellSize - 4
        let objX = drawX + 2
        let objY = drawY + 2
        
        # Draw border first
        bxy.drawRect(
          rect = rect(objX - 1, objY - 1, objSize + 2, objSize + 2),
          color = borderColor
        )
        
        # Draw main object
        bxy.drawRect(
          rect = rect(objX, objY, objSize, objSize),
          color = objectColor
        )
  
  # Draw agents on top
  for i in 0 ..< min(env.agents.len, MapAgents):
    let agent = env.agents[i]
    if not isNil(agent) and agent.frozen < 999999:  # Agent is alive
      let drawX = offsetX + agent.pos.x.float32 * cellSize
      let drawY = offsetY + agent.pos.y.float32 * cellSize
      
      # Enhanced agent colors with better contrast
      let agentColor = case (i mod 10):
        of 0: color(1.0, 0.3, 0.3, 1.0)  # Bright red
        of 1: color(0.3, 1.0, 0.3, 1.0)  # Bright green  
        of 2: color(0.3, 0.3, 1.0, 1.0)  # Bright blue
        of 3: color(1.0, 1.0, 0.3, 1.0)  # Bright yellow
        of 4: color(1.0, 0.3, 1.0, 1.0)  # Bright magenta
        of 5: color(0.3, 1.0, 1.0, 1.0)  # Bright cyan
        of 6: color(1.0, 0.7, 0.3, 1.0)  # Bright orange
        of 7: color(1.0, 1.0, 1.0, 1.0)  # White
        of 8: color(1.0, 0.5, 0.7, 1.0)  # Pink
        else: color(0.7, 1.0, 0.5, 1.0)  # Light green
      
      # Draw agent as circle-like shape (rounded rectangle)
      let agentSize = cellSize - 6
      let agentX = drawX + 3
      let agentY = drawY + 3
      
      # Draw black border
      bxy.drawRect(
        rect = rect(agentX - 1, agentY - 1, agentSize + 2, agentSize + 2),
        color = color(0.0, 0.0, 0.0, 1.0)
      )
      
      # Draw agent
      bxy.drawRect(
        rect = rect(agentX, agentY, agentSize, agentSize),
        color = agentColor
      )
      
      # Add small directional indicator based on orientation
      let dirSize = 2.0
      let centerX = agentX + agentSize / 2
      let centerY = agentY + agentSize / 2
      
      let (dirX, dirY) = case agent.orientation:
        of N: (centerX - dirSize/2, agentY + 1)
        of S: (centerX - dirSize/2, agentY + agentSize - dirSize - 1) 
        of E: (agentX + agentSize - dirSize - 1, centerY - dirSize/2)
        of W: (agentX + 1, centerY - dirSize/2)
        else: (centerX - dirSize/2, centerY - dirSize/2)  # Default center for diagonal orientations
      
      bxy.drawRect(
        rect = rect(dirX, dirY, dirSize, dirSize),
        color = color(0.0, 0.0, 0.0, 1.0)  # Black directional dot
      )
  
  # Commented out complex UI code - now using simple rectangle rendering

proc runMainLoop() =
  ## Main game loop with file-based communication
  echo "🚀 Starting main loop with file-based communication"
  echo "   Python writes actions to: ", ActionsFile
  echo "   Nim writes state to: ", StateFile
  echo "   Control via: ", ControlFile
  
  let targetFPS = 30.0
  let frameTime = 1.0 / targetFPS
  var lastFrameTime = epochTime()
  
  # Reset environment
  env.reset()
  writeStateToFile()
  
  # Main loop with graphics
  while not window.closeRequested:
    let currentTime = epochTime()
    
    # Check for communication from Python
    if not checkCommunication():
      sleep(100)  # Wait 100ms before checking again
      pollEvents()  # Still need to poll events to keep window responsive
      continue
    
    # Read actions from Python (if available)
    let actions = readActionsFromFile()
    
    # Step environment with error handling
    try:
      env.step(actions.addr)
    except Exception as e:
      echo "❌ CRASH in env.step: ", e.msg
      quit(1)
    
    # Write current state back to Python with error handling
    try:
      writeStateToFile()
    except Exception as e:
      echo "❌ CRASH in writeStateToFile: ", e.msg
      quit(1)
    
    # Render frame safely
    try:
      renderFrame()
    except Exception as e:
      echo "❌ CRASH in renderFrame: ", e.msg
      quit(1)
    
    # Handle window events
    pollEvents()
    
    # Frame rate limiting
    let frameElapsed = currentTime - lastFrameTime
    if frameElapsed < frameTime:
      let sleepMs = int((frameTime - frameElapsed) * 1000)
      if sleepMs > 0:
        sleep(sleepMs)
    lastFrameTime = epochTime()

proc runTestLoop() =
  ## Standalone test loop that shows the environment without Python communication
  echo "🚀 Starting test loop - showing environment directly"
  
  let targetFPS = 30.0
  let frameTime = 1.0 / targetFPS
  var lastFrameTime = epochTime()
  var stepCounter = 0
  
  # Reset environment
  env.reset()
  echo "✅ Environment reset for test mode"
  
  # Seed random for reproducible agent movement
  randomize()
  
  # Main test loop with graphics
  while not window.closeRequested:
    let currentTime = epochTime()
    
    # Simple agent movement every 10 frames for visual interest
    if stepCounter mod 10 == 0:
      echo "🎲 Taking random environment step ", stepCounter div 10
      
      # Generate random actions for all agents
      var actions: array[MapAgents, array[2, uint8]]
      
      # Default to NOOP actions
      for i in 0..<MapAgents:
        actions[i] = [0'u8, 0'u8]  # NOOP action
      
      # Generate random actions for alive agents
      var activeActions = 0
      for i in 0 ..< min(env.agents.len, MapAgents):
        if not isNil(env.agents[i]) and env.agents[i].frozen < 999999:
          # Random actions: 0=NOOP, 1=MOVE(0-3), 2=ROTATE, 3=GET(0-7)
          let actionType = rand(3)  # 0-3 
          let argument = case actionType:
            of 1: rand(3)  # MOVE: N, S, W, E (0-3)
            of 3: rand(7)  # GET: 8 directions (0-7)
            else: 0
          
          actions[i] = [actionType.uint8, argument.uint8]
          inc activeActions
      
      # Step environment with random actions
      try:
        env.step(actions.addr)
        echo "✅ Environment stepped with ", activeActions, " active agents"
      except Exception as e:
        echo "⚠️  Error in env.step (continuing): ", e.msg
    
    stepCounter += 1
    
    # Render frame
    try:
      renderFrame()
    except Exception as e:
      echo "❌ Error in renderFrame: ", e.msg
      # Don't quit, keep trying
    
    # Handle window events
    pollEvents()
    
    # Frame rate limiting
    let frameElapsed = currentTime - lastFrameTime
    if frameElapsed < frameTime:
      let sleepMs = int((frameTime - frameElapsed) * 1000)
      if sleepMs > 0:
        sleep(sleepMs)
    lastFrameTime = epochTime()
  
  echo "🏁 Test loop finished"

# Main entry point
proc main() =
  ## Main entry point for the process viewer
  echo "🎮 Tribal Process-Separated Viewer Starting"
  
  # Check for test mode argument
  if paramCount() > 0 and paramStr(1) == "--test":
    testMode = true
    echo "🧪 Test mode enabled - running standalone without Python communication"
  
  if not initViewer():
    echo "❌ Failed to initialize viewer"
    quit(1)
  
  # Now we know the crash is in panels, not assets - re-enable asset loading!
  if loadAssets():
    assetsLoaded = true
    echo "✅ PNG assets loaded - using full sprite rendering"
  else:
    assetsLoaded = false
    echo "⚠️  Asset loading failed - using safe rectangle rendering"
    
  if not initEnvironment():
    echo "❌ Failed to initialize environment" 
    quit(1)
  
  try:
    if testMode:
      # Run standalone test mode 
      echo "🧪 Starting standalone test mode"
      runTestLoop()
    else:
      # Create initial control file to signal readiness
      let controlData = %*{"ready": true, "active": false, "pid": getCurrentProcessId()}
      writeFile(ControlFile, $controlData)
      
      echo "🎯 Viewer ready - waiting for Python communication"
      echo "   Python should write to: ", ActionsFile
      echo "   Python should set active=true in: ", ControlFile
      
      runMainLoop()
  except Exception as e:
    echo "❌ Error in main loop: ", e.msg
  finally:
    echo "🧹 Cleaning up viewer"
    if fileExists(ControlFile):
      removeFile(ControlFile)
    if fileExists(StateFile):
      removeFile(StateFile)
    if fileExists(ActionsFile):
      removeFile(ActionsFile)

when isMainModule:
  main()