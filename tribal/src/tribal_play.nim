## Tribal Play Interface - Direct Nim Implementation with nimpy
## This program runs the tribal environment play interface directly from Nim,
## interfacing with Python through nimpy for web server functionality.

import nimpy, json, asynchttpserver, asyncdispatch, websocket, uri, strutils, times
import tribal/[environment, common]

type
  TribalPlayServer* = ref object
    env*: Environment
    httpServer*: AsyncHttpServer
    port*: int
    isRunning*: bool

var gPlayServer*: TribalPlayServer

proc initTribalPlayServer*(port: int = 8765): TribalPlayServer =
  ## Initialize the tribal play server
  result = TribalPlayServer(
    env: newEnvironment(defaultEnvironmentConfig()),
    httpServer: newAsyncHttpServer(),
    port: port,
    isRunning: false
  )
  gPlayServer = result

proc handleWebSocket(req: Request) {.async.} =
  ## Handle WebSocket connections for real-time game interaction
  let ws = await newWebSocket(req)
  echo "New WebSocket connection established"
  
  # Send initial game state
  let initialState = %*{
    "type": "gameState",
    "observations": gPlayServer.env.getObservationsJson(),
    "agents": gPlayServer.env.getAgentsJson(),
    "step": gPlayServer.env.currentStep
  }
  await ws.send($initialState)
  
  while ws.readyState == Open:
    try:
      let message = await ws.receiveStrPacket()
      let data = parseJson(message)
      
      case data["type"].getStr():
      of "step":
        # Execute one simulation step with provided actions
        if data.hasKey("actions"):
          let actions = data["actions"]
          var nimActions: array[MapAgents, array[2, uint8]]
          
          # Convert JSON actions to Nim format
          for i in 0..<MapAgents:
            if i < actions.len:
              nimActions[i][0] = actions[i][0].getInt().uint8
              nimActions[i][1] = actions[i][1].getInt().uint8
            else:
              nimActions[i][0] = 0  # NOOP
              nimActions[i][1] = 0
          
          gPlayServer.env.step(nimActions.addr)
        else:
          # Step with random actions for testing
          gPlayServer.env.stepRandom()
        
        # Send updated game state
        let gameState = %*{
          "type": "gameState", 
          "observations": gPlayServer.env.getObservationsJson(),
          "agents": gPlayServer.env.getAgentsJson(),
          "rewards": gPlayServer.env.getRewardsJson(),
          "step": gPlayServer.env.currentStep,
          "done": gPlayServer.env.isDone()
        }
        await ws.send($gameState)
        
      of "reset":
        # Reset the environment
        gPlayServer.env.reset()
        let gameState = %*{
          "type": "gameState",
          "observations": gPlayServer.env.getObservationsJson(),
          "agents": gPlayServer.env.getAgentsJson(),
          "step": gPlayServer.env.currentStep
        }
        await ws.send($gameState)
        
      of "pause":
        # Pause/unpause simulation (for future implementation)
        echo "Pause command received"
        
      else:
        echo "Unknown message type: ", data["type"].getStr()
        
    except WebSocketClosedError:
      echo "WebSocket connection closed"
      break
    except Exception as e:
      echo "Error handling WebSocket message: ", e.msg
      break

proc handleHttpRequest(req: Request) {.async.} =
  ## Handle HTTP requests and upgrade to WebSocket when needed
  if req.url.path == "/ws":
    await handleWebSocket(req)
  else:
    # Serve basic HTTP response
    await req.respond(Http200, "Tribal Play Server Running")

proc startPlayServer*(): Future[void] {.async.} =
  ## Start the tribal play server
  echo "Starting Tribal Play Server on port ", gPlayServer.port
  gPlayServer.isRunning = true
  gPlayServer.httpServer.listen(Port(gPlayServer.port))
  
  while gPlayServer.isRunning:
    if gPlayServer.httpServer.shouldAcceptRequest():
      await gPlayServer.httpServer.acceptRequest(handleHttpRequest)
    await sleepAsync(1)

proc stopPlayServer*() =
  ## Stop the tribal play server
  echo "Stopping Tribal Play Server"
  gPlayServer.isRunning = false
  gPlayServer.httpServer.close()

proc openBrowserUrl*(url: string) =
  ## Open browser URL using Python webbrowser module via nimpy
  let webbrowser = pyImport("webbrowser")
  discard webbrowser.open(url)

proc main() =
  ## Main entry point for tribal play
  echo "Tribal Play - Direct Nim Implementation"
  echo "======================================="
  
  # Initialize the server
  let server = initTribalPlayServer(8765)
  
  # Open browser (optional)
  try:
    let url = "http://localhost:8765/ws"
    echo "Opening browser at: ", url
    openBrowserUrl(url)
  except:
    echo "Could not open browser automatically"
    echo "Please visit: http://localhost:8765/ws"
  
  # Start the server
  echo "Starting server..."
  try:
    waitFor startPlayServer()
  except KeyboardInterrupt:
    echo "\nShutting down server..."
    stopPlayServer()
  except Exception as e:
    echo "Server error: ", e.msg
    stopPlayServer()

# Add helper methods to Environment for JSON serialization
proc getObservationsJson*(env: Environment): JsonNode =
  ## Convert observations to JSON format
  result = newJArray()
  for agentId in 0..<MapAgents:
    let agentObs = newJArray()
    for layer in 0..<ObservationLayers:
      let layerData = newJArray()
      for y in 0..<ObservationHeight:
        let rowData = newJArray()
        for x in 0..<ObservationWidth:
          rowData.add(newJInt(env.observations[agentId][layer][x][y].int))
        layerData.add(rowData)
      agentObs.add(layerData)
    result.add(agentObs)

proc getAgentsJson*(env: Environment): JsonNode =
  ## Convert agents to JSON format
  result = newJArray()
  for i in 0..<min(MapAgents, env.agents.len):
    let agent = env.agents[i]
    result.add(%*{
      "id": i,
      "x": agent.x,
      "y": agent.y,
      "health": agent.health,
      "alive": agent.alive
    })

proc getRewardsJson*(env: Environment): JsonNode =
  ## Convert rewards to JSON format
  result = newJArray()
  for i in 0..<MapAgents:
    if i < env.agents.len:
      result.add(newJFloat(env.agents[i].reward))
    else:
      result.add(newJFloat(0.0))

proc stepRandom*(env: Environment) =
  ## Step with random actions for testing
  var actions: array[MapAgents, array[2, uint8]]
  for i in 0..<MapAgents:
    actions[i][0] = rand(6).uint8  # Random action type (0-5)
    actions[i][1] = rand(8).uint8  # Random direction/argument (0-7)
  env.step(actions.addr)

proc isDone*(env: Environment): bool =
  ## Check if episode is done
  env.currentStep >= env.config.maxSteps

when isMainModule:
  main()