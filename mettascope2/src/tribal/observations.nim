## Observation system for agent perception
## Handles observation updates and rendering for the environment

import std/strformat, vmath
import environment_core

type
  ObservationName* = enum
    AgentLayer = 0
    AgentOrientationLayer = 1
    AgentInventoryOreLayer = 2
    AgentInventoryBatteryLayer = 3
    AgentInventoryWaterLayer = 4
    AgentInventoryWheatLayer = 5
    AgentInventoryWoodLayer = 6
    AgentInventorySpearLayer = 7
    WallLayer = 8
    MineLayer = 9
    MineResourceLayer = 10
    MineReadyLayer = 11
    ConverterLayer = 12
    ConverterReadyLayer = 13
    AltarLayer = 14
    AltarHeartsLayer = 15  # Hearts for respawning
    AltarReadyLayer = 16

proc clear[T](s: var openarray[T]) =
  ## Clear the entire array and set everything to 0
  let p = cast[pointer](s[0].addr)
  zeroMem(p, s.len * sizeof(T))

proc clear[N: int, T](s: ptr array[N, T]) =
  ## Clear the entire array and set everything to 0
  let p = cast[pointer](s[][0].addr)
  zeroMem(p, s[].len * sizeof(T))

proc updateObservations*(env: Environment, agentId: int) =
  ## Update observations for a specific agent
  var obs = env.observations[agentId].addr
  obs.clear()
  
  let agent = env.agents[agentId]
  var
    gridOffset = agent.pos - ivec2(ObservationWidth div 2, ObservationHeight div 2)
    gridStart = gridOffset
    gridEnd = gridOffset + ivec2(ObservationWidth, ObservationHeight)
  
  if gridStart.x < 0:
    gridStart.x = 0
  if gridStart.y < 0:
    gridStart.y = 0
  if gridEnd.x > MapWidth: gridEnd.x = MapWidth
  if gridEnd.y > MapHeight: gridEnd.y = MapHeight
  
  for gy in gridStart.y ..< gridEnd.y:
    for gx in gridStart.x ..< gridEnd.x:
      let thing = env.grid[gx][gy]
      if thing == nil:
        continue
      let x = gx - gridOffset.x
      let y = gy - gridOffset.y
      
      case thing.kind
      of Agent:
        # Layer 0: Agent present
        obs[0][x][y] = 1
        # Layer 1: Agent orientation
        obs[1][x][y] = thing.orientation.uint8
        # Layer 2: Agent ore inventory
        obs[2][x][y] = thing.inventoryOre.uint8
        # Layer 3: Agent battery inventory
        obs[3][x][y] = thing.inventoryBattery.uint8
        # Layer 4: Agent water inventory
        obs[4][x][y] = thing.inventoryWater.uint8
        # Layer 5: Agent wheat inventory
        obs[5][x][y] = thing.inventoryWheat.uint8
        # Layer 6: Agent wood inventory
        obs[6][x][y] = thing.inventoryWood.uint8
        # Layer 7: Agent spear inventory
        obs[7][x][y] = thing.inventorySpear.uint8
      
      of Wall:
        # Layer 8: Wall
        obs[8][x][y] = 1
      
      of Mine:
        # Layer 9: Mine
        obs[9][x][y] = 1
        # Layer 10: Mine resources
        obs[10][x][y] = thing.resources.uint8
        # Layer 11: Mine ready
        obs[11][x][y] = (thing.cooldown == 0).uint8
      
      of Converter:
        # Layer 12: Converter
        obs[12][x][y] = 1
        # Layer 13: Converter ready
        obs[13][x][y] = (thing.cooldown == 0).uint8
      
      of Altar:
        # Layer 14: Altar
        obs[14][x][y] = 1
        # Layer 15: Altar hearts
        obs[15][x][y] = thing.hearts.uint8
        # Layer 16: Altar ready
        obs[16][x][y] = (thing.cooldown == 0).uint8
      
      of Temple, Clippy, Armory, Forge, ClayOven, WeavingLoom:
        # These entities don't have observations in the current system
        discard

proc updateObservations*(env: Environment, layer: ObservationName, pos: IVec2, value: int) =
  ## Update a specific observation layer at a position for all agents
  let layerId = ord(layer)
  for agentId in 0 ..< MapAgents:
    let x = pos.x - env.agents[agentId].pos.x + ObservationWidth div 2
    let y = pos.y - env.agents[agentId].pos.y + ObservationHeight div 2
    if x < 0 or x >= ObservationWidth or y < 0 or y >= ObservationHeight:
      continue
    env.observations[agentId][layerId][x][y] = value.uint8

proc renderObservations*(env: Environment): string =
  ## Render the observations as a string for debugging
  const featureNames = [
    "agent",
    "agent:orientation",
    "agent:inv:ore",
    "agent:inv:battery",
    "agent:inv:water",
    "agent:inv:wheat",
    "agent:inv:wood",
    "agent:inv:spear",
    "wall",
    "mine",
    "mine:resources",
    "mine:ready",
    "converter",
    "converter:ready",
    "altar",
    "altar:hearts",
    "altar:ready",
  ]
  
  for id, obs in env.observations:
    result.add "Agent: " & $id & "\n"
    for layer in 0 ..< ObservationLayers:
      result.add "Feature " & $featureNames[layer] & " " & $layer & "\n"
      for y in 0 ..< ObservationHeight:
        for x in 0 ..< ObservationWidth:
          result.formatValue(obs[layer][x][y], "4d")
        result.add "\n"