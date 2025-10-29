## Ultra-Fast Direct Buffer Interface
## Zero-copy numpy buffer communication - no conversions

import environment, external_actions

var globalEnv: Environment = nil

proc tribal_village_create(): pointer {.exportc, dynlib.} =
  ## Create environment for direct buffer interface
  try:
    let config = defaultEnvironmentConfig()
    globalEnv = newEnvironment(config)
    initGlobalController(ExternalNN)
    return cast[pointer](globalEnv)
  except:
    return nil

proc tribal_village_reset_and_get_obs(
  env: pointer,
  obs_buffer: ptr UncheckedArray[uint8],    # [60, 21, 11, 11] direct
  rewards_buffer: ptr UncheckedArray[float32],
  terminals_buffer: ptr UncheckedArray[uint8],
  truncations_buffer: ptr UncheckedArray[uint8]
): int32 {.exportc, dynlib.} =
  ## Reset and write directly to buffers - no conversions
  if globalEnv == nil:
    return 0

  try:
    globalEnv.reset()

    # Direct memory copy of observations (zero conversion)
    let obs_size = MapAgents * ObservationLayers * ObservationWidth * ObservationHeight
    copyMem(obs_buffer, globalEnv.observations.addr, obs_size)

    # Clear rewards/terminals/truncations
    for i in 0..<MapAgents:
      rewards_buffer[i] = 0.0
      terminals_buffer[i] = 0
      truncations_buffer[i] = 0

    return 1
  except:
    return 0

proc tribal_village_step_with_pointers(
  env: pointer,
  actions_buffer: ptr UncheckedArray[uint8],    # [60, 2] direct read
  obs_buffer: ptr UncheckedArray[uint8],        # [60, 21, 11, 11] direct write
  rewards_buffer: ptr UncheckedArray[float32],
  terminals_buffer: ptr UncheckedArray[uint8],
  truncations_buffer: ptr UncheckedArray[uint8]
): int32 {.exportc, dynlib.} =
  ## Ultra-fast step with direct buffer access
  if globalEnv == nil:
    return 0

  try:
    # Read actions directly from buffer (no conversion)
    var actions: array[MapAgents, array[2, uint8]]
    for i in 0..<MapAgents:
      actions[i][0] = actions_buffer[i * 2]
      actions[i][1] = actions_buffer[i * 2 + 1]

    # Step environment
    globalEnv.step(unsafeAddr actions)

    # Direct memory copy of observations (zero conversion overhead)
    let obs_size = MapAgents * ObservationLayers * ObservationWidth * ObservationHeight
    copyMem(obs_buffer, globalEnv.observations.addr, obs_size)

    # Direct buffer writes (no dict conversion)
    for i in 0..<MapAgents:
      rewards_buffer[i] = globalEnv.agents[i].reward
      terminals_buffer[i] = if globalEnv.terminated[i] > 0.0: 1 else: 0
      truncations_buffer[i] = if globalEnv.truncated[i] > 0.0: 1 else: 0

    return 1
  except:
    return 0

proc tribal_village_get_num_agents(): int32 {.exportc, dynlib.} =
  return MapAgents.int32

proc tribal_village_get_obs_layers(): int32 {.exportc, dynlib.} =
  return ObservationLayers.int32

proc tribal_village_get_obs_width(): int32 {.exportc, dynlib.} =
  return ObservationWidth.int32


proc tribal_village_get_map_width(): int32 {.exportc, dynlib.} =
  return MapWidth.int32

proc tribal_village_get_map_height(): int32 {.exportc, dynlib.} =
  return MapHeight.int32

# Render full map as HxWx3 RGB (uint8)
proc toByte(value: float32): uint8 =
  var iv = int(value * 255.0)
  if iv < 0:
    iv = 0
  elif iv > 255:
    iv = 255
  result = uint8(iv)

proc tribal_village_render_rgb(
  env: pointer,
  out_buffer: ptr UncheckedArray[uint8],
  out_w: int32,
  out_h: int32
): int32 {.exportc, dynlib.} =
  if globalEnv == nil or out_buffer.isNil:
    return 0

  let width = int(out_w)
  let height = int(out_h)
  if width <= 0 or height <= 0:
    return 0
  if width mod MapWidth != 0 or height mod MapHeight != 0:
    return 0

  let scaleX = width div MapWidth
  let scaleY = height div MapHeight
  let stride = width * 3

  try:
    for y in 0 ..< MapHeight:
      for sy in 0 ..< scaleY:
        let rowBase = (y * scaleY + sy) * stride
        for x in 0 ..< MapWidth:
          var rByte = toByte(globalEnv.tileColors[x][y].r)
          var gByte = toByte(globalEnv.tileColors[x][y].g)
          var bByte = toByte(globalEnv.tileColors[x][y].b)

          let thing = globalEnv.grid[x][y]
          if thing != nil:
            case thing.kind
            of Agent:
              rByte = 255'u8
              gByte = 255'u8
              bByte = 0'u8
            of Tumor:
              rByte = 160'u8
              gByte = 32'u8
              bByte = 240'u8
            of Wall:
              rByte = 96'u8
              gByte = 96'u8
              bByte = 96'u8
            of Mine:
              rByte = 184'u8
              gByte = 134'u8
              bByte = 11'u8
            of Converter:
              rByte = 0'u8
              gByte = 200'u8
              bByte = 200'u8
            of Altar:
              rByte = 220'u8
              gByte = 0'u8
              bByte = 220'u8
            of Spawner:
              rByte = 255'u8
              gByte = 170'u8
              bByte = 0'u8
            of Armory:
              rByte = 255'u8
              gByte = 120'u8
              bByte = 40'u8
            of Forge:
              rByte = 255'u8
              gByte = 80'u8
              bByte = 0'u8
            of ClayOven:
              rByte = 255'u8
              gByte = 180'u8
              bByte = 120'u8
            of WeavingLoom:
              rByte = 0'u8
              gByte = 180'u8
              bByte = 255'u8
            of PlantedLantern:
              rByte = 255'u8
              gByte = 240'u8
              bByte = 128'u8
            else:
              discard

          let xBase = rowBase + x * scaleX * 3
          for sx in 0 ..< scaleX:
            let idx = xBase + sx * 3
            out_buffer[idx] = rByte
            out_buffer[idx + 1] = gByte
            out_buffer[idx + 2] = bByte
    return 1
  except:
    return 0
proc tribal_village_get_obs_height(): int32 {.exportc, dynlib.} =
  return ObservationHeight.int32

proc tribal_village_destroy(env: pointer) {.exportc, dynlib.} =
  ## Clean up environment
  globalEnv = nil

# --- Rendering interface (ANSI) ---
proc tribal_village_render_ansi(
  env: pointer,
  out_buffer: ptr UncheckedArray[char],
  buf_len: int32
): int32 {.exportc, dynlib.} =
  ## Write an ANSI string render into out_buffer (null-terminated).
  ## Returns number of bytes written (excluding terminator). 0 on error.
  if globalEnv == nil or out_buffer.isNil or buf_len <= 1:
    return 0

  try:
    let s = render(globalEnv)  # environment.render*(env: Environment): string
    let n = min(s.len, max(0, buf_len - 1).int)
    if n > 0:
      copyMem(out_buffer, cast[pointer](s.cstring), n)
    out_buffer[n] = '\0'  # null-terminate
    return n.int32
  except:
    return 0
