import
  std/[times, tables, os, strutils, pathnorm],
  boxy, windy, vmath, silky,
  replays

var rootDir* = "packages/mettagrid/nim/mettascope/"
var dataDir* = rootDir / "data"

proc setDataDir*(path: string) =
  dataDir = path.normalizePath
  rootDir = dataDir.parentDir.normalizePath

type
  IRect* = object
    x*: int32
    y*: int32
    w*: int32
    h*: int32

  Settings* = object
    showFogOfWar* = false
    showVisualRange* = true
    showGrid* = true
    showResources* = true
    showObservations* = -1
    lockFocus* = false

  PlayMode* = enum
    Historical
    Realtime

var
  sk*: Silky
  bxy*: Boxy
  window*: Window
  frame*: int

  settings* = Settings()
  selection*: Entity

  step*: int = 0
  stepFloat*: float32 = 0
  previousStep*: int = -1
  replay*: Replay
  play*: bool
  playSpeed*: float32 = 10.0
  lastSimTime*: float64 = epochTime()
  playMode* = Historical

  ## Signals when we want to give control back to Python (DLL mode only).
  requestPython*: bool = false

  # Command line arguments.
  commandLineReplay*: string = ""


type
  ActionRequest* = object
    agentId*: int
    actionName*: cstring

  ObjectiveKind* = enum
    Move # Move to a specific position.
    Bump # Bump an object at a specific position to interact with it.
    Vibe # Execute a specific vibe action.

  Objective* = object
    case kind*: ObjectiveKind
    of Move, Bump:
      pos*: IVec2
      approachDir*: IVec2 ## Direction to approach from for Bump actions (e.g., ivec2(-1, 0) means approach from the left).
    of Vibe:
      vibeActionId*: int
    repeat*: bool ## If true, this objective will be re-queued at the end when completed.

  PathAction* = object
    case kind*: ObjectiveKind
    of Move:
      pos*: IVec2 ## Target position for move.
    of Bump:
      bumpPos*: IVec2 ## Bump target position.
      bumpDir*: IVec2 ## Direction to bump for bump actions.
    of Vibe:
      vibeActionId*: int

var
  requestActions*: seq[ActionRequest]

var
  ## Path queue for each agent. Maps agentId to a sequence of path actions.
  agentPaths* = initTable[int, seq[PathAction]]()
  ## Objective queue for each agent. Maps agentId to a sequence of objectives.
  agentObjectives* = initTable[int, seq[Objective]]()
  ## Track mouse down position to distinguish clicks from drags.
  mouseDownPos*: Vec2

proc at*[T](sequence: seq[T], step: int): T =
  # Get the value at the given step.
  if sequence.len == 0:
    return default(T)
  sequence[step.clamp(0, sequence.len - 1)]

proc at*[T](sequence: seq[T]): T =
  # Get the value at the current step.
  sequence.at(step)

proc irect*(x, y, w, h: SomeNumber): IRect =
  IRect(x: x.int32, y: y.int32, w: w.int32, h: h.int32)

proc rect*(rect: IRect): Rect =
  Rect(
    x: rect.x.float32,
    y: rect.y.float32,
    w: rect.w.float32,
    h: rect.h.float32
  )

proc xy*(rect: IRect): IVec2 =
  ivec2(rect.x, rect.y)

proc wh*(rect: IRect): IVec2 =
  ivec2(rect.w, rect.h)

proc getAgentById*(agentId: int): Entity =
  ## Get an agent by ID. Asserts the agent exists.
  for obj in replay.objects:
    if obj.isAgent and obj.agentId == agentId:
      return obj
  raise newException(ValueError, "Agent with ID " & $agentId & " does not exist")

proc getObjectById*(objectId: int): Entity =
  ## Get an object by ID. Asserts the object exists.
  for obj in replay.objects:
    if obj.id == objectId:
      return obj
  raise newException(ValueError, "Object with ID " & $objectId & " does not exist")

proc getObjectAtLocation*(pos: IVec2): Entity =
  ## Get the first object at the given position. Returns nil if no object is there.
  for obj in replay.objects:
    if obj.location.at(step).xy == pos:
      return obj
  return nil

proc getVibeName*(vibeId: int): string =
  if vibeId >= 0 and vibeId < replay.config.game.vibeNames.len:
    result = replay.config.game.vibeNames[vibeId]
  else:
    raise newException(ValueError, "Vibe with ID " & $vibeId & " does not exist")
