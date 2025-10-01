import
  std/[times, tables],
  boxy, windy, vmath, fidget2,
  replays

type
  IRect* = object
    x*: int32
    y*: int32
    w*: int32
    h*: int32

  PanelType* = enum
    GlobalHeader
    GlobalFooter
    GlobalTimeline

    WorldMap
    Minimap
    AgentTable
    AgentTraces
    EnvironmentInfo
    ObjectInfo

  Panel* = ref object
    panelType*: PanelType
    rect*: IRect
    name*: string            ## The name of the panel.
    header*: Node            ## The header of the panel.
    node*: Node              ## The node of the panel.
    parentArea*: Area        ## The parent area of the panel.

    pos*: Vec2
    vel*: Vec2
    zoom*: float32 = 10
    zoomVel*: float32
    minZoom*: float32 = 2
    maxZoom*: float32 = 1000
    scrollArea*: Rect
    hasMouse*: bool = false

  AreaLayout* = enum
    Horizontal
    Vertical

  Area* = ref object
    node*: Node              ## The node of the area.
    layout*: AreaLayout      ## The layout of the area.
    areas*: seq[Area]        ## The subareas in the area (0 or 2)
    panels*: seq[Panel]      ## The panels in the area.
    split*: float32          ## The split percentage of the area.
    selectedPanelNum*: int   ## The index of the selected panel in the area.

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
  frame*: int

  globalTimelinePanel*: Panel
  globalFooterPanel*: Panel
  globalHeaderPanel*: Panel

  worldMapPanel*: Panel
  minimapPanel*: Panel
  agentTablePanel*: Panel
  agentTracesPanel*: Panel
  objectInfoPanel*: Panel
  environmentInfoPanel*: Panel

  settings* = Settings()
  selection*: Entity

  step*: int = 0
  stepFloat*: float32 = 0
  replay*: Replay
  play*: bool
  playSpeed*: float32 = 0.1
  lastSimTime*: float64 = epochTime()
  playMode* = Historical

  ## Signals when we want to give control back to Python (DLL mode only).
  requestPython*: bool = false
  requestAction*: bool = false
  requestActionAgentId*: int
  requestActionActionId*: int
  requestActionArgument*: int

  followSelection*: bool = false
  mouseCaptured*: bool = false
  mouseCapturedPanel*: Panel = nil

## Path queue for each agent. Maps agentId to a sequence of grid positions.
var agentPaths* = initTable[int, seq[IVec2]]()

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
