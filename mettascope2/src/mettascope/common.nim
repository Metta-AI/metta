import std/[times],
  boxy, windy, vmath, replays

type
  IRect* = object
    x*: int
    y*: int
    w*: int
    h*: int

  PanelType* = enum
    GlobalHeader
    GlobalFooter
    GlobalTimeline

    WorldMap
    Minimap
    AgentTable
    AgentTraces
    EnvConfig

  Panel* = ref object
    panelType*: PanelType
    rect*: IRect
    name*: string

    pos*: Vec2
    vel*: Vec2
    zoom*: float32 = 10
    zoomVel*: float32
    minZoom*: float32 = 5
    maxZoom*: float32 = 100
    scrollArea*: Rect
    hasMouse*: bool = false

  AreaLayout* = enum
    Horizontal
    Vertical

  Area* = ref object
    layout*: AreaLayout
    rect*: IRect
    areas*: seq[Area]
    selectedPanelNum*: int
    panels*: seq[Panel]

  Settings* = object
    showFogOfWar* = false
    showVisualRange* = true
    showGrid* = true
    showResources* = true
    showObservations* = -1
    lockFocus* = false

var
  rootArea*: Area
  frame*: int

  globalTimelinePanel*: Panel
  globalFooterPanel*: Panel
  globalHeaderPanel*: Panel

  worldMapPanel*: Panel
  minimapPanel*: Panel
  agentTablePanel*: Panel
  agentTracesPanel*: Panel
  mgConfigPanel*: Panel

  settings* = Settings()
  selection*: Entity

  step*: int = 0
  stepFloat*: float32 = 0
  replay*: Replay
  play*: bool
  playSpeed*: float32 = 0.1
  lastSimTime*: float64 = epochTime()

  followSelection*: bool = false
  mouseCaptured*: bool = false
  mouseCapturedPanel*: Panel = nil

proc at*[T](sequence: seq[T], step: int): T =
  # Get the value at the given step.
  if sequence.len == 0:
    return default(T)
  sequence[step.clamp(0, sequence.len - 1)]

proc at*[T](sequence: seq[T]): T =
  # Get the value at the current step.
  sequence.at(step)
