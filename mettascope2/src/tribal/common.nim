import std/[times],
  boxy, windy, vmath

type
  IRect* = object
    x*: int
    y*: int
    w*: int
    h*: int

  PanelType* = enum
    WorldMap
    Minimap
    AgentTable
    AgentTraces
    GlobalHeader
    GlobalFooter
    GlobalTimeline

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
  window*: Window
  rootArea*: Area
  bxy*: Boxy
  frame*: int


  worldMapPanel*: Panel
  minimapPanel*: Panel
  agentTablePanel*: Panel
  agentTracesPanel*: Panel
  globalTimelinePanel*: Panel
  globalFooterPanel*: Panel
  globalHeaderPanel*: Panel

  settings* = Settings()

  play*: bool
  playSpeed*: float32 = 1/60.0
  lastSimTime*: float64 = epochTime()

  followSelection*: bool = false
  mouseCaptured*: bool = false
  mouseCapturedPanel*: Panel = nil
