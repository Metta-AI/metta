import std/[times],
  sim, boxy, windy, vmath

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
    showObservations* = -1
    lockFocus* = false

var
  window*: Window
  rootArea*: Area
  bxy*: Boxy
  frame*: int
  env*: Environment


  worldMapPanel*: Panel
  minimapPanel*: Panel
  agentTablePanel*: Panel
  agentTracesPanel*: Panel
  globalTimelinePanel*: Panel
  globalFooterPanel*: Panel
  globalHeaderPanel*: Panel

  settings* = Settings()

  selection*: Thing

  play*: bool
  playSpeed*: float32 = 1/60.0
  lastSimTime*: float64 = epochTime()
