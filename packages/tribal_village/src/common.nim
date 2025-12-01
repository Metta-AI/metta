when defined(emscripten):
  import windy/platforms/emscripten/emdefs
else:
  import std/[times]

import
  boxy, windy, vmath

type
  IRect* = object
    x*: int
    y*: int
    w*: int
    h*: int

  PanelType* = enum
    WorldMap
    GlobalFooter

  Panel* = ref object
    panelType*: PanelType
    rect*: IRect
    name*: string

    pos*: Vec2
    vel*: Vec2
    zoom*: float32 = 10
    zoomVel*: float32
    minZoom*: float32 = 3.5
    maxZoom*: float32 = 100
    scrollArea*: Rect
    hasMouse*: bool = false
    visible*: bool = true
    focused*: bool = false

  AreaLayout* = enum
    Horizontal
    Vertical

  Area* = ref object
    layout*: AreaLayout
    rect*: IRect
    areas*: seq[Area]
    selectedPanelNum*: int
    panels*: seq[Panel]
    minSize*: float32 = 100.0
    resizable*: bool = true


  Settings* = object
    showFogOfWar* = false
    showVisualRange* = true
    showGrid* = true
    showObservations* = -1
    debugMode* = false
    showPerformanceStats* = false
    enableLogging* = true

proc nowSeconds*(): float64 =
  when defined(emscripten):
    emscripten_get_now() / 1000.0
  else:
    epochTime()

var
  window*: Window
  rootArea*: Area
  bxy*: Boxy
  frame*: int


  worldMapPanel*: Panel
  globalFooterPanel*: Panel

  settings* = Settings()

  play*: bool
  playSpeed*: float32 = 0.0625  # default to former middle speed (4x)
  lastSimTime*: float64 = nowSeconds()

const
  DefaultPlaySpeed* = 0.0625

var
  followSelection*: bool = false
  mouseCaptured*: bool = false
  mouseCapturedPanel*: Panel = nil
  mouseDownPos*: Vec2 = vec2(0, 0)

proc logicalMousePos*(window: Window): Vec2 =
  ## Mouse position in logical coordinates (accounts for HiDPI scaling).
  window.mousePos.vec2 / window.contentScale

proc logicalMouseDelta*(window: Window): Vec2 =
  ## Mouse delta in logical coordinates (accounts for HiDPI scaling).
  window.mouseDelta.vec2 / window.contentScale

proc manhattanDistance*(a, b: IVec2): int =
  abs(a.x - b.x) + abs(a.y - b.y)

proc irect*(x, y, w, h: int): IRect =
  ## Utility function to create IRect from coordinates
  result.x = x
  result.y = y
  result.w = w
  result.h = h

proc irect*(rect: Rect): IRect =
  ## Convert floating point Rect to integer IRect
  result.x = rect.x.int
  result.y = rect.y.int
  result.w = rect.w.int
  result.h = rect.h.int

proc rect*(irect: IRect): Rect =
  ## Convert integer IRect to floating point Rect
  result.x = irect.x.float32
  result.y = irect.y.float32
  result.w = irect.w.float32
  result.h = irect.h.float32

proc euclideanDistance*(a, b: IVec2): float =
  let dx = (a.x - b.x).float
  let dy = (a.y - b.y).float
  sqrt(dx * dx + dy * dy)

type
  OrientationDelta* = tuple[x, y: int]
  Orientation* = enum
    N = 0  # North (Up)
    S = 1  # South (Down) 
    W = 2  # West (Left)
    E = 3  # East (Right)
    NW = 4 # Northwest (Up-Left)
    NE = 5 # Northeast (Up-Right)
    SW = 6 # Southwest (Down-Left)
    SE = 7 # Southeast (Down-Right)

const OrientationDeltas*: array[8, OrientationDelta] = [
  (x: 0, y: -1),   # N (North)
  (x: 0, y: 1),    # S (South)
  (x: -1, y: 0),   # W (West)
  (x: 1, y: 0),    # E (East)
  (x: -1, y: -1),  # NW (Northwest)
  (x: 1, y: -1),   # NE (Northeast)
  (x: -1, y: 1),   # SW (Southwest)
  (x: 1, y: 1)     # SE (Southeast)
]

const
  ActionVerbCount* = 7
  ActionArgumentCount* = 8

proc encodeAction*(verb: uint8, argument: uint8): uint8 =
  (verb.int * ActionArgumentCount + argument.int).uint8

proc decodeAction*(value: uint8): tuple[verb: uint8, argument: uint8] =
  (verb: (value.int div ActionArgumentCount).uint8, argument: (value.int mod ActionArgumentCount).uint8)

{.push inline.}
proc getOrientationDelta*(orient: Orientation): OrientationDelta =
  OrientationDeltas[ord(orient)]
{.pop.}

proc isDiagonal*(orient: Orientation): bool =
  ord(orient) >= ord(NW)

proc getOpposite*(orient: Orientation): Orientation =
  case orient
  of N: S
  of S: N
  of W: E
  of E: W
  of NW: SE
  of NE: SW
  of SW: NE
  of SE: NW

proc orientationToVec*(orientation: Orientation): IVec2 =
  case orientation
  of N: result = ivec2(0, -1)
  of S: result = ivec2(0, 1)
  of E: result = ivec2(1, 0)
  of W: result = ivec2(-1, 0)
  of NW: result = ivec2(-1, -1)
  of NE: result = ivec2(1, -1)
  of SW: result = ivec2(-1, 1)
  of SE: result = ivec2(1, 1)

proc ivec2*(x, y: int): IVec2 =
  result.x = x.int32
  result.y = y.int32

proc getDirectionTo*(fromPos, toPos: IVec2): IVec2 =
  let dx = toPos.x - fromPos.x
  let dy = toPos.y - fromPos.y
  
  result.x = if dx > 0: 1 elif dx < 0: -1 else: 0
  result.y = if dy > 0: 1 elif dy < 0: -1 else: 0

proc relativeLocation*(orientation: Orientation, distance, offset: int): IVec2 =
  case orientation
  of N: ivec2(-offset, -distance)
  of S: ivec2(offset, distance)
  of E: ivec2(distance, -offset)
  of W: ivec2(-distance, offset)
  of NW: ivec2(-distance - offset, -distance + offset)
  of NE: ivec2(distance - offset, -distance - offset)
  of SW: ivec2(-distance + offset, distance + offset)
  of SE: ivec2(distance + offset, distance - offset)
