import { Context3d } from './context3d.js'
import type { HoverBubble } from './hoverbubbles.js'
import { find, localStorageGetNumber, parseHtmlColor, toggleOpacity } from './htmlutils.js'
import { PanelInfo } from './panels.js'
import { Vec2f } from './vector_math.js'

// The 3D context, used for nearly everything.
export const ctx = new Context3d(find('#global-canvas') as HTMLCanvasElement)
;(window as any).ctx = ctx

// Constants
export const MIN_ZOOM_LEVEL = 0.025
export const MAX_ZOOM_LEVEL = 2.5
export const DEFAULT_ZOOM_LEVEL = 1 / 3
export const DEFAULT_TRACE_ZOOM_LEVEL = 1 / 4
export const SPLIT_DRAG_THRESHOLD = 10 // Pixels to detect split dragging.
export const ZOOM_SENSITIVITY = 0.003 // Controls zoom speed - smaller = slower, smoother zoom
export const MACROMAP_ZOOM_THRESHOLD = 0.05 // when to switch the map to 'macromap' minimap-style rendering
export const PANEL_BOTTOM_MARGIN = 60
export const HEADER_HEIGHT = 60
export const FOOTER_HEIGHT = 128
export const SPEEDS = [0.02, 0.1, 0.25, 0.5, 1.0, 5.0]

// GitHub constants - keep in sync with metta/common/src/metta/common/util/constants.py
export const METTA_GITHUB_ORGANIZATION = 'Metta-AI'
export const METTA_GITHUB_REPO = 'metta'
export const METTA_GITHUB_PRIMARY_BRANCH = 'main'

// Map constants
export const TILE_SIZE = 200
export const INVENTORY_PADDING = 16
export const MINI_MAP_TILE_SIZE = 2

// Agent defaults
export const DEFAULT_VISION_SIZE = 11

// Trace constants
export const TRACE_HEIGHT = 512
export const TRACE_WIDTH = 54

// Info panel constants
export const INFO_PANEL_POP_TIME = 300 // ms

// Colors for resources
export const COLORS: [string, [number, number, number, number]][] = [
  ['red', parseHtmlColor('#E4433A')],
  ['green', parseHtmlColor('#66BB6A')],
  ['blue', parseHtmlColor('#3498DB')],
]

export const ui = {
  // Mouse events
  mouseDown: false,
  mouseUp: false,
  mouseClick: false,
  mouseDoubleClick: false,
  mousePos: new Vec2f(0, 0),
  mouseTargets: [] as string[],
  dragging: '',
  dragHtml: null as HTMLElement | null,
  dragOffset: new Vec2f(0, 0),
  lastMousePos: new Vec2f(0, 0),
  mouseDownPos: new Vec2f(0, 0),
  scrollDelta: 0,
  lastClickTime: 0, // For double-click detection.
  mainScrubberDown: false,
  mainTraceMinimapDown: false,

  // Touch events for pinch-to-zoom
  touches: [] as Touch[],
  lastTouches: [] as Touch[],
  pinchDistance: 0,
  lastPinchDistance: 0,
  pinchCenter: new Vec2f(0, 0),
  lastPinchCenter: new Vec2f(0, 0),
  isPinching: false,

  dpr: 1, // DPI scale factor used for Retina displays.

  // Split between trace and info panels.
  traceSplit: localStorageGetNumber('traceSplit', 0.8),
  agentPanelSplit: localStorageGetNumber('agentPanelSplit', 0.5),

  // Panels
  mapPanel: new PanelInfo('#worldmap-panel'),
  miniMapPanel: new PanelInfo('#minimap-panel'),
  tracePanel: new PanelInfo('#trace-panel'),
  infoPanel: new PanelInfo('#info-panel'),
  agentPanel: new PanelInfo('#agent-panel'),
  timelinePanel: new PanelInfo('#timeline-panel'),

  hoverBubbles: [] as HoverBubble[],
  hoverObject: null as any,
  hoverTimer: null as any,
  delayedHoverObject: null as any,
  hideHoverTimer: null as any,
}

export const state = {
  // Replay data and player state
  replay: null as any,
  selectedGridObject: null as any,
  followSelection: false, // Flag to follow the selected entity.

  // Playback state
  step: 0,
  isPlaying: false,
  partialStep: 0,
  playbackSpeed: 0.1,
  demoMode: false,
  fullScreen: false,

  // What to show?
  showUi: true,
  showResources: true,
  showGrid: true,
  showVisualRanges: true,
  showFogOfWar: false,
  showMiniMap: false,
  showInfo: false,
  showTraces: true,
  showActionButtons: false,
  showAgentPanel: false,
  showAttackMode: false,
  showGlyphEditor: false,

  // Playing over a WebSocket
  ws: null as WebSocket | null,
  isOneToOneAction: false,
}

// Expose state for easier testing
;(window as any).state = state

export const html = {
  globalCanvas: find('#global-canvas') as HTMLCanvasElement,

  // Header area
  fileName: find('#file-name'),
  helpButton: find('#help-button'),
  shareButton: find('#share-button'),

  rewindToStartButton: find('#rewind-to-start'),
  stepBackButton: find('#step-back'),
  playButton: find('#play'),
  stepForwardButton: find('#step-forward'),
  rewindToEndButton: find('#rewind-to-end'),
  demoModeToggle: find('#demo-mode-toggle'),
  fullScreenToggle: find('#full-screen-toggle'),
  tracesToggle: find('#traces-toggle'),
  glyphToggle: find('#glyph-toggle'),

  actionButtons: find('#action-buttons'),

  speedButtons: [find('#speed1'), find('#speed2'), find('#speed3'), find('#speed4'), find('#speed5'), find('#speed6')],

  focusToggle: find('#focus-toggle'),

  minimapToggle: find('#minimap-toggle'),
  controlsToggle: find('#controls-toggle'),
  infoToggle: find('#info-toggle'),
  agentPanelToggle: find('#agent-panel-toggle'),

  resourcesToggle: find('#resources-toggle'),
  gridToggle: find('#grid-toggle'),
  visualRangeToggle: find('#visual-range-toggle'),
  fogOfWarToggle: find('#fog-of-war-toggle'),

  stepCounter: find('#step-counter'),

  // Utility
  modal: find('#modal'),
  toast: find('#toast'),
}

/** Generates a color from an agent ID. */
export function colorFromId(agentId: number) {
  const n = agentId + Math.PI + Math.E + Math.SQRT2
  return [(n * Math.PI) % 1.0, (n * Math.E) % 1.0, (n * Math.SQRT2) % 1.0, 1.0]
}

/** Sets the follow selection state. You can pass null to leave a state unchanged. */
export function setFollowSelection(map: boolean | null) {
  if (map != null) {
    state.followSelection = map
    toggleOpacity(html.focusToggle, map)
  }
}

/** Shows the modal. */
export function showModal(type: string, title: string, message: string) {
  html.modal.classList.remove('hidden')
  html.modal.classList.add(type)
  const header = html.modal.querySelector('.header')
  if (header) {
    header.textContent = title
  }
  const content = html.modal.querySelector('.message')
  if (content) {
    content.textContent = message
  }
}

/** Closes the modal. */
export function closeModal() {
  // Remove error class from modal.
  html.modal.classList.remove('error')
  html.modal.classList.remove('info')
  html.modal.classList.add('hidden')
}

/** Functions to show and hide toast notifications. */
export function showToast(message: string, duration = 3000) {
  // Set the message
  let msg = html.toast.querySelector('.message')
  if (msg != null) {
    msg.textContent = message
  }
  // Remove any existing classes
  html.toast.classList.remove('hiding')
  // Make the toast visible
  html.toast.classList.add('visible')
  // Set a timeout to hide the toast after the specified duration
  setTimeout(() => {
    hideToast()
  }, duration)
}

/** Hides the currently visible toast with an upward animation. */
export function hideToast() {
  // Add the hiding class for the upward animation
  html.toast.classList.add('hiding')
  // Remove the visible class after the animation completes
  setTimeout(() => {
    html.toast.classList.remove('visible')
    html.toast.classList.remove('hiding')
  }, 300) // Match the transition duration from CSS
}
