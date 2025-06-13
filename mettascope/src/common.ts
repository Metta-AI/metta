import { Vec2f } from './vector_math.js';
import { Context3d } from './context3d.js';
import { find, parseHtmlColor, localStorageGetNumber } from './htmlutils.js';
import { PanelInfo } from './panels.js';
import { InfoPanel } from './infopanels.js';

// The 3d context, used for nearly everything.
export const ctx = new Context3d(find('#global-canvas') as HTMLCanvasElement);

// Constants
export const MIN_ZOOM_LEVEL = 0.025;
export const MAX_ZOOM_LEVEL = 2.0;
export const DEFAULT_ZOOM_LEVEL = 1 / 2;
export const DEFAULT_TRACE_ZOOM_LEVEL = 1 / 4;
export const SPLIT_DRAG_THRESHOLD = 10;  // pixels to detect split dragging
export const SCROLL_ZOOM_FACTOR = 1000;  // divisor for scroll delta to zoom conversion
export const PANEL_BOTTOM_MARGIN = 60;
export const HEADER_HEIGHT = 60;
export const FOOTER_HEIGHT = 128;
export const SPEEDS = [0.02, 0.1, 0.25, 0.5, 1.0, 5.0];

// Map constants
export const TILE_SIZE = 200;
export const INVENTORY_PADDING = 16;
export const MINI_MAP_TILE_SIZE = 2;

// Agent defaults
export const DEFAULT_VISION_SIZE = 11;

// Trace constants
export const TRACE_HEIGHT = 512;
export const TRACE_WIDTH = 54;

// Info panel constants
export const INFO_PANEL_POP_TIME = 300; // ms

// Colors for resources
export const COLORS: [string, [number, number, number, number]][] = [
  ["red", parseHtmlColor("#E4433A")],
  ["green", parseHtmlColor("#66BB6A")],
  ["blue", parseHtmlColor("#3498DB")],
];

export const ui = {
  // Mouse events
  mouseDown: false,
  mouseUp: false,
  mouseClick: false,
  mouseDoubleClick: false,
  mousePos: new Vec2f(0, 0),
  mouseTarget: "",
  dragging: "",
  dragHtml: null as HTMLElement | null,
  dragOffset: new Vec2f(0, 0),
  lastMousePos: new Vec2f(0, 0),
  mouseDownPos: new Vec2f(0, 0),
  scrollDelta: 0,
  lastClickTime: 0, // For double-click detection
  mainScrubberDown: false,

  // Split between trace and info panels.
  traceSplit: localStorageGetNumber("traceSplit", 0.8),
  agentPanelSplit: localStorageGetNumber("agentPanelSplit", 0.5),

  // Panels
  mapPanel: new PanelInfo("#worldmap-panel"),
  miniMapPanel: new PanelInfo("#minimap-panel"),
  tracePanel: new PanelInfo("#trace-panel"),
  infoPanel: new PanelInfo("#info-panel"),
  agentPanel: new PanelInfo("#agent-panel"),
  timelinePanel: new PanelInfo("#timeline-panel"),

  infoPanels: [] as InfoPanel[],
  hoverObject: null as any,
  hoverTimer: null as any,
  delayedHoverObject: null as any,
};

export const state = {
  // Replay data and player state
  replay: null as any,
  selectedGridObject: null as any,
  followSelection: false, // Flag to follow selected entity

  // Playback state
  step: 0,
  isPlaying: false,
  partialStep: 0,
  playbackSpeed: 0.1,

  // What to show?
  showResources: true,
  showGrid: true,
  showVisualRanges: true,
  showFogOfWar: false,
  showMiniMap: true,
  showInfo: true,
  showControls: true,
  showAgentPanel: true,

  showAttackMode: false,

  // Playing over WebSocket
  ws: null as WebSocket | null,
  isOneToOneAction: false,
};

export const html = {
  globalCanvas: find('#global-canvas') as HTMLCanvasElement,

  // Header area
  fileName: find('#file-name') as HTMLDivElement,
  helpButton: find('#help-button') as HTMLButtonElement,
  shareButton: find('#share-button') as HTMLButtonElement,

  rewindToStartButton: find('#rewind-to-start') as HTMLImageElement,
  stepBackButton: find('#step-back') as HTMLImageElement,
  playButton: find('#play') as HTMLButtonElement,
  stepForwardButton: find('#step-forward') as HTMLImageElement,
  rewindToEndButton: find('#rewind-to-end') as HTMLImageElement,

  actionButtons: find('#action-buttons'),

  speedButtons: [
    find('#speed1') as HTMLImageElement,
    find('#speed2') as HTMLImageElement,
    find('#speed3') as HTMLImageElement,
    find('#speed4') as HTMLImageElement,
    find('#speed5') as HTMLImageElement,
    find('#speed6') as HTMLImageElement,
  ],

  focusToggle: find('#focus-toggle') as HTMLImageElement,

  minimapToggle: find('#minimap-toggle') as HTMLImageElement,
  controlsToggle: find('#controls-toggle') as HTMLImageElement,
  infoToggle: find('#info-toggle') as HTMLImageElement,
  agentPanelToggle: find('#agent-panel-toggle') as HTMLImageElement,

  resourcesToggle: find('#resources-toggle') as HTMLImageElement,
  gridToggle: find('#grid-toggle') as HTMLImageElement,
  visualRangeToggle: find('#visual-range-toggle') as HTMLImageElement,
  fogOfWarToggle: find('#fog-of-war-toggle') as HTMLImageElement,

  stepCounter: find('#step-counter') as HTMLSpanElement,

  // Utility
  modal: find('#modal') as HTMLDivElement,
  toast: find('#toast') as HTMLDivElement,
}

/** Set the follow selection state, you can pass null to leave a state unchanged. */
export function setFollowSelection(map: boolean | null) {
  if (map != null) {
    state.followSelection = map;
    if (map) {
      html.focusToggle.style.opacity = "1";
    } else {
      html.focusToggle.style.opacity = "0.2";
    }
  }
}

/** Show the modal. */
export function showModal(type: string, title: string, message: string) {
  html.modal.classList.remove('hidden');
  html.modal.classList.add(type);
  const header = html.modal.querySelector('.header');
  if (header) {
    header.textContent = title;
  }
  const content = html.modal.querySelector('.message');
  if (content) {
    content.textContent = message;
  }
}

/** Close the modal. */
export function closeModal() {
  // Remove error class from modal.
  html.modal.classList.remove('error');
  html.modal.classList.remove('info');
  html.modal.classList.add('hidden');
}

/** Functions to show and hide toast notifications. */
export function showToast(message: string, duration = 3000) {
  // Set the message
  let msg = html.toast.querySelector('.message')
  if (msg != null) {
    msg.textContent = message
  }
  // Remove any existing classes
  html.toast.classList.remove('hiding');
  // Make the toast visible
  html.toast.classList.add('visible');
  // Set a timeout to hide the toast after the specified duration
  setTimeout(() => {
    hideToast();
  }, duration);
}

/** Hides the currently visible toast with an upward animation. */
export function hideToast() {
  // Add the hiding class for the upward animation
  html.toast.classList.add('hiding');
  // Remove the visible class after the animation completes
  setTimeout(() => {
    html.toast.classList.remove('visible');
    html.toast.classList.remove('hiding');
  }, 300); // Match the transition duration from CSS
}
