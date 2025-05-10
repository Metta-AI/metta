import { Vec2f } from './vector_math.js';
import { Context3d } from './context3d.js';
import { find, parseHtmlColor } from './htmlutils.js';
import { PanelInfo } from './panels.js';

// The 3d context, used for nearly everything.
export const ctx = new Context3d(find('#global-canvas') as HTMLCanvasElement);

// Constants
export const MIN_ZOOM_LEVEL = 0.025;
export const MAX_ZOOM_LEVEL = 2.0;
export const DEFAULT_ZOOM_LEVEL = 1 / 2;
export const DEFAULT_TRACE_ZOOM_LEVEL = 1 / 4;
export const SPLIT_DRAG_THRESHOLD = 10;  // pixels to detect split dragging
export const SCROLL_ZOOM_FACTOR = 1000;  // divisor for scroll delta to zoom conversion
export const DEFAULT_TRACE_SPLIT = 0.80;  // default horizontal split ratio
export const PANEL_BOTTOM_MARGIN = 60;    // bottom margin for panels
export const HEADER_HEIGHT = 60;          // height of the header
export const SCRUBBER_HEIGHT = 120;        // height of the scrubber
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
  lastMousePos: new Vec2f(0, 0),
  mouseDownPos: new Vec2f(0, 0),
  scrollDelta: 0,
  lastClickTime: 0, // For double-click detection

  // Split between trace and info panels.
  traceSplit: DEFAULT_TRACE_SPLIT,
  traceDragging: false,

  // Panels
  mapPanel: new PanelInfo("map"),
  miniMapPanel: new PanelInfo("mini-map"),
  tracePanel: new PanelInfo("trace"),
  infoPanel: new PanelInfo("info"),
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
  sortTraces: false,
  showResources: true,
  showGrid: true,
  showViewRanges: true,
  showFogOfWar: false,
};

export const html = {
  globalCanvas: find('#global-canvas') as HTMLCanvasElement,

  // Header area
  fileName: find('#file-name') as HTMLDivElement,
  shareButton: find('#share-button') as HTMLButtonElement,
  mainFilter: find('#main-filter') as HTMLInputElement,

  // Bottom area
  scrubber: find('#main-scrubber') as HTMLInputElement,

  rewindToStartButton: find('#rewind-to-start') as HTMLImageElement,
  stepBackButton: find('#step-back') as HTMLImageElement,
  playButton: find('#play') as HTMLButtonElement,
  stepForwardButton: find('#step-forward') as HTMLImageElement,
  rewindToEndButton: find('#rewind-to-end') as HTMLImageElement,

  speedButtons: [
    find('#speed1') as HTMLImageElement,
    find('#speed2') as HTMLImageElement,
    find('#speed3') as HTMLImageElement,
    find('#speed4') as HTMLImageElement,
    find('#speed5') as HTMLImageElement,
    find('#speed6') as HTMLImageElement,
  ],

  sortButton: find('#sort') as HTMLImageElement,
  resourcesButton: find('#resources') as HTMLImageElement,
  focusButton: find('#tack') as HTMLImageElement,
  gridButton: find('#grid') as HTMLImageElement,
  showViewButton: find('#eye') as HTMLImageElement,
  showFogOfWarButton: find('#cloud') as HTMLImageElement,

  // Utility
  modal: find('#modal') as HTMLDivElement,
  toast: find('#toast') as HTMLDivElement,
}

// Set the follow selection state, you can pass null to leave a state unchanged.
export function setFollowSelection(map: boolean | null) {
  if (map != null) {
    state.followSelection = map;
    if (map) {
      html.focusButton.style.opacity = "1";
    } else {
      html.focusButton.style.opacity = "0.2";
    }
  }
}

// Show the modal
export function showModal(type: string, title: string, message: string) {
  html.modal.style.display = 'block';
  html.modal.classList.add(type);
  const header = html.modal.querySelector('h2');
  if (header) {
    header.textContent = title;
  }
  const content = html.modal.querySelector('p');
  if (content) {
    content.textContent = message;
  }
}

// Close the modal
export function closeModal() {
  // Remove error class from modal.
  html.modal.classList.remove('error');
  html.modal.classList.remove('info');
  html.modal.style.display = 'none';
}

// Functions to show and hide toast notifications
export function showToast(message: string, duration = 3000) {
  // Set the message
  html.toast.textContent = message;
  // Remove any existing classes
  html.toast.classList.remove('hiding');
  // Make the toast visible
  html.toast.classList.add('visible');
  // Set a timeout to hide the toast after the specified duration
  setTimeout(() => {
    hideToast();
  }, duration);
}

// Hides the currently visible toast with an upward animation
export function hideToast() {
  // Add the hiding class for the upward animation
  html.toast.classList.add('hiding');
  // Remove the visible class after the animation completes
  setTimeout(() => {
    html.toast.classList.remove('visible');
    html.toast.classList.remove('hiding');
  }, 300); // Match the transition duration from CSS
}
