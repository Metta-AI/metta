import { Vec2f } from './vector_math.js';
import { Context3d, parseHtmlColor } from './context3d.js';
import { PanelInfo } from './panels.js';

// The 3d context, used for nearly everything.
export const ctx = new Context3d(document.getElementById('global-canvas') as HTMLCanvasElement);

// Constants
export const MIN_ZOOM_LEVEL = 0.025;
export const MAX_ZOOM_LEVEL = 2.0;

export const SPLIT_DRAG_THRESHOLD = 10;  // pixels to detect split dragging
export const SCROLL_ZOOM_FACTOR = 1000;  // divisor for scroll delta to zoom conversion
export const DEFAULT_TRACE_SPLIT = 0.75;  // default horizontal split ratio
export const PANEL_BOTTOM_MARGIN = 60;    // bottom margin for panels
export const HEADER_HEIGHT = 60;          // height of the header
export const SCRUBBER_HEIGHT = 120;        // height of the scrubber

// Map constants
export const TILE_SIZE = 200;
export const INVENTORY_PADDING = 16;
export const MINI_MAP_TILE_SIZE = 2;

// Agent defaults
export const DEFAULT_VISION_SIZE = 11;

// Trace constants
export const TRACE_HEIGHT = 256;
export const TRACE_WIDTH = 32;

// Colors for resources
export const COLORS: [string, [number, number, number, number]][] = [
  ["red", parseHtmlColor("#E4433A")],
  ["green", parseHtmlColor("#66BB6A")],
  ["blue", parseHtmlColor("#3498DB")],
];

export const ui = {
  // Mouse events
  mouseDown: false,
  mouseClick: false,
  mouseDoubleClick: false,
  mousePos: new Vec2f(0, 0),
  lastMousePos: new Vec2f(0, 0),
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
  followTraceSelection: false, // Flag to follow trace selection

  // Playback state
  step: 0,
  isPlaying: false,
  partialStep: 0,
  playbackSpeed: 0.1,
};

export const html = {
  scrubber: document.getElementById('main-scrubber') as HTMLInputElement,
  playButton: document.getElementById('play-button') as HTMLButtonElement,
  globalCanvas: document.getElementById('global-canvas') as HTMLCanvasElement,
  fileName: document.getElementById('file-name') as HTMLDivElement,
  shareButton: document.getElementById('share-button') as HTMLButtonElement,
  mainFilter: document.getElementById('main-filter') as HTMLInputElement,
  toast: document.getElementById('toast') as HTMLDivElement,
}


// Get the modal element
const modal = document.getElementById('modal');

// Show the modal
export function showModal(type: string, title: string, message: string) {
  if (modal) {
    modal.style.display = 'block';
    modal.classList.add(type);
    const header = modal.querySelector('h2');
    if (header) {
      header.textContent = title;
    }
    const content = modal.querySelector('p');
    if (content) {
      content.textContent = message;
    }
  }
}

// Close the modal
export function closeModal() {
  if (modal) {
    // Remove error class from modal.
    modal.classList.remove('error');
    modal.classList.remove('info');
    modal.style.display = 'none';
  }
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
