import { Vec2f } from './vector_math.js';
import { Context3d } from './context3d.js';
import { PanelInfo } from './panels.js';
import { onFrame } from './main.js';

// The 3d context, used for nearly everything.
export const ctx = new Context3d(document.getElementById('global-canvas') as HTMLCanvasElement);

// Constants
export const MIN_ZOOM_LEVEL = 0.025;
export const MAX_ZOOM_LEVEL = 2.0;

export const SPLIT_DRAG_THRESHOLD = 10;  // pixels to detect split dragging
export const SCROLL_ZOOM_FACTOR = 1000;  // divisor for scroll delta to zoom conversion
export const DEFAULT_TRACE_SPLIT = 0.80;  // default horizontal split ratio
export const DEFAULT_INFO_SPLIT = 0.25;   // default vertical split ratio
export const PANEL_BOTTOM_MARGIN = 60;    // bottom margin for panels

// Map constants
export const TILE_SIZE = 200;
export const INVENTORY_PADDING = 16;

// Agent defaults
export const DEFAULT_VISION_SIZE = 11;

// Trace constants
export const TRACE_HEIGHT = 256;
export const TRACE_WIDTH = 32;

// Colors for resources
export const COLORS: [string, [number, number, number, number]][] = [
  ["red", [1, 0, 0, 1]],
  ["green", [0, 1, 0, 1]],
  ["blue", [0, 0, 1, 1]],
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
  infoSplit: DEFAULT_INFO_SPLIT,
  infoDragging: false,

  // Panels
  mapPanel: new PanelInfo("map"),
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
