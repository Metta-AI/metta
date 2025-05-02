import { Vec2f } from './vector_math.js';

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
