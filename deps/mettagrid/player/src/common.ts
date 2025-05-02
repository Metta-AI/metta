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

// Interaction state
export let mouseDown = false;
export let mouseClick = false;
export let mouseDoubleClick = false;
export let mousePos = new Vec2f(0, 0);
export let lastMousePos = new Vec2f(0, 0);
export let scrollDelta = 0;
export let lastClickTime = 0; // For double-click detection

// Panel split state
export let traceSplit = DEFAULT_TRACE_SPLIT;
export let traceDragging = false;
export let infoSplit = DEFAULT_INFO_SPLIT;
export let infoDragging = false;

// Replay data and player state
export let replay: any = null;
export let selectedGridObject: any = null;
export let followSelection = false; // Flag to follow selected entity
export let followTraceSelection = false; // Flag to follow trace selection

// Playback state
export let step = 0;
export let isPlaying = false;
export let partialStep = 0;
export let playbackSpeed = 0.1;
