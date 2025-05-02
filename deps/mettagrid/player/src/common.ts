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
