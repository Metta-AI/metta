import { Vec2f, Mat3f } from './vector_math.js';
import { Context3d } from './context3d.js';
import * as Common from './common.js';
import { ui, state, html, ctx } from './common.js';
import { fetchReplay, readFile } from './replay.js';
import { focusFullMap, updateReadout, drawMap, drawTrace, requestFrame } from './drawing.js';
import { PanelInfo } from './panels.js';

// Handle resize events.
function onResize() {
  // Adjust for high DPI displays.
  const dpr = window.devicePixelRatio || 1;

  const mapWidth = window.innerWidth;
  const mapHeight = window.innerHeight;

  // Make sure traceSplit and infoSplit are not too small or too large.
  const a = 0.025;
  ui.traceSplit = Math.max(a, Math.min(ui.traceSplit, 1 - a));
  ui.infoSplit = Math.max(a, Math.min(ui.infoSplit, 1 - a));

  ui.mapPanel.x = 0;
  ui.mapPanel.y = 0;
  ui.mapPanel.width = mapWidth * ui.traceSplit;
  ui.mapPanel.height = mapHeight - Common.PANEL_BOTTOM_MARGIN;

  ui.tracePanel.x = mapWidth * ui.traceSplit;
  ui.tracePanel.y = mapHeight * ui.infoSplit;
  ui.tracePanel.width = mapWidth * (1 - ui.traceSplit);
  ui.tracePanel.height = mapHeight * (1 - ui.infoSplit) - Common.PANEL_BOTTOM_MARGIN;

  ui.infoPanel.x = mapWidth * ui.traceSplit;
  ui.infoPanel.y = 0;
  ui.infoPanel.width = mapWidth * (1 - ui.traceSplit);
  ui.infoPanel.height = mapHeight * ui.infoSplit;
  if (ui.infoPanel.div === null) {
    ui.infoPanel.div = document.createElement("div");
    ui.infoPanel.div.id = ui.infoPanel.name + "-div";
    document.body.appendChild(ui.infoPanel.div);
  }
  if (ui.infoPanel.div !== null) {
    const div = ui.infoPanel.div;
    div.style.position = 'absolute';
    div.style.top = ui.infoPanel.y + 'px';
    div.style.left = ui.infoPanel.x + 'px';
    div.style.width = ui.infoPanel.width + 'px';
    div.style.height = ui.infoPanel.height + 'px';
  }

  // Redraw the square after resizing.
  requestFrame();
}

// Handle mouse down events.
function onMouseDown() {
  ui.lastMousePos = ui.mousePos;
  ui.mouseClick = true;
  const currentTime = new Date().getTime();
  ui.mouseDoubleClick = currentTime - ui.lastClickTime < 300; // 300ms threshold for double-click
  ui.lastClickTime = currentTime;

  if (Math.abs(ui.mousePos.x() - ui.mapPanel.width) < Common.SPLIT_DRAG_THRESHOLD) {
    ui.traceDragging = true
    console.log("Started trace dragging")
  } else if (ui.mousePos.x() > ui.mapPanel.width && Math.abs(ui.mousePos.y() - ui.infoPanel.height) < Common.SPLIT_DRAG_THRESHOLD) {
    ui.infoDragging = true
    console.log("Started info dragging")
  } else {
    ui.mouseDown = true;
  }

  requestFrame();
}

// Handle mouse up events.
function onMouseUp() {
  ui.mouseDown = false;
  ui.traceDragging = false;
  ui.infoDragging = false;
  requestFrame();
}

// Handle mouse move events.
function onMouseMove(event: MouseEvent) {
  ui.mousePos = new Vec2f(event.clientX, event.clientY);

  // If mouse is close to a panels edge change cursor to edge changer.
  document.body.style.cursor = "default";

  if (Math.abs(ui.mousePos.x() - ui.mapPanel.width) < Common.SPLIT_DRAG_THRESHOLD) {
    document.body.style.cursor = "ew-resize";
  }

  if (ui.mousePos.x() > ui.mapPanel.width &&
    Math.abs(ui.mousePos.y() - ui.infoPanel.height) < Common.SPLIT_DRAG_THRESHOLD
  ) {
    document.body.style.cursor = "ns-resize";
  }

  if (ui.traceDragging) {
    ui.traceSplit = ui.mousePos.x() / window.innerWidth
    onResize();
  } else if (ui.infoDragging) {
    ui.infoSplit = ui.mousePos.y() / window.innerHeight
    onResize()
  }
  requestFrame();
}

// Handle scroll events.
function onScroll(event: WheelEvent) {
  ui.scrollDelta = event.deltaY;
  requestFrame();
}


// Update all URL parameters without creating browser history entries
function updateUrlParams() {
  // Get current URL params
  const urlParams = new URLSearchParams(window.location.search);

  // Update step when its not zero:
  if (state.step !== 0) {
    urlParams.set('step', state.step.toString());
  } else {
    urlParams.delete('step');
  }

  // Handle selected object
  if (state.selectedGridObject !== null) {
    // Find the index of the selected object
    const selectedObjectIndex = state.replay.grid_objects.indexOf(state.selectedGridObject);
    if (selectedObjectIndex !== -1) {
      urlParams.set('selectedObjectId', (selectedObjectIndex + 1).toString());
      // Remove map position parameters when an object is selected
      urlParams.delete('mapPanX');
      urlParams.delete('mapPanY');
    }
  } else {
    // Include map position
    urlParams.set('mapPanX', Math.round(ui.mapPanel.panPos.x()).toString());
    urlParams.set('mapPanY', Math.round(ui.mapPanel.panPos.y()).toString());
    // Remove selected object when there is no selection
    urlParams.delete('selectedObjectId');
  }

  // Include map zoom level
  if (ui.mapPanel.zoomLevel != 1) {
    // Only include zoom to 3 decimal places.
    urlParams.set('mapZoom', ui.mapPanel.zoomLevel.toFixed(3));
  }

  // Handle play state - only include when true
  if (state.isPlaying) {
    urlParams.set('play', 'true');
  } else {
    urlParams.delete('play');
  }

  // Replace current state without creating history entry
  const newUrl = window.location.pathname + '?' + urlParams.toString();
  history.replaceState(null, '', newUrl);
}

// Centralized function to update the step and handle all related updates
export function updateStep(newStep: number, skipScrubberUpdate = false) {
  // Update the step variable
  state.step = newStep;

  // Update the scrubber value (unless told to skip)
  if (!skipScrubberUpdate) {
    html.scrubber.value = state.step.toString();
  }

  // Update trace panel position
  ui.tracePanel.panPos.setX(-state.step * 32);

  // Request a new frame
  requestFrame();
}

// Handle scrubber change events.
function onScrubberChange() {
  updateStep(parseInt(html.scrubber.value), true);
}

// Handle key down events.
function onKeyDown(event: KeyboardEvent) {
  if (event.key == "Escape") {
    state.selectedGridObject = null;
    state.followSelection = false; // Also stop following when selection is cleared
    state.followTraceSelection = false;
  }
  // '[' and ']' to scrub forward and backward.
  if (event.key == "[") {
    updateStep(Math.max(state.step - 1, 0));
  }
  if (event.key == "]") {
    updateStep(Math.min(state.step + 1, state.replay.max_steps - 1));
  }
  // '<' and '>' control the playback speed.
  if (event.key == ",") {
    state.playbackSpeed = Math.max(state.playbackSpeed * 0.9, 0.01);
    console.log("playbackSpeed: ", state.playbackSpeed);
  }
  if (event.key == ".") {
    state.playbackSpeed = Math.min(state.playbackSpeed * 1.1, 1000);
    console.log("playbackSpeed: ", state.playbackSpeed);
  }
  // If space make it press the play button.
  if (event.key == " ") {
    onPlayButtonClick();
  }
  requestFrame();
}

// Draw a frame.
export function onFrame() {
  if (state.replay === null || ctx === null || ctx.ready === false) {
    return;
  }

  // Make sure the canvas is the size of the window.
  html.globalCanvas.width = window.innerWidth;
  html.globalCanvas.height = window.innerHeight;

  ctx.clear();

  var fullUpdate = true;
  if (ui.mapPanel.inside(ui.mousePos)) {
    if (ui.mapPanel.updatePanAndZoom()) {
      fullUpdate = false;
    }
  }

  if (ui.tracePanel.inside(ui.mousePos)) {
    if (ui.tracePanel.updatePanAndZoom()) {
      fullUpdate = false;
    }
  }

  updateReadout();
  ctx.useMesh("map");
  drawMap(ui.mapPanel);
  ctx.useMesh("trace");
  drawTrace(ui.tracePanel);

  ctx.flush();
  console.log("Flushed ctx.");

  // Update URL parameters with current state once per frame
  updateUrlParams();

  if (state.isPlaying) {
    state.partialStep += state.playbackSpeed;
    if (state.partialStep >= 1) {
      const nextStep = (state.step + Math.floor(state.partialStep)) % state.replay.max_steps;
      state.partialStep -= Math.floor(state.partialStep);
      updateStep(nextStep);
    }
    requestFrame();
  }

  ui.mouseClick = false;
  ui.mouseDoubleClick = false;
}

function preventDefaults(event: Event) {
  event.preventDefault();
  event.stopPropagation();
}

function handleDrop(event: DragEvent) {
  event.preventDefault();
  event.stopPropagation();
  const dt = event.dataTransfer;
  if (dt && dt.files.length) {
    const file = dt.files[0];
    readFile(file);
  }
}

// Handle play button click
function onPlayButtonClick() {
  state.isPlaying = !state.isPlaying;

  if (state.isPlaying) {
    html.playButton.classList.add('paused');
  } else {
    html.playButton.classList.remove('paused');
  }

  requestFrame();
}

// Parse URL parameters, and modify the map and trace panels accordingly.
async function parseUrlParams() {

  const urlParams = new URLSearchParams(window.location.search);

    // Load the replay.
  const replayUrl = urlParams.get('replayUrl');
  if (replayUrl) {
    console.log("Loading replay from URL: ", replayUrl);
    await fetchReplay(replayUrl);
    focusFullMap(ui.mapPanel);
  } else {
    Common.showModal(
      "info",
      "Welcome to MettaScope",
      "Please drop a replay file here to see the replay."
    );
  }

  // Set the current step.
  if (urlParams.get('step') !== null) {
    const initialStep = parseInt(urlParams.get('step') || "0");
    console.info("Step via query parameter:", initialStep);
    updateStep(initialStep, false);
  }

  // Set the playing state.
  if (urlParams.get('play') !== null) {
    state.isPlaying = urlParams.get('play') === "true";
    console.info("Playing state via query parameter:", state.isPlaying);
  }

  // Set selected object.
  if (urlParams.get('selectedObjectId') !== null) {
    const selectedObjectId = parseInt(urlParams.get('selectedObjectId') || "-1") - 1;
    if (selectedObjectId >= 0 && selectedObjectId < state.replay.grid_objects.length) {
      state.selectedGridObject = state.replay.grid_objects[selectedObjectId];
      state.followSelection = true;
      ui.mapPanel.zoomLevel = 1/2;
      ui.tracePanel.zoomLevel = 1;
      console.info("Selected object via query parameter:", state.selectedGridObject);
    } else {
      console.warn("Invalid selectedObjectId:", selectedObjectId);
    }
  }

  // Set the map pan and zoom.
  if (urlParams.get('mapPanX') !== null && urlParams.get('mapPanY') !== null) {
    const mapPanX = parseInt(urlParams.get('mapPanX') || "0");
    const mapPanY = parseInt(urlParams.get('mapPanY') || "0");
    ui.mapPanel.panPos = new Vec2f(mapPanX, mapPanY);
  }
  if (urlParams.get('mapZoom') !== null) {
    ui.mapPanel.zoomLevel = parseFloat(urlParams.get('mapZoom') || "1");
  }

  requestFrame();
}

// Initial resize.
onResize();

// Add event listener to resize the canvas when the window is resized.
window.addEventListener('resize', onResize);
window.addEventListener('keydown', onKeyDown);
window.addEventListener('mousedown', onMouseDown);
window.addEventListener('mouseup', onMouseUp);
window.addEventListener('mousemove', onMouseMove);
window.addEventListener('wheel', onScroll);

html.scrubber.addEventListener('input', onScrubberChange);
html.playButton.addEventListener('click', onPlayButtonClick);

window.addEventListener('dragenter', preventDefaults, false);
window.addEventListener('dragleave', preventDefaults, false);
window.addEventListener('dragover', preventDefaults, false);
window.addEventListener('drop', handleDrop, false);

window.addEventListener('load', async () => {

  // Use local atlas texture.
  const atlasImageUrl = 'dist/atlas.png';
  const atlasJsonUrl = 'dist/atlas.json';

  const success = await ctx.init(atlasJsonUrl, atlasImageUrl);
  if (!success) {
    Common.showModal(
      "error",
      "Initialization failed",
      "Please check the console for more information."
    );
    return;
  } else {
    console.log("Context3d initialized successfully.");
  }

  await parseUrlParams();

  requestFrame();
});
