import { Vec2f, Mat3f } from './vector_math.js';
import * as Common from './common.js';
import { ui, state, html, ctx, setFollowSelection } from './common.js';
import { fetchReplay, getAttr, initWebSocket, readFile, sendAction } from './replay.js';
import { focusFullMap, updateReadout, drawMap, requestFrame } from './worldmap.js';
import { drawTrace } from './traces.js';
import { drawMiniMap } from './minimap.js';
import { processActions, initActionButtons } from './actions.js';

// Handle resize events.
export function onResize() {
  // Adjust for high DPI displays.
  const dpr = window.devicePixelRatio || 1;

  const screenWidth = window.innerWidth;
  const screenHeight = window.innerHeight;

  // Make sure traceSplit and infoSplit are not too small or too large.
  const a = 0.025;
  ui.traceSplit = Math.max(a, Math.min(ui.traceSplit, 1 - a));

  ui.mapPanel.x = 0;
  ui.mapPanel.y = Common.HEADER_HEIGHT;
  ui.mapPanel.width = screenWidth;
  let maxMapHeight = screenHeight - Common.HEADER_HEIGHT - Common.SCRUBBER_HEIGHT;
  ui.mapPanel.height = Math.min(screenHeight * ui.traceSplit - Common.HEADER_HEIGHT, maxMapHeight);

  // Minimap goes in the bottom left corner of the mapPanel.
  if (state.replay != null) {
    const miniMapWidth = state.replay.map_size[0] * 2;
    const miniMapHeight = state.replay.map_size[1] * 2;
    ui.miniMapPanel.x = 0;
    ui.miniMapPanel.y = ui.mapPanel.y + ui.mapPanel.height - miniMapHeight;
    ui.miniMapPanel.width = miniMapWidth;
    ui.miniMapPanel.height = miniMapHeight;
  }

  ui.infoPanel.x = screenWidth - 400;
  ui.infoPanel.y = ui.mapPanel.y + ui.mapPanel.height - 200;
  ui.infoPanel.width = 400;
  ui.infoPanel.height = 200;

  // Trace panel is always on the bottom of the screen.
  ui.tracePanel.x = 0;
  ui.tracePanel.y = ui.mapPanel.y + ui.mapPanel.height;
  ui.tracePanel.width = screenWidth;
  ui.tracePanel.height = screenHeight - ui.tracePanel.y - Common.SCRUBBER_HEIGHT;

  html.actionButtons.style.top = (ui.tracePanel.y - 148) + 'px';

  ui.mapPanel.updateDiv();
  ui.miniMapPanel.updateDiv();
  ui.infoPanel.updateDiv();
  ui.tracePanel.updateDiv();

  // Redraw the square after resizing.
  requestFrame();
}

// Handle mouse down events.
function onMouseDown() {
  ui.lastMousePos = ui.mousePos;
  ui.mouseDownPos = ui.mousePos;
  ui.mouseClick = true;

  if (Math.abs(ui.mousePos.y() - ui.tracePanel.y) < Common.SPLIT_DRAG_THRESHOLD) {
    ui.traceDragging = true
  } else {
    ui.mouseDown = true;
  }

  requestFrame();
}

// Handle mouse up events.
function onMouseUp() {
  ui.mouseUp = true;
  ui.mouseDown = false;
  ui.traceDragging = false;

  // Due to how we select objects on mouse-up (mouse-down is drag/pan),
  // we need to check for double-click on mouse-up as well.
  const currentTime = new Date().getTime();
  ui.mouseDoubleClick = currentTime - ui.lastClickTime < 300; // 300ms threshold for double-click
  ui.lastClickTime = currentTime;

  requestFrame();
}

// Handle mouse move events.
function onMouseMove(event: MouseEvent) {
  ui.mousePos = new Vec2f(event.clientX, event.clientY);
  var target = event.target as HTMLElement;
  while (target.id === "") {
    target = target.parentElement as HTMLElement;
  }
  ui.mouseTarget = target.id;

  // If mouse is close to a panels edge change cursor to edge changer.
  document.body.style.cursor = "default";
  if (Math.abs(ui.mousePos.y() - ui.tracePanel.y) < Common.SPLIT_DRAG_THRESHOLD) {
    document.body.style.cursor = "ns-resize";
  }

  // Drag the trace panel up or down.
  if (ui.traceDragging) {
    ui.traceSplit = ui.mousePos.y() / window.innerHeight
    onResize();
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
  // ui.tracePanel.panPos.setX(-state.step * 32);

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
    setFollowSelection(false);
  }
  // '[' and ']' to scrub forward and backward.
  if (event.key == "[") {
    setIsPlaying(false);
    updateStep(Math.max(state.step - 1, 0));
  }
  if (event.key == "]") {
    setIsPlaying(false);
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
    setIsPlaying(!state.isPlaying);
  }

  processActions(event);

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

  ctx.useMesh("map");
  drawMap(ui.mapPanel);

  if (state.showMiniMap) {
    ui.miniMapPanel.div.style.display = "block";
    ctx.useMesh("mini-map");
    drawMiniMap(ui.miniMapPanel);
  } else {
    ui.miniMapPanel.div.style.display = "none";
  }

  ctx.useMesh("trace");
  drawTrace(ui.tracePanel);

  if (state.showInfo) {
    ui.infoPanel.div.style.display = "block";
    updateReadout();
  } else {
    ui.infoPanel.div.style.display = "none";
  }

  if (state.showControls) {
    html.actionButtons.style.display = "block";
  } else {
    html.actionButtons.style.display = "none";
  }

  ctx.flush();

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

  ui.mouseUp = false;
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

// Parse URL parameters, and modify the map and trace panels accordingly.
async function parseUrlParams() {

  const urlParams = new URLSearchParams(window.location.search);

  // Load the replay.
  const replayUrl = urlParams.get('replayUrl');
  const wsUrl = urlParams.get('wsUrl');
  if (replayUrl) {
    console.log("Loading replay from URL: ", replayUrl);
    await fetchReplay(replayUrl);
    focusFullMap(ui.mapPanel);
  } else if (wsUrl) {
    Common.showModal(
      "info",
      "Connecting to a websocket",
      "Please wait a few seconds for the environment to load."
    );
    initWebSocket(wsUrl);
  } else {
    Common.showModal(
      "info",
      "Welcome to MettaScope",
      "Please drop a replay file here to see the replay."
    );
  }

  if (state.replay !== null) {
    // Set the current step.
    if (urlParams.get('step') !== null) {
      const initialStep = parseInt(urlParams.get('step') || "0");
      console.info("Step via query parameter:", initialStep);
      updateStep(initialStep, false);
    }

    // Set the playing state.
    if (urlParams.get('play') !== null) {
      setIsPlaying(urlParams.get('play') === "true");
      console.info("Playing state via query parameter:", state.isPlaying);
    }

    // Set selected object.
    if (urlParams.get('selectedObjectId') !== null) {
      const selectedObjectId = parseInt(urlParams.get('selectedObjectId') || "-1") - 1;
      if (selectedObjectId >= 0 && selectedObjectId < state.replay.grid_objects.length) {
        state.selectedGridObject = state.replay.grid_objects[selectedObjectId];
        setFollowSelection(true);
        ui.mapPanel.zoomLevel = Common.DEFAULT_ZOOM_LEVEL;
        ui.tracePanel.zoomLevel = Common.DEFAULT_TRACE_ZOOM_LEVEL;
        console.info("Selected object via query parameter:", state.selectedGridObject);
      } else {
        console.warn("Invalid selectedObjectId:", selectedObjectId);
      }
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

// Handle share button click
function onShareButtonClick() {
  // Copy the current URL to the clipboard
  navigator.clipboard.writeText(window.location.href);
  // Show a toast notification
  Common.showToast("URL copied to clipboard");
}

function setIsPlaying(isPlaying: boolean) {
  state.isPlaying = isPlaying;
  if (state.isPlaying) {
    html.playButton.setAttribute("src", "data/ui/pause.png");
  } else {
    html.playButton.setAttribute("src", "data/ui/play.png");
  }
  requestFrame();
}

function toggleOpacity(button: HTMLImageElement, show: boolean) {
  if (show) {
    button.style.opacity = "1";
  } else {
    button.style.opacity = "0.2";
  }
}

// Set the playback speed and update the speed buttons.
function setPlaybackSpeed(speed: number) {
  state.playbackSpeed = speed;
  // Update the speed buttons to show the current speed.
  for (let i = 0; i < html.speedButtons.length; i++) {
    toggleOpacity(html.speedButtons[i], Common.SPEEDS[i] <= speed);
  }
}

// Initial resize.
onResize();

html.modal.classList.add("hidden");
html.toast.classList.add("hiding");
html.actionButtons.classList.add("hidden");

// Each panel has a div we use for event handling.
// But rendering happens bellow on global canvas.
// We make divs transparent to see through them.
ui.mapPanel.div.style.backgroundColor = "rgba(0, 0, 0, 0.0)";
ui.tracePanel.div.style.backgroundColor = "rgba(0, 0, 0, 0.0)";
ui.miniMapPanel.div.style.backgroundColor = "rgba(0, 0, 0, 0.0)";

// Add event listener to resize the canvas when the window is resized.
window.addEventListener('resize', onResize);
window.addEventListener('keydown', onKeyDown);
window.addEventListener('mousedown', onMouseDown);
window.addEventListener('mouseup', onMouseUp);
window.addEventListener('mousemove', onMouseMove);
window.addEventListener('wheel', onScroll);
window.addEventListener('dragenter', preventDefaults, false);
window.addEventListener('dragleave', preventDefaults, false);
window.addEventListener('dragover', preventDefaults, false);
window.addEventListener('drop', handleDrop, false);

// Header area
html.shareButton.addEventListener('click', onShareButtonClick);
html.helpButton.addEventListener('click', () => {
  window.open("mettascope_info.html", "_blank");
});

// Bottom area
html.scrubber.addEventListener('input', onScrubberChange);
html.scrubber.setAttribute("type", "range");
html.scrubber.setAttribute("value", "0");

html.rewindToStartButton.addEventListener('click', () => {
  setIsPlaying(false);
  updateStep(0)
});
html.stepBackButton.addEventListener('click', () => {
  setIsPlaying(false);
  updateStep(Math.max(state.step - 1, 0))
});
html.playButton.addEventListener('click', () =>
  setIsPlaying(!state.isPlaying)
);
html.stepForwardButton.addEventListener('click', () => {
  setIsPlaying(false);
  updateStep(Math.min(state.step + 1, state.replay.max_steps - 1))
});
html.rewindToEndButton.addEventListener('click', () => {
  setIsPlaying(false);
  updateStep(state.replay.max_steps - 1)
});

// Speed buttons
for (let i = 0; i < html.speedButtons.length; i++) {
  html.speedButtons[i].addEventListener('click',
    () => setPlaybackSpeed(Common.SPEEDS[i])
  );
}

html.resourcesToggle.addEventListener('click', () => {
  state.showResources = !state.showResources;
  localStorage.setItem("showResources", state.showResources.toString());
  toggleOpacity(html.resourcesToggle, state.showResources);
  requestFrame();
});
if (localStorage.hasOwnProperty("showResources")) {
  state.showResources = localStorage.getItem("showResources") === "true";
}
toggleOpacity(html.resourcesToggle, state.showResources);

// Toggle follow selection state.
html.focusToggle.addEventListener('click', () => {
  setFollowSelection(!state.followSelection);
});
toggleOpacity(html.focusToggle, state.followSelection);

html.gridToggle.addEventListener('click', () => {
  state.showGrid = !state.showGrid;
  localStorage.setItem("showGrid", state.showGrid.toString());
  toggleOpacity(html.gridToggle, state.showGrid);
  requestFrame();
});
if (localStorage.hasOwnProperty("showGrid")) {
  state.showGrid = localStorage.getItem("showGrid") === "true";
}
toggleOpacity(html.gridToggle, state.showGrid);

html.visualRangeToggle.addEventListener('click', () => {
  state.showVisualRanges = !state.showVisualRanges;
  localStorage.setItem("showVisualRanges", state.showVisualRanges.toString());
  toggleOpacity(html.visualRangeToggle, state.showVisualRanges);
  requestFrame();
});
if (localStorage.hasOwnProperty("showVisualRanges")) {
  state.showVisualRanges = localStorage.getItem("showVisualRanges") === "true";
}
toggleOpacity(html.visualRangeToggle, state.showVisualRanges);

html.fogOfWarToggle.addEventListener('click', () => {
  state.showFogOfWar = !state.showFogOfWar;
  localStorage.setItem("showFogOfWar", state.showFogOfWar.toString());
  toggleOpacity(html.fogOfWarToggle, state.showFogOfWar);
  requestFrame();
});
if (localStorage.hasOwnProperty("showFogOfWar")) {
  state.showFogOfWar = localStorage.getItem("showFogOfWar") === "true";
}
toggleOpacity(html.fogOfWarToggle, state.showFogOfWar);

html.minimapToggle.addEventListener('click', () => {
  state.showMiniMap = !state.showMiniMap;
  localStorage.setItem("showMiniMap", state.showMiniMap.toString());
  toggleOpacity(html.minimapToggle, state.showMiniMap);
  requestFrame();
});
if (localStorage.hasOwnProperty("showMiniMap")) {
  state.showMiniMap = localStorage.getItem("showMiniMap") === "true";
}
toggleOpacity(html.minimapToggle, state.showMiniMap);

html.controlsToggle.addEventListener('click', () => {
  state.showControls = !state.showControls;
  localStorage.setItem("showControls", state.showControls.toString());
  toggleOpacity(html.controlsToggle, state.showControls);
  requestFrame();
});
if (localStorage.hasOwnProperty("showControls")) {
  state.showControls = localStorage.getItem("showControls") === "true";
}
toggleOpacity(html.controlsToggle, state.showControls);

html.infoToggle.addEventListener('click', () => {
  state.showInfo = !state.showInfo;
  localStorage.setItem("showInfo", state.showInfo.toString());
  toggleOpacity(html.infoToggle, state.showInfo);
  requestFrame();
});
if (localStorage.hasOwnProperty("showInfo")) {
  state.showInfo = localStorage.getItem("showInfo") === "true";
}
toggleOpacity(html.infoToggle, state.showInfo);

initActionButtons();

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
  setPlaybackSpeed(0.1);

  requestFrame();
});
