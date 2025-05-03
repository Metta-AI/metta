import { Vec2f, Mat3f } from './vector_math.js';
import { Grid } from './grid.js';
import { Drawer } from './drawer.js';
import * as Common from './common.js';
import { ui, state } from './common.js';

export class PanelInfo {
  public x: number = 0;
  public y: number = 0;
  public width: number = 0;
  public height: number = 0;
  public name: string = "";
  public isPanning: boolean = false;
  public panPos: Vec2f = new Vec2f(0, 0);
  public zoomLevel: number = 1;
  public div: HTMLDivElement | null;

  constructor(name: string) {
    this.name = name;
    this.div = null;
  }

  // Check if a point is inside the panel.
  inside(point: Vec2f): boolean {
    return point.x() >= this.x && point.x() < this.x + this.width &&
      point.y() >= this.y && point.y() < this.y + this.height;
  }

  // Transform a point from the canvas to the map coordinate system.
  transformPoint(point: Vec2f): Vec2f {
    const m = Mat3f.translate(this.x + this.width / 2, this.y + this.height / 2)
      .mul(Mat3f.scale(this.zoomLevel, this.zoomLevel))
      .mul(Mat3f.translate(this.panPos.x(), this.panPos.y()));
    return m.inverse().transform(point);
  }

  // Make the panel focus on a specific position in the panel.
  focusPos(x: number, y: number) {
    this.panPos = new Vec2f(
      -x,
      -y
    );
    this.zoomLevel = 1/2;
  }

  // Update the pan and zoom level based on the mouse position and scroll delta.
  updatePanAndZoom(): boolean {

    if (ui.mouseClick) {
      this.isPanning = true;
    }
    if (!ui.mouseDown) {
      this.isPanning = false;
    }

    if (this.isPanning && ui.mousePos.sub(ui.lastMousePos).length() > 1) {
      const lastMousePoint = this.transformPoint(ui.lastMousePos);
      const newMousePoint = this.transformPoint(ui.mousePos);
      this.panPos = this.panPos.add(newMousePoint.sub(lastMousePoint));
      ui.lastMousePos = ui.mousePos;
      return true;
    }

    if (ui.scrollDelta !== 0) {
      const oldMousePoint = this.transformPoint(ui.mousePos);
      this.zoomLevel = this.zoomLevel + ui.scrollDelta / Common.SCROLL_ZOOM_FACTOR;
      this.zoomLevel = Math.max(Math.min(this.zoomLevel, Common.MAX_ZOOM_LEVEL), Common.MIN_ZOOM_LEVEL);
      const newMousePoint = this.transformPoint(ui.mousePos);
      if (oldMousePoint != null && newMousePoint != null) {
        this.panPos = this.panPos.add(newMousePoint.sub(oldMousePoint));
      }
      ui.scrollDelta = 0;
      return true;
    }
    return false;
  }
}


let drawer: Drawer;

// Flag to prevent multiple calls to requestAnimationFrame
let frameRequested = false;

// Function to safely request animation frame
function requestFrame() {
  if (!frameRequested) {
    frameRequested = true;
    requestAnimationFrame((time) => {
      frameRequested = false;
      onFrame();
    });
  }
}

// Get the html elements we will use.
const scrubber = document.getElementById('main-scrubber') as HTMLInputElement;
const playButton = document.getElementById('play-button') as HTMLButtonElement;

// Get the canvas element.
const globalCanvas = document.getElementById('global-canvas') as HTMLCanvasElement;

const mapPanel = new PanelInfo("map");
const tracePanel = new PanelInfo("trace");
const infoPanel = new PanelInfo("info");

// Get the modal element
const modal = document.getElementById('modal');

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

  mapPanel.x = 0;
  mapPanel.y = 0;
  mapPanel.width = mapWidth * ui.traceSplit;
  mapPanel.height = mapHeight - Common.PANEL_BOTTOM_MARGIN;

  tracePanel.x = mapWidth * ui.traceSplit;
  tracePanel.y = mapHeight * ui.infoSplit;
  tracePanel.width = mapWidth * (1 - ui.traceSplit);
  tracePanel.height = mapHeight * (1 - ui.infoSplit) - Common.PANEL_BOTTOM_MARGIN;

  infoPanel.x = mapWidth * ui.traceSplit;
  infoPanel.y = 0;
  infoPanel.width = mapWidth * (1 - ui.traceSplit);
  infoPanel.height = mapHeight * ui.infoSplit;
  if (infoPanel.div === null) {
    infoPanel.div = document.createElement("div");
    infoPanel.div.id = infoPanel.name + "-div";
    document.body.appendChild(infoPanel.div);
  }
  if (infoPanel.div !== null) {
    const div = infoPanel.div;
    div.style.position = 'absolute';
    div.style.top = infoPanel.y + 'px';
    div.style.left = infoPanel.x + 'px';
    div.style.width = infoPanel.width + 'px';
    div.style.height = infoPanel.height + 'px';
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

  if (Math.abs(ui.mousePos.x() - mapPanel.width) < Common.SPLIT_DRAG_THRESHOLD) {
    ui.traceDragging = true
    console.log("Started trace dragging")
  } else if (ui.mousePos.x() > mapPanel.width && Math.abs(ui.mousePos.y() - infoPanel.height) < Common.SPLIT_DRAG_THRESHOLD) {
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

  if (Math.abs(ui.mousePos.x() - mapPanel.width) < Common.SPLIT_DRAG_THRESHOLD) {
    document.body.style.cursor = "ew-resize";
  }

  if (ui.mousePos.x() > mapPanel.width &&
    Math.abs(ui.mousePos.y() - infoPanel.height) < Common.SPLIT_DRAG_THRESHOLD
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

// Decompress a stream, used for compressed JSON from fetch or drag and drop.
async function decompressStream(stream: ReadableStream<Uint8Array>): Promise<string> {
  const decompressionStream = new DecompressionStream('deflate');
  const decompressedStream = stream.pipeThrough(decompressionStream);

  const reader = decompressedStream.getReader();
  const chunks: Uint8Array[] = [];
  let result;
  while (!(result = await reader.read()).done) {
    chunks.push(result.value);
  }

  const totalLength = chunks.reduce((acc, val) => acc + val.length, 0);
  const flattenedChunks = new Uint8Array(totalLength);

  let offset = 0;
  for (const chunk of chunks) {
    flattenedChunks.set(chunk, offset);
    offset += chunk.length;
  }

  const decoder = new TextDecoder();
  return decoder.decode(flattenedChunks);
}

// Load the replay from a URL.
async function fetchReplay(replayUrl: string) {
  try {
    const response = await fetch(replayUrl);
    if (!response.ok) {
      throw new Error("Network response was not ok");
    }
    if (response.body === null) {
      throw new Error("Response body is null");
    }
    // Check the Content-Type header
    const contentType = response.headers.get('Content-Type');
    console.log("Content-Type: ", contentType);
    if (contentType === "application/json") {
      let replayData = await response.text();
      loadReplayText(replayData);
    } else if (contentType === "application/x-compress" || contentType === "application/octet-stream") {
      // Compressed JSON.
      const decompressedData = await decompressStream(response.body);
      loadReplayText(decompressedData);
    } else {
      throw new Error("Unsupported content type: " + contentType);
    }
  } catch (error) {
    showModal("error", "Error fetching replay", "Message: " + error);
  }
}

// Read a file from drag and drop.
async function readFile(file: File) {
  try {
    const contentType = file.type;
    console.log("Content-Type: ", contentType);
    if (contentType === "application/json") {
      loadReplayText(await file.text());
    } else if (contentType === "application/x-compress" || contentType === "application/octet-stream") {
      // Compressed JSON.
      console.log("Decompressing file");
      const decompressedData = await decompressStream(file.stream());
      console.log("Decompressed file");
      loadReplayText(decompressedData);
    }
  } catch (error) {
    showModal("error", "Error reading file", "Message: " + error);
  }
}

// Expand a sequence of values
// [[0, value1], [2, value2], ...] -> [value1, value1, value2, ...]
function expandSequence(sequence: any[], numSteps: number): any[] {
  var expanded: any[] = [];
  var i = 0
  var j = 0
  var v: any = null
  for (i = 0; i < numSteps; i++) {
    if (j < sequence.length && sequence[j][0] == i) {
      v = sequence[j][1];
      j++;
    }
    expanded.push(v);
  }
  return expanded;
}

// Remove a prefix from a string.
function removePrefix(str: string, prefix: string) {
  return str.startsWith(prefix) ? str.slice(prefix.length) : str;
}

// Remove a suffix from a string.
function removeSuffix(str: string, suffix: string) {
  return str.endsWith(suffix) ? str.slice(0, -suffix.length) : str;
}

// Load the replay text.
async function loadReplayText(replayData: any) {
  state.replay = JSON.parse(replayData);

  // Go through each grid object and expand its key sequence.
  for (const gridObject of state.replay.grid_objects) {
    for (const key in gridObject) {
      if (gridObject[key] instanceof Array) {
        gridObject[key] = expandSequence(gridObject[key], state.replay.max_steps);
      }
    }
  }

  // Find all agents for faster access.
  state.replay.agents = [];
  for (let i = 0; i < state.replay.num_agents; i++) {
    for (const gridObject of state.replay.grid_objects) {
      if (gridObject["agent_id"] == i) {
        state.replay.agents.push(gridObject);
      }
    }
  }

  // Create action image mappings for faster access.
  state.replay.action_images = [];
  for (const actionName of state.replay.action_names) {
    let path = "trace/" + actionName + ".png";
    if (drawer.hasImage(path)) {
      state.replay.action_images.push(path);
    } else {
      console.warn("Action not supported: ", path);
      state.replay.action_images.push("trace/unknown.png");
    }
  }

  // Create a list of all keys objects can have.
  state.replay.all_keys = new Set();
  for (const gridObject of state.replay.grid_objects) {
    for (const key in gridObject) {
      state.replay.all_keys.add(key);
    }
  }

  // Create object image mapping for faster access.
  // Example: 3 -> ["objects/altar.png", "objects/altar.empty.png"]
  // Example: 1 -> ["objects/wall.png", "objects/wall.png"]
  state.replay.object_images = []
  for (let i = 0; i < state.replay.object_types.length; i++) {
    const typeName = state.replay.object_types[i];
    var image = "objects/" + typeName + ".png";
    if (!drawer.hasImage(image)) {
      console.warn("Object not supported: ", typeName);
      image = "objects/unknown.png";
    }
    var imageEmpty = "objects/" + typeName + ".empty.png";
    if (!drawer.hasImage(imageEmpty)) {
      imageEmpty = image;
    }
    state.replay.object_images.push([image, imageEmpty]);
  }

  // Create resource inventory mapping for faster access.
  // Example: "inv:heart" -> ["resources/heart.png", [1, 1, 1, 1]]
  // Example: "inv:ore.red" -> ["resources/ore.red.png", [1, 1, 1, 1]]
  // Example: "agent:inv:heart.blue" -> ["resources/heart.png", [0, 0, 1, 1]]
  // Example: "inv:cat_food.red" -> ["resources/unknown.png", [1, 0, 0, 1]]
  state.replay.resource_inventory = new Map();
  for (const key of state.replay.all_keys) {
    if (key.startsWith("inv:") || key.startsWith("agent:inv:")) {
      var type: string = key;
      type = removePrefix(type, "inv:")
      type = removePrefix(type, "agent:inv:");
      var color = [1, 1, 1, 1]; // Default to white.
      for (const [colorName, colorValue] of Common.COLORS) {
        if (type.endsWith(colorName)) {
          if(drawer.hasImage("resources/" + type + ".png")) {
            // Use the resource.color.png with white color.
            break;
          } else {
            // Use the resource.png with specific color.
            type = removeSuffix(type, "." + colorName);
            color = colorValue as number[];
            if (!drawer.hasImage("resources/" + type + ".png")) {
              // Use the unknown.png with specific color.
              console.warn("Resource not supported: ", type);
              type = "unknown";
            }
          }
        }
      }
      image = "resources/" + type + ".png";
      state.replay.resource_inventory.set(key, [image, color]);
    }
  }

  console.info("replay: ", state.replay);

  // Set the scrubber max value to the max steps.
  scrubber.max = (state.replay.max_steps - 1).toString();

  closeModal();
  focusFullMap(mapPanel);
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
    urlParams.set('mapPanX', Math.round(mapPanel.panPos.x()).toString());
    urlParams.set('mapPanY', Math.round(mapPanel.panPos.y()).toString());
    // Remove selected object when there is no selection
    urlParams.delete('selectedObjectId');
  }

  // Include map zoom level
  if (mapPanel.zoomLevel != 1) {
    // Only include zoom to 3 decimal places.
    urlParams.set('mapZoom', mapPanel.zoomLevel.toFixed(3));
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
function updateStep(newStep: number, skipScrubberUpdate = false) {
  // Update the step variable
  state.step = newStep;

  // Update the scrubber value (unless told to skip)
  if (!skipScrubberUpdate) {
    scrubber.value = state.step.toString();
  }

  // Update trace panel position
  tracePanel.panPos.setX(-state.step * 32);

  // Request a new frame
  requestFrame();
}

// Handle scrubber change events.
function onScrubberChange() {
  updateStep(parseInt(scrubber.value), true);
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

// Gets an attribute from a grid object respecting the current step.
function getAttr(obj: any, attr: string, atStep = -1, defaultValue = 0): any {
  if (atStep == -1) {
    // When step is not defined, use global step.
    atStep = state.step;
  }
  if (obj[attr] === undefined) {
    return defaultValue;
  } else if (obj[attr] instanceof Array) {
    return obj[attr][atStep];
  } else {
    // Must be a constant that does not change over time.
    return obj[attr];
  }
}

// Generate a color from an agent id.
function colorFromId(agentId: number) {
  let n = agentId + Math.PI + Math.E + Math.SQRT2;
  return [
    n * Math.PI % 1.0,
    n * Math.E % 1.0,
    n * Math.SQRT2 % 1.0,
    1.0
  ]
}

// Checks to see of object has any inventory.
function hasInventory(obj: any) {
  for (const [key, [icon, color]] of state.replay.resource_inventory) {
    if (getAttr(obj, key) > 0) {
      return true;
    }
  }
  return false;
}

// Make the panel focus on the full map, used at the start of the replay.
function focusFullMap(panel: PanelInfo) {
  if (state.replay === null) {
    return;
  }
  const width = state.replay.map_size[0] * Common.TILE_SIZE;
  const height = state.replay.map_size[1] * Common.TILE_SIZE;
  panel.focusPos(width / 2, height / 2);
  panel.zoomLevel = Math.min(panel.width / width, panel.height / height);
}

// Draw the tiles that make up the floor.
function drawFloor(mapSize: [number, number]) {

  // Compute the visibility map, each agent contributes to the visibility map.
  const visibilityMap = new Grid(mapSize[0], mapSize[1]);

  // Update the visibility map for a grid object.
  function updateVisibilityMap(gridObject: any) {
    const x = getAttr(gridObject, "c");
    const y = getAttr(gridObject, "r");
    var visionSize = Math.floor(getAttr(
      gridObject,
      "agent:vision_size",
      state.step,
      Common.DEFAULT_VISION_SIZE
    ) / 2);
    for (let dx = -visionSize; dx <= visionSize; dx++) {
      for (let dy = -visionSize; dy <= visionSize; dy++) {
        visibilityMap.set(
          x +dx,
          y + dy,
          true
        );
      }
    }
  }

  if (state.selectedGridObject !== null && state.selectedGridObject.agent_id !== undefined) {
    // When there is a selected grid object only update its visibility.
    updateVisibilityMap(state.selectedGridObject);
  } else {
    // When there is no selected grid object update the visibility map for all agents.
    for (const gridObject of state.replay.grid_objects) {
      const type = gridObject.type;
      const typeName = state.replay.object_types[type];
      if (typeName == "agent") {
        updateVisibilityMap(gridObject);
      }
    }
  }

  // Draw the floor, darker where there is no visibility.
  for (let x = 0; x < mapSize[0]; x++) {
    for (let y = 0; y < mapSize[1]; y++) {
      const color = visibilityMap.get(x, y) ? [1, 1, 1, 1] : [0.75, 0.75, 0.75, 1];
      drawer.drawSprite('objects/floor.png', x * Common.TILE_SIZE, y * Common.TILE_SIZE, color);
    }
  }
}

// Draw the walls, based on the adjacency map, and fill any holes.
function drawWalls() {
  // Construct wall adjacency map.
  var wallMap = new Grid(state.replay.map_size[0], state.replay.map_size[1]);
  for (const gridObject of state.replay.grid_objects) {
    const type = gridObject.type;
    const typeName = state.replay.object_types[type];
    if (typeName !== "wall") {
      continue;
    }
    const x = getAttr(gridObject, "c");
    const y = getAttr(gridObject, "r");
    wallMap.set(x, y, true);
  }

  // Draw the walls following the adjacency map.
  for (const gridObject of state.replay.grid_objects) {
    const type = gridObject.type;
    const typeName = state.replay.object_types[type];
    if (typeName !== "wall") {
      continue;
    }
    const x = getAttr(gridObject, "c");
    const y = getAttr(gridObject, "r");
    var suffix = "0";
    var n = false, w = false, e = false, s = false;
    if (wallMap.get(x, y - 1)) {
      n = true;
    }
    if (wallMap.get(x - 1, y)) {
      w = true;
    }
    if (wallMap.get(x, y + 1)) {
      s = true;
    }
    if (wallMap.get(x + 1, y)) {
      e = true;
    }
    if (n || w || e || s) {
      suffix = (n ? "n" : "") + (w ? "w" : "") + (s ? "s" : "") + (e ? "e" : "");
    }
    drawer.drawSprite('objects/wall.' + suffix + '.png', x * Common.TILE_SIZE, y * Common.TILE_SIZE);
  }

  // Draw the wall in-fill following the adjacency map.
  for (const gridObject of state.replay.grid_objects) {
    const type = gridObject.type;
    const typeName = state.replay.object_types[type];
    if (typeName !== "wall") {
      continue;
    }
    const x = getAttr(gridObject, "c");
    const y = getAttr(gridObject, "r");
    // If walls to E, S and SE is filled, draw a wall fill.
    var s = false, e = false, se = false;
    if (wallMap.get(x + 1, y)) {
      e = true;
    }
    if (wallMap.get(x, y + 1)) {
      s = true;
    }
    if (wallMap.get(x + 1, y + 1)) {
      se = true;
    }
    if (e && s && se) {
      drawer.drawSprite(
        'objects/wall.fill.png',
        x * Common.TILE_SIZE + Common.TILE_SIZE / 2,
        y * Common.TILE_SIZE + Common.TILE_SIZE / 2 - 42
      );
    }
  }
}

// Draw all objects on the map (that are not walls).
function drawObjects() {
  for (const gridObject of state.replay.grid_objects) {
    const type: number = gridObject.type;
    const typeName: string = state.replay.object_types[type];
    if (typeName === "wall") {
      // Walls are drawn in a different way.
      continue;
    }
    const x = getAttr(gridObject, "c")
    const y = getAttr(gridObject, "r")

    if (gridObject["agent_id"] !== undefined) {
      // Respect orientation of an object usually an agent.
      const orientation = getAttr(gridObject, "agent:orientation");
      var suffix = "";
      if (orientation == 0) {
        suffix = "n";
      } else if (orientation == 1) {
        suffix = "s";
      } else if (orientation == 2) {
        suffix = "w";
      } else if (orientation == 3) {
        suffix = "e";
      }

      const agent_id = gridObject["agent_id"];

      drawer.drawSprite(
        "agents/agent." + suffix + ".png",
        x * Common.TILE_SIZE,
        y * Common.TILE_SIZE,
        colorFromId(agent_id)
      );
    } else {
      // Draw regular objects.
      if (hasInventory(gridObject)) {
        // object.png
        drawer.drawSprite(
          state.replay.object_images[type][0],
          x * Common.TILE_SIZE,
          y * Common.TILE_SIZE
        );
      } else {
        // object.empty.png
        drawer.drawSprite(
          state.replay.object_images[type][1],
          x * Common.TILE_SIZE,
          y * Common.TILE_SIZE
        );
      }
    }
  }
}

function drawActions() {
  // Draw actions above the objects.
  for (const gridObject of state.replay.grid_objects) {
    const x = getAttr(gridObject, "c")
    const y = getAttr(gridObject, "r")

    // Do agent actions.
    if (gridObject["action"] !== undefined) {
      // Draw the action:
      const action = getAttr(gridObject, "action");
      const action_success = getAttr(gridObject, "action_success");
      if (action_success && action != null) {
        const action_name = state.replay.action_names[action[0]];
        const orientation = getAttr(gridObject, "agent:orientation");
        var rotation = 0;
        if (orientation == 0) {
          rotation = Math.PI / 2; // North
        } else if (orientation == 1) {
          rotation = -Math.PI / 2; // South
        } else if (orientation == 2) {
          rotation = Math.PI; // West
        } else if (orientation == 3) {
          rotation = 0; // East
        }
        if (action_name == "attack" && action[1] >= 0 && action[1] <= 8) {
          drawer.drawSprite(
            "actions/attack" + (action[1] + 1) + ".png",
            x * Common.TILE_SIZE,
            y * Common.TILE_SIZE,
            [1, 1, 1, 1],
            1,
            rotation
          );
        } else if (action_name == "attack_nearest") {
          drawer.drawSprite(
            "actions/attack_nearest.png",
            x * Common.TILE_SIZE,
            y * Common.TILE_SIZE,
            [1, 1, 1, 1],
            1,
            rotation
          );
        } else if (action_name == "put_recipe_items") {
          drawer.drawSprite(
            "actions/put_recipe_items.png",
            x * Common.TILE_SIZE,
            y * Common.TILE_SIZE,
            [1, 1, 1, 1],
            1,
            rotation
          );
        } else if (action_name == "get_output") {
          drawer.drawSprite(
            "actions/get_output.png",
            x * Common.TILE_SIZE,
            y * Common.TILE_SIZE,
            [1, 1, 1, 1],
            1,
            rotation
          );
        } else if (action_name == "swap") {
          drawer.drawSprite(
            "actions/swap.png",
            x * Common.TILE_SIZE,
            y * Common.TILE_SIZE,
            [1, 1, 1, 1],
            1,
            rotation
          );
        }
      }
    }

    // Do building actions.
    if (getAttr(gridObject, "converting") > 0) {
      drawer.drawSprite(
        "actions/converting.png",
        x * Common.TILE_SIZE,
        y * Common.TILE_SIZE - 100,
        [1, 1, 1, 1],
        1,
        // Apply the gentle rotation.
        -state.step * 0.1
      );
    }

    // Do states
    if (getAttr(gridObject, "agent:frozen") > 0) {
      drawer.drawSprite(
        "agents/frozen.png",
        x * Common.TILE_SIZE,
        y * Common.TILE_SIZE,
      );
    }
  }
}

function drawInventory() {
  // Draw the object's inventory.
  for (const gridObject of state.replay.grid_objects) {
    const x = getAttr(gridObject, "c")
    const y = getAttr(gridObject, "r")

    // Sum up the objects inventory, in case we need to condense it.
    let inventoryX = Common.INVENTORY_PADDING;
    let numItems = 0;
    for (const [key, [icon, color]] of state.replay.resource_inventory) {
      const num = getAttr(gridObject, key);
      numItems += num;
    }
    // Draw the actual inventory icons.
    let advanceX = Math.min(32, (Common.TILE_SIZE - Common.INVENTORY_PADDING*2) / numItems);
    for (const [key, [icon, color]] of state.replay.resource_inventory) {
      const num = getAttr(gridObject, key);
      for (let i = 0; i < num; i++) {
        drawer.drawSprite(
          icon,
          x * Common.TILE_SIZE + inventoryX - Common.TILE_SIZE/2,
          y * Common.TILE_SIZE - Common.TILE_SIZE/2 + 16,
          color,
          1/8,
          0
        );
        inventoryX += advanceX;
      }
    }
  }
}

function drawRewards() {
  // Draw the reward on the bottom of the object.
  for (const gridObject of state.replay.grid_objects) {
    const x = getAttr(gridObject, "c")
    const y = getAttr(gridObject, "r")
    if (gridObject["total_reward"] !== undefined) {
      const totalReward = getAttr(gridObject, "total_reward");
      let rewardX = 0;
      let advanceX = Math.min(32, Common.TILE_SIZE / totalReward);
      for (let i = 0; i < totalReward; i++) {
        drawer.save()
        drawer.translate(
          x * Common.TILE_SIZE + rewardX - Common.TILE_SIZE/2,
          y * Common.TILE_SIZE + Common.TILE_SIZE/2 - 16
        );
        drawer.scale(1/8, 1/8);
        drawer.drawSprite("resources/reward.png", 0, 0);
        drawer.restore()
        rewardX += advanceX;
      }
    }
  }
}

function drawSelection() {
  if (state.selectedGridObject === null) {
    return;
  }

  const x = getAttr(state.selectedGridObject, "c")
  const y = getAttr(state.selectedGridObject, "r")
  drawer.drawSprite("selection.png", x * Common.TILE_SIZE, y * Common.TILE_SIZE);
}

function drawTrajectory() {
  if (state.selectedGridObject === null) {
    return;
  }
  if (state.selectedGridObject.c.length > 0 || state.selectedGridObject.r.length > 0) {

    // Draw both past and future trajectories.
    for (let i = 1; i < state.replay.max_steps; i++) {
      const cx0 = getAttr(state.selectedGridObject, "c", i - 1);
      const cy0 = getAttr(state.selectedGridObject, "r", i - 1);
      const cx1 = getAttr(state.selectedGridObject, "c", i);
      const cy1 = getAttr(state.selectedGridObject, "r", i);
      if (cx0 !== cx1 || cy0 !== cy1) {
        const a = 1 - Math.abs(i - state.step) / 200;
        if (a > 0) {
          let color = [0, 0, 0, a];
          let image = "";
          if (state.step >= i) {
            // Past trajectory is black.
            color = [0, 0, 0, a];
            if (state.selectedGridObject.agent_id !== undefined) {
              image = "agents/footprints.png";
            } else {
              image = "agents/past_arrow.png";
            }
          } else {
            // Future trajectory is white.
            color = [a, a, a, a];
            if (state.selectedGridObject.agent_id !== undefined) {
              image = "agents/path.png";
            } else {
              image = "agents/future_arrow.png";
            }
          }

          if (cx1 > cx0) { // east
            drawer.drawSprite(
              image,
              cx0 * Common.TILE_SIZE,
              cy0 * Common.TILE_SIZE + 60,
              color,
              1,
              0
            );
          } else if (cx1 < cx0) { // west
            drawer.drawSprite(
              image,
              cx0 * Common.TILE_SIZE,
              cy0 * Common.TILE_SIZE + 60,
              color,
              1,
              Math.PI
            );
          } else if (cy1 > cy0) { // south
            drawer.drawSprite(
              image,
              cx0 * Common.TILE_SIZE,
              cy0 * Common.TILE_SIZE + 60,
              color,
              1,
              -Math.PI / 2
            );
          } else if (cy1 < cy0) { // north
            drawer.drawSprite(
              image,
              cx0 * Common.TILE_SIZE,
              cy0 * Common.TILE_SIZE + 60,
              color,
              1,
              Math.PI / 2
            );
          }
        }
      }
    }
  }
}

function drawMap(panel: PanelInfo) {
  if (state.replay === null || drawer === null || drawer.ready === false) {
    return;
  }

  const localMousePos = panel.transformPoint(ui.mousePos);

  if (ui.mouseClick) {
    // Reset the follow flags.
    state.followSelection = false;
    state.followTraceSelection = false;

    if (localMousePos != null) {
      const gridMousePos = new Vec2f(
        Math.round(localMousePos.x() / Common.TILE_SIZE),
        Math.round(localMousePos.y() / Common.TILE_SIZE)
      );
      const gridObject = state.replay.grid_objects.find((obj: any) => {
        const x: number = getAttr(obj, "c");
        const y: number = getAttr(obj, "r");
        return x === gridMousePos.x() && y === gridMousePos.y();
      });
      if (gridObject !== undefined) {
        state.selectedGridObject = gridObject;
        console.log("selectedGridObject on map:", state.selectedGridObject);

        if (ui.mouseDoubleClick) {
          // Toggle followSelection on double-click
          state.followSelection = true;
          state.followTraceSelection = true;
          panel.zoomLevel = 1/2;
          tracePanel.zoomLevel = 1;
        }
      }
    }
  }

  // If we're following a selection, center the map on it
  if (state.followSelection && state.selectedGridObject !== null) {
    const x = getAttr(state.selectedGridObject, "c");
    const y = getAttr(state.selectedGridObject, "r");
    panel.panPos = new Vec2f(-x * Common.TILE_SIZE, -y * Common.TILE_SIZE);
  }

  drawer.save();
  drawer.setScissorRect(panel.x, panel.y, panel.width, panel.height);

  drawer.translate(panel.x + panel.width / 2, panel.y + panel.height / 2);
  drawer.scale(panel.zoomLevel, panel.zoomLevel);
  drawer.translate(panel.panPos.x(), panel.panPos.y());

  drawFloor(state.replay.map_size);
  drawWalls();
  drawTrajectory();
  drawObjects();
  drawSelection();
  drawActions();
  drawInventory();
  drawRewards();

  drawer.restore();
}

function drawTrace(panel: PanelInfo) {
  if (state.replay === null || drawer === null || drawer.ready === false) {
    return;
  }

  const localMousePos = panel.transformPoint(ui.mousePos);

  if (state.followTraceSelection && state.selectedGridObject !== null) {
    panel.focusPos(
      state.step * Common.TRACE_WIDTH + Common.TRACE_WIDTH/2,
      getAttr(state.selectedGridObject, "agent_id") * Common.TRACE_HEIGHT + Common.TRACE_HEIGHT/2
    );
  }

  if (ui.mouseClick &&panel.inside(ui.mousePos)) {
    if (localMousePos != null) {
      const mapX = localMousePos.x();
      if (mapX > 0 && mapX < state.replay.max_steps * Common.TRACE_WIDTH &&
        localMousePos.y() > 0 && localMousePos.y() < state.replay.num_agents * Common.TRACE_HEIGHT) {
        const agentId = Math.floor(localMousePos.y() / Common.TRACE_HEIGHT);
        if (agentId >= 0 && agentId < state.replay.num_agents) {
          state.followSelection = true;
          state.selectedGridObject = state.replay.agents[agentId];
          console.log("selectedGridObject on a trace:", state.selectedGridObject);
          mapPanel.focusPos(
            getAttr(state.selectedGridObject, "c") * Common.TILE_SIZE,
            getAttr(state.selectedGridObject, "r") * Common.TILE_SIZE
          );
          // Update the step to the clicked step.
          updateStep(Math.floor(mapX / Common.TRACE_WIDTH));

          if (ui.mouseDoubleClick) {
            state.followTraceSelection = true;
            panel.zoomLevel = 1;
          }
        }
      }
    }
  }

  drawer.save();
  drawer.setScissorRect(panel.x, panel.y, panel.width, panel.height);

  const fullSize = new Vec2f(
    state.replay.max_steps * Common.TRACE_WIDTH,
    state.replay.num_agents * Common.TRACE_HEIGHT
  );

  // Draw background
  drawer.drawSolidRect(
    panel.x, panel.y, panel.width, panel.height,
    [0.08, 0.08, 0.08, 1.0] // Dark background
  );

  drawer.translate(panel.x + panel.width / 2, panel.y + panel.height / 2);
  drawer.scale(panel.zoomLevel, panel.zoomLevel);
  drawer.translate(panel.panPos.x(), panel.panPos.y());

  // Draw rectangle around the selected agent
  if (state.selectedGridObject !== null && state.selectedGridObject.agent_id !== undefined) {
    const agentId = state.selectedGridObject.agent_id;

    // Draw selection rectangle
    drawer.drawSolidRect(
      0, agentId * Common.TRACE_HEIGHT, fullSize.x(), Common.TRACE_HEIGHT,
      [.3, .3, .3, 1]
    );
  }

  // Draw current step line that goes through all of the traces
  drawer.drawSolidRect(
    state.step * Common.TRACE_WIDTH, 0,
    Common.TRACE_WIDTH, fullSize.y(),
    [0.5, 0.5, 0.5, 0.5] // White with 50% opacity
  );

  // Draw agent traces
  for (let i = 0; i < state.replay.num_agents; i++) {
    const agent = state.replay.agents[i];
    for (let j = 0; j < state.replay.max_steps; j++) {
      const action = getAttr(agent, "action", j);
      const action_success = getAttr(agent, "action_success", j);

      if (action_success && action != null && action[0] > 0 && action[0] < state.replay.action_images.length) {
        drawer.drawSprite(
          state.replay.action_images[action[0]],
          j * Common.TRACE_WIDTH + Common.TRACE_WIDTH/2,
          i * Common.TRACE_HEIGHT + Common.TRACE_HEIGHT/2,
        );
      } else if (action != null && action[0] > 0 && action[0] < state.replay.action_images.length) {
        drawer.drawSprite(
          state.replay.action_images[action[0]],
          j * Common.TRACE_WIDTH + Common.TRACE_WIDTH/2,
          i * Common.TRACE_HEIGHT + Common.TRACE_HEIGHT/2,
          [0.01, 0.01, 0.01, 0.01],
        );
      }

      if (getAttr(agent, "agent:frozen", j) > 0) {
        drawer.drawSprite(
          "trace/frozen.png",
          j * Common.TRACE_WIDTH + Common.TRACE_WIDTH/2,
          i * Common.TRACE_HEIGHT + Common.TRACE_HEIGHT/2,
        );
      }

      const reward = getAttr(agent, "reward", j);
      // If there is reward, draw a star.
      if (reward > 0) {
        drawer.drawSprite(
          "resources/reward.png",
          j * Common.TRACE_WIDTH + Common.TRACE_WIDTH/2,
          i * Common.TRACE_HEIGHT + 256 - 32,
          [1.0, 1.0, 1.0, 1.0],
          1/8
        );
      }
    }
  }

  drawer.restore();
}

// Updates the readout of the selected object or replay info.
function updateReadout() {
  var readout = ""
  if (state.selectedGridObject !== null) {
    if (state.followSelection) {
      readout += "FOLLOWING SELECTION (double-click to unfollow)\n\n";
    }
    for (const key in state.selectedGridObject) {
      var value = getAttr(state.selectedGridObject, key);
      if (key == "type") {
        value = state.replay.object_types[value] + " (" + value + ")";
      }
      readout += key + ": " + value + "\n";
    }
  } else {
    readout += "Step: " + state.step + "\n";
    readout += "Map size: " + state.replay.map_size[0] + "x" + state.replay.map_size[1] + "\n";
    readout += "Num agents: " + state.replay.num_agents + "\n";
    readout += "Max steps: " + state.replay.max_steps + "\n";

    var objectTypeCounts = new Map<string, number>();
    for (const gridObject of state.replay.grid_objects) {
      const type = gridObject.type;
      const typeName = state.replay.object_types[type];
      objectTypeCounts.set(typeName, (objectTypeCounts.get(typeName) || 0) + 1);
    }
    for (const [key, value] of objectTypeCounts.entries()) {
      readout += key + " count: " + value + "\n";
    }
  }
  if (infoPanel.div !== null) {
    infoPanel.div.innerHTML = readout;
  }
}

// Draw a frame.
function onFrame() {
  if (state.replay === null || drawer === null || drawer.ready === false) {
    return;
  }

  // Make sure the canvas is the size of the window.
  globalCanvas.width = window.innerWidth;
  globalCanvas.height = window.innerHeight;

  drawer.clear();

  var fullUpdate = true;
  if (mapPanel.inside(ui.mousePos)) {
    if (mapPanel.updatePanAndZoom()) {
      fullUpdate = false;
    }
  }

  if (tracePanel.inside(ui.mousePos)) {
    if (tracePanel.updatePanAndZoom()) {
      fullUpdate = false;
    }
  }

  updateReadout();
  drawer.useMesh("map");
  drawMap(mapPanel);
  drawer.useMesh("trace");
  drawTrace(tracePanel);

  drawer.flush();
  console.log("Flushed drawer.");

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

// Show the modal
function showModal(type: string, title: string, message: string) {
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
function closeModal() {
  if (modal) {
    // Remove error class from modal.
    modal.classList.remove('error');
    modal.classList.remove('info');
    modal.style.display = 'none';
  }
}

// Handle play button click
function onPlayButtonClick() {
  state.isPlaying = !state.isPlaying;

  if (state.isPlaying) {
    playButton.classList.add('paused');
  } else {
    playButton.classList.remove('paused');
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
    focusFullMap(mapPanel);
  } else {
    showModal(
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
      state.followTraceSelection = true;
      mapPanel.zoomLevel = 1/2;
      tracePanel.zoomLevel = 1;
      console.info("Selected object via query parameter:", state.selectedGridObject);
    } else {
      console.warn("Invalid selectedObjectId:", selectedObjectId);
    }
  }

  // Set the map pan and zoom.
  if (urlParams.get('mapPanX') !== null && urlParams.get('mapPanY') !== null) {
    const mapPanX = parseInt(urlParams.get('mapPanX') || "0");
    const mapPanY = parseInt(urlParams.get('mapPanY') || "0");
    mapPanel.panPos = new Vec2f(mapPanX, mapPanY);
  }
  if (urlParams.get('mapZoom') !== null) {
    mapPanel.zoomLevel = parseFloat(urlParams.get('mapZoom') || "1");
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

scrubber.addEventListener('input', onScrubberChange);
playButton.addEventListener('click', onPlayButtonClick);

window.addEventListener('dragenter', preventDefaults, false);
window.addEventListener('dragleave', preventDefaults, false);
window.addEventListener('dragover', preventDefaults, false);
window.addEventListener('drop', handleDrop, false);

window.addEventListener('load', async () => {
  drawer = new Drawer(globalCanvas);

  // Use local atlas texture.
  const atlasImageUrl = 'dist/atlas.png';
  const atlasJsonUrl = 'dist/atlas.json';

  const success = await drawer.init(atlasJsonUrl, atlasImageUrl);
  if (!success) {
    showModal(
      "error",
      "Initialization failed",
      "Please check the console for more information."
    );
    return;
  } else {
    console.log("Drawer initialized successfully.");
  }

  await parseUrlParams();

  requestFrame();
});
