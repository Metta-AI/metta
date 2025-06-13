import * as Common from './common.js';
import { ui, state, html, ctx } from './common.js';
import { focusFullMap, requestFrame } from './worldmap.js';
import { onResize, updateStep } from './main.js';
import { updateAgentTable } from './agentpanel.js';

/** Gets an attribute from a grid object respecting the current step. */
export function getAttr(obj: any, attr: string, atStep = -1, defaultValue = 0): any {
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

/** Decompress a stream, used for compressed JSON from fetch or drag and drop. */
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

/** Load the replay from a URL. */
export async function fetchReplay(replayUrl: string) {

  // If its an S3 url, we can convert it to a http url.
  const s3Prefix = "s3://softmax-public/";
  let httpUrl = replayUrl;
  if (replayUrl.startsWith(s3Prefix)) {
    const httpPrefix = "https://softmax-public.s3.us-east-1.amazonaws.com/";
    httpUrl = httpPrefix + replayUrl.slice(s3Prefix.length);
    console.info("Converted S3 url to http url: ", httpUrl);
  }

  try {
    const response = await fetch(httpUrl);
    if (!response.ok) {
      throw new Error("Network response was not ok");
    }
    if (response.body === null) {
      throw new Error("Response body is null");
    }
    // Check the Content-Type header
    const contentType = response.headers.get('Content-Type');
    console.info("Content-Type: ", contentType);
    if (contentType === "application/json") {
      let replayData = await response.text();
      loadReplayText(replayUrl, replayData);
    } else if (contentType === "application/x-compress" || contentType === "application/octet-stream") {
      // Compressed JSON.
      const decompressedData = await decompressStream(response.body);
      loadReplayText(replayUrl, decompressedData);
    } else {
      throw new Error("Unsupported content type: " + contentType);
    }
  } catch (error) {
    Common.showModal("error", "Error fetching replay", "Message: " + error);
  }
}

/** Read a file from drag and drop. */
export async function readFile(file: File) {
  try {
    const contentType = file.type;
    console.info("Content-Type: ", contentType);
    if (contentType === "application/json") {
      loadReplayText(file.name, await file.text());
    } else if (contentType === "application/x-compress" || contentType === "application/octet-stream") {
      // Compressed JSON.
      const decompressedData = await decompressStream(file.stream());
      loadReplayText(file.name, decompressedData);
    }
  } catch (error) {
    Common.showModal("error", "Error reading file", "Message: " + error);
  }
}

/**
 * Expand a sequence of values.
 * Example: [[0, value1], [2, value2], ...] -> [value1, value1, value2, ...]
 */
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
async function loadReplayText(url: string, replayData: string) {
  loadReplayJson(url, JSON.parse(replayData));
}

// Replays can be in many different formats, with stuff missing or broken.
// This function fixes the replay to be in a consistent format.
// Adding missing keys, recomputing invalid values. etc...
// It also creates some internal data structures for faster access to images.
function fixReplay() {
  // Create action image mappings for faster access.
  state.replay.action_images = [];
  for (const actionName of state.replay.action_names) {
    let path = "trace/" + actionName + ".png";
    if (ctx.hasImage(path)) {
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
  // Example: 3 -> ["objects/altar.png", "objects/altar.item.png", "objects/altar.color.png"]
  // Example: 1 -> ["objects/unknown.png", "objects/unknown.item.png", "objects/unknown.color.png"]
  state.replay.object_images = []
  for (let i = 0; i < state.replay.object_types.length; i++) {
    const typeName = state.replay.object_types[i];
    var image = "objects/" + typeName + ".png";
    var imageItem = "objects/" + typeName + ".item.png";
    var imageColor = "objects/" + typeName + ".color.png";
    if (!ctx.hasImage(image)) {
      console.warn("Object not supported: ", typeName);
      // Use the unknown image.
      image = "objects/unknown.png";
      imageItem = "objects/unknown.item.png";
      imageColor = "objects/unknown.color.png";
    }
    state.replay.object_images.push([image, imageItem, imageColor]);
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
          if (ctx.hasImage("resources/" + type + ".png")) {
            // Use the resource.color.png with white color.
            break;
          } else {
            // Use the resource.png with specific color.
            type = removeSuffix(type, "." + colorName);
            color = colorValue as number[];
            if (!ctx.hasImage("resources/" + type + ".png")) {
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

  // Map size is not to be trusted. Recompute map size just in case.
  let oldMapSize = [state.replay.map_size[0], state.replay.map_size[1]];
  state.replay.map_size[0] = 1;
  state.replay.map_size[1] = 1;
  for (const gridObject of state.replay.grid_objects) {
    let x = getAttr(gridObject, "c") + 1;
    let y = getAttr(gridObject, "r") + 1;
    state.replay.map_size[0] = Math.max(state.replay.map_size[0], x);
    state.replay.map_size[1] = Math.max(state.replay.map_size[1], y);
  }
  if (oldMapSize[0] != state.replay.map_size[0] || oldMapSize[1] != state.replay.map_size[1]) {
    // Map size changed, update the map.
    console.info("Map size changed to: ", state.replay.map_size[0], "x", state.replay.map_size[1]);
    focusFullMap(ui.mapPanel);
    // Force a resize to update the minimap panel.
    onResize();
  }

  console.info("replay: ", state.replay);

}

/** Load a replay from a JSON object. */
async function loadReplayJson(url: string, replayData: any) {
  state.replay = replayData;

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
    state.replay.agents.push({});
    for (const gridObject of state.replay.grid_objects) {
      if (gridObject["agent_id"] == i) {
        state.replay.agents[i] = gridObject;
      }
    }
  }

  fixReplay();

  if (state.replay.file_name) {
    html.fileName.textContent = state.replay.file_name;
  } else {
    html.fileName.textContent = url.split("/").pop() || "unknown";
  }

  Common.closeModal();
  focusFullMap(ui.mapPanel);
  updateAgentTable();
  onResize();
  requestFrame();
}

/** Load a single step of a replay. */
export function loadReplayStep(replayStep: any) {
  // This gets us a simple replay step that we can overwrite.

  // Update the grid objects.
  const step = replayStep.step;

  state.replay.max_steps = Math.max(state.replay.max_steps, step + 1);


  for (const gridObject of replayStep.grid_objects) {
    // Grid objects are 1-indexed.
    const index = gridObject.id - 1;
    for (const key in gridObject) {
      const value = gridObject[key];
      // Ensure the grid object exists.
      while (state.replay.grid_objects.length <= index) {
        state.replay.grid_objects.push({});
      }
      // Ensure the key exists.
      if (state.replay.grid_objects[index][key] === undefined) {
        state.replay.grid_objects[index][key] = [];
        while (state.replay.grid_objects[index][key].length <= step) {
          state.replay.grid_objects[index][key].push(null);
        }
      }
      state.replay.grid_objects[index][key][step] = value;


      if (key == "agent_id") {
        // Update the agent.
        while (state.replay.agents.length <= value) {
          state.replay.agents.push({});
        }
        state.replay.agents[value] = state.replay.grid_objects[index];
      }
    }
  }

  fixReplay()

  updateStep(step)

  requestFrame();
}

/** Initialize the WebSocket connection. */
export function initWebSocket(wsUrl: string) {
  state.ws = new WebSocket(wsUrl);
  state.ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.info("Received message: ", data.type);
    if (data.type === "replay") {
      loadReplayJson(wsUrl, data.replay);
      Common.closeModal();
      html.actionButtons.classList.remove("hidden");
    } else if (data.type === "replay_step") {
      loadReplayStep(data.replay_step);
    } else if (data.type === "message") {
      console.info("Received message: ", data.message);
    } else if (data.type === "memory_copied") {
      navigator.clipboard.writeText(JSON.stringify(data.memory));
    }
  };
  state.ws.onopen = () => {
    Common.showModal("info",
      "Starting environment",
      "Please wait while live environment is starting..."
    );
  };
  state.ws.onclose = () => {
    Common.showModal("error",
      "WebSocket closed",
      "Please check your connection and refresh this page."
    );
  };
  state.ws.onerror = (event) => {
    Common.showModal("error",
      "WebSocket error",
      "Websocket error: " + event
    );
  };
}

/** Send an action to the server. */
export function sendAction(actionName: string, actionParam: number) {
  if (state.ws === null) {
    console.error("WebSocket is not connected");
    return;
  }
  const agentId = getAttr(state.selectedGridObject, "agent_id");
  if (agentId != null) {
    const actionId = state.replay.action_names.indexOf(actionName);
    if (actionId == -1) {
      console.error("Action not found: ", actionName);
      return;
    }
    state.ws.send(JSON.stringify({
      type: "action",
      agent_id: agentId,
      action: [actionId, actionParam]
    }));
  } else {
    console.error("No selected grid object");
  }
}
