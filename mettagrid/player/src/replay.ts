import * as Common from './common.js';
import { ui, state, html, ctx } from './common.js';
import { focusFullMap, requestFrame } from './drawing.js';

// Gets an attribute from a grid object respecting the current step.
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
export async function fetchReplay(replayUrl: string) {
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
    Common.showModal("error", "Error fetching replay", "Message: " + error);
  }
}

// Read a file from drag and drop.
export async function readFile(file: File) {
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
    Common.showModal("error", "Error reading file", "Message: " + error);
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
  // Example: 3 -> ["objects/altar.png", "objects/altar.empty.png"]
  // Example: 1 -> ["objects/wall.png", "objects/wall.png"]
  state.replay.object_images = []
  for (let i = 0; i < state.replay.object_types.length; i++) {
    const typeName = state.replay.object_types[i];
    var image = "objects/" + typeName + ".png";
    if (!ctx.hasImage(image)) {
      console.warn("Object not supported: ", typeName);
      image = "objects/unknown.png";
    }
    var imageEmpty = "objects/" + typeName + ".empty.png";
    if (!ctx.hasImage(imageEmpty)) {
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
          if(ctx.hasImage("resources/" + type + ".png")) {
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
  state.replay.map_size[0] = 0;
  state.replay.map_size[1] = 0;
  for (const gridObject of state.replay.grid_objects) {
    let x = getAttr(gridObject, "c") + 1;
    let y = getAttr(gridObject, "r") + 1;
    state.replay.map_size[0] = Math.max(state.replay.map_size[0], x);
    state.replay.map_size[1] = Math.max(state.replay.map_size[1], y);
  }

  console.info("replay: ", state.replay);

  // Set the scrubber max value to the max steps.
  html.scrubber.max = (state.replay.max_steps - 1).toString();

  Common.closeModal();
  focusFullMap(ui.mapPanel);
  requestFrame();
}
