import { updateAgentTable } from './agentpanel.js'
import * as Common from './common.js'
import { ctx, html, state, ui } from './common.js'
import { onResize, requestFrame, updateStep } from './main.js'
import { focusFullMap } from './worldmap.js'

/** Gets an attribute from a grid object, respecting the current step. */
export function getAttr(obj: any, attr: string, atStep = -1, defaultValue = 0): any {
  const prop = obj[attr]
  if (prop === undefined) {
    return defaultValue
  }
  if (!Array.isArray(prop) || prop.length != state.replay.max_steps) {
    return prop // This must be a constant that does not change over time.
  }
  return prop[atStep === -1 ? state.step : atStep] // When the step is not passed in, use the global step.
}

/** Decompresses a stream. Used for compressed JSON from fetch or drag-and-drop. */
async function decompressStream(stream: ReadableStream<Uint8Array>): Promise<string> {
  const decompressionStream = new DecompressionStream('deflate')
  const decompressedStream = stream.pipeThrough(decompressionStream)
  const reader = decompressedStream.getReader()
  const chunks: Uint8Array[] = []
  let result: ReadableStreamReadResult<Uint8Array>

  result = await reader.read()
  while (!result.done) {
    chunks.push(result.value)
    result = await reader.read()
  }

  const totalLength = chunks.reduce((acc, val) => acc + val.length, 0)
  const flattenedChunks = new Uint8Array(totalLength)
  let offset = 0
  for (const chunk of chunks) {
    flattenedChunks.set(chunk, offset)
    offset += chunk.length
  }
  const decoder = new TextDecoder()
  return decoder.decode(flattenedChunks)
}

/** Loads the replay from a URL. */
export async function fetchReplay(replayUrl: string) {
  // If it's an S3 URL, we can convert it to an HTTP URL.
  const s3Prefix = 's3://softmax-public/'
  let httpUrl = replayUrl
  if (replayUrl.startsWith(s3Prefix)) {
    const httpPrefix = 'https://softmax-public.s3.us-east-1.amazonaws.com/'
    httpUrl = httpPrefix + replayUrl.slice(s3Prefix.length)
    console.info('Converted S3 url to http url: ', httpUrl)
  }

  try {
    const response = await fetch(httpUrl)
    if (!response.ok) {
      throw new Error('Network response was not ok')
    }
    if (response.body === null) {
      throw new Error('Response body is null')
    }
    // Check the Content-Type header.
    const contentType = response.headers.get('Content-Type')
    console.info('Content-Type: ', contentType)
    if (contentType === 'application/json') {
      const replayData = await response.text()
      loadReplayText(replayUrl, replayData)
    } else if (contentType === 'application/x-compress' || contentType === 'application/octet-stream') {
      // This is compressed JSON.
      const decompressedData = await decompressStream(response.body)
      loadReplayText(replayUrl, decompressedData)
    } else {
      throw new Error(`Unsupported content type: ${contentType}`)
    }
  } catch (error) {
    Common.showModal('error', 'Error fetching replay', `Message: ${error}`)
  }
}

/** Reads a file from drag-and-drop. */
export async function readFile(file: File) {
  try {
    const contentType = file.type
    console.info('Content-Type: ', contentType)
    if (contentType === 'application/json') {
      loadReplayText(file.name, await file.text())
    } else if (contentType === 'application/x-compress' || contentType === 'application/octet-stream') {
      // This is compressed JSON.
      const decompressedData = await decompressStream(file.stream())
      loadReplayText(file.name, decompressedData)
    }
  } catch (error) {
    Common.showModal('error', 'Error reading file', `Message: ${error}`)
  }
}

/**
 * Expands a sequence of values.
 * Example: [[0, value1], [2, value2], ...] -> [value1, value1, value2, ...]
 */
// [[0, value1], [2, value2], ...] -> [value1, value1, value2, ...]
function expandSequence(sequence: any[], numSteps: number): any[] {
  const expanded: any[] = []
  let i = 0
  let j = 0
  let v: any = null
  for (i = 0; i < numSteps; i++) {
    if (j < sequence.length && sequence[j][0] === i) {
      v = sequence[j][1]
      j++
    }
    expanded.push(v)
  }
  return expanded
}

// Removes a prefix from a string.
function removePrefix(str: string, prefix: string) {
  return str.startsWith(prefix) ? str.slice(prefix.length) : str
}

// Removes a suffix from a string.
function removeSuffix(str: string, suffix: string) {
  return str.endsWith(suffix) ? str.slice(0, -suffix.length) : str
}

// Loads the replay text.
function loadReplayText(url: string, replayData: string) {
  loadReplayJson(url, JSON.parse(replayData))
}

// Replays can be in many different formats, with stuff missing or broken.
// This function fixes the replay to be in a consistent format,
// adding missing keys, recomputing invalid values, etc.
// It also creates some internal data structures for faster access to images.
function fixReplay() {
  // // Fix "agent.agent" -> "agent".
  // for (let i = 0; i < state.replay.type_names.length; i++) {
  //   if (state.replay.type_names[i] === 'agent.agent') {
  //     state.replay.type_names[i] = 'agent'
  //   }
  // }

  // Create action image mappings for faster access.
  state.replay.action_images = []
  for (const actionName of state.replay.action_names) {
    const path = `trace/${actionName}.png`
    if (ctx.hasImage(path)) {
      state.replay.action_images.push(path)
    } else {
      console.warn('Action not supported: ', path)
      state.replay.action_images.push('trace/unknown.png')
    }
  }

  // Create a list of all keys that objects can have.
  state.replay.all_keys = new Set()
  for (const gridObject of state.replay.objects) {
    for (const key in gridObject) {
      state.replay.all_keys.add(key)
    }
  }

  // Create an object image mapping for faster access.
  // Example: 3 -> ["objects/altar.png", "objects/altar.item.png", "objects/altar.color.png"]
  // Example: 1 -> ["objects/unknown.png", "objects/unknown.item.png", "objects/unknown.color.png"]
  state.replay.object_images = []
  state.replay.type_names.forEach((originalTypeName: string) => {
    let typeName = originalTypeName
    // Remove known color suffixes.
    for (const colorName of Common.COLORS.keys()) {
      if (typeName.endsWith(`_${colorName}`)) {
        typeName = typeName.slice(0, -colorName.length - 1)
        break
      }
    }
    let image = `objects/${typeName}.png`
    let imageItem = `objects/${typeName}.item.png`
    let imageColor = `objects/${typeName}.color.png`
    if (!ctx.hasImage(image)) {
      console.warn(`Object name not supported: "${typeName}"`)
      // Use the "unknown" image.
      image = 'objects/unknown.png'
      imageItem = 'objects/unknown.item.png'
      imageColor = 'objects/unknown.color.png'
    }
    state.replay.object_images.push([image, imageItem, imageColor])
  })

  // Create a resource inventory mapping for faster access.
  // Example: "inv:heart" -> ["resources/heart.png", [1, 1, 1, 1]]
  // Example: "inv:ore.red" -> ["resources/ore.red.png", [1, 1, 1, 1]]
  // Example: "agent:inv:heart.blue" -> ["resources/heart.png", [0, 0, 1, 1]]
  // Example: "inv:cat_food.red" -> ["resources/unknown.png", [1, 0, 0, 1]]
  state.replay.resource_inventory = new Map()
  for (const key of state.replay.all_keys) {
    if (key.startsWith('inv:') || key.startsWith('agent:inv:')) {
      let type: string = key
      type = removePrefix(type, 'inv:')
      type = removePrefix(type, 'agent:inv:')
      let color = [1, 1, 1, 1] // Default to white.
      for (const [colorName, colorValue] of Common.COLORS) {
        if (type.endsWith(colorName)) {
          if (ctx.hasImage(`resources/${type}.png`)) {
            // Use the resource.color.png with a white color.
            break
          }
          // Use the resource.png with a specific color.
          type = removeSuffix(type, `.${colorName}`)
          color = colorValue as number[]
          if (!ctx.hasImage(`resources/${type}.png`)) {
            // Use the unknown.png with a specific color.
            console.warn('Resource not supported: ', type)
            type = 'unknown'
          }
        }
      }
      const image = `resources/${type}.png`
      state.replay.resource_inventory.set(key, [image, color])
    }
  }


  // Compute gain/loss of for agents.
  for (const agent of state.replay.agents) {
    // Gain map exists for the duration of the replay.
    agent.gainMap = []
    // Gain map for step 0 is empty.
    {
      const inventory = getAttr(agent, 'inventory', 0)
      const gainMap = new Map<number, number>()
      if (inventory != null && inventory != 0) {
        for (const inventoryPair of inventory) {
          const inventoryId = inventoryPair[0]
          const inventoryAmount = inventoryPair[1]
          gainMap.set(inventoryId, inventoryAmount)
        }
      }
      agent.gainMap.push(gainMap)
    }
    // We compute the gain map for each step > 1.
    for (let step = 1; step < state.replay.max_steps; step++) {
      const inventory = getAttr(agent, 'inventory', step)
      const prevInventory = getAttr(agent, 'inventory', step - 1)
      const gainMap = new Map<number, number>()
      // We add current's frame inventory to the gain map.
      for (const inventoryPair of inventory) {
        const inventoryId = inventoryPair[0]
        const inventoryAmount = inventoryPair[1]
        if (gainMap.has(inventoryId)) {
          gainMap.set(inventoryId, gainMap.get(inventoryId)! + inventoryAmount)
        } else {
          gainMap.set(inventoryId, inventoryAmount)
        }
      }
      // We subtract previous frame's inventory from the gain map.
      for (const inventoryPair of prevInventory) {
        const inventoryId = inventoryPair[0]
        const inventoryAmount = inventoryPair[1]
        if (gainMap.has(inventoryId)) {
          gainMap.set(inventoryId, gainMap.get(inventoryId)! - inventoryAmount)
        } else {
          gainMap.set(inventoryId, -inventoryAmount)
        }
      }
      // Clean zeros out of the gain map.
      for (const [inventoryId, inventoryAmount] of gainMap) {
        if (inventoryAmount === 0) {
          gainMap.delete(inventoryId)
        }
      }
      agent.gainMap.push(gainMap)
    }
  }
}

/** Converts a replay from version 1 to version 2. */
function convertReplayV1ToV2(replayData: any) {
  console.log('Converting replay from version 1 to version 2...')
  console.log('Replay data: ', replayData)
  let data: any = {
    version: 2,
  }
  data.action_names = replayData.action_names
  data.item_names = replayData.inventory_items
  data.type_names = replayData.object_types
  data.map_size = replayData.map_size
  data.num_agents = replayData.num_agents
  data.max_steps = replayData.max_steps

  /** Gets an attribute from a grid object, respecting the current step. */
  function getAttrV1(obj: any, attr: string, atStep = -1, defaultValue = 0): any {
    const prop = obj[attr]
    if (prop === undefined) {
      return defaultValue
    }
    if (!Array.isArray(prop)) {
      return prop // This must be a constant that does not change over time.
    }
    return prop[atStep === -1 ? state.step : atStep] // When the step is not passed in, use the global step.
  }

  function expandSequenceV2(sequence: any[], numSteps: number): any[] {
    if (!Array.isArray(sequence)) {
      return sequence
    }
    const expanded: any[] = []
    let i = 0
    let j = 0
    let v: any = null
    for (i = 0; i < numSteps; i++) {
      if (j < sequence.length && sequence[j][0] === i) {
        v = sequence[j][1]
        j++
      }
      expanded.push(v)
    }
    return expanded
  }

  data.objects = []
  for (const gridObject of replayData.grid_objects) {
    let location = []
    gridObject["c"] = expandSequenceV2(gridObject["c"], replayData.max_steps)
    gridObject["r"] = expandSequenceV2(gridObject["r"], replayData.max_steps)
    gridObject["layer"] = expandSequenceV2(gridObject["layer"], replayData.max_steps)
    for (let step = 0; step < replayData.max_steps; step++) {
      let x = getAttrV1(gridObject, 'c', step, 0)
      let y = getAttrV1(gridObject, 'r', step, 0)
      let z = getAttrV1(gridObject, 'layer', step, 0)
      location.push([step, [x, y, z]])
    }

    let inventory = []
    for (let inventoryId = 0; inventoryId < replayData.inventory_items.length; inventoryId++) {
      let inventoryName = replayData.inventory_items[inventoryId]
      if ("inv:" + inventoryName in gridObject) {
        console.log("Expanding inventory: ", "inv:" + inventoryName)
        gridObject["inv:" + inventoryName] = expandSequenceV2(gridObject["inv:" + inventoryName], replayData.max_steps)
      }
      if ("agent:inv:" + inventoryName in gridObject) {
        console.log("Expanding inventory: ", "agent:inv:" + inventoryName)
        gridObject["inv:" + inventoryName] = expandSequenceV2(gridObject["agent:inv:" + inventoryName], replayData.max_steps)
      }
    }
    for (let step = 0; step < replayData.max_steps; step++) {
      let inventoryList = []
      for (let inventoryId = 0; inventoryId < replayData.inventory_items.length; inventoryId++) {
        let inventoryName = replayData.inventory_items[inventoryId]
        let inventoryAmount = getAttrV1(gridObject, "inv:" + inventoryName, step, 0)
        if (inventoryAmount != 0) {
          inventoryList.push([inventoryId, inventoryAmount])
        }
      }
      inventory.push([step, inventoryList])
    }

    let object: any = {
      id: gridObject.id,
      type_id: gridObject.type,
      location: location,
      inventory: inventory,
      orientation: gridObject.orientation,
    }

    if (gridObject.agent_id != null) {
      object.agent_id = gridObject.agent_id
      object.is_frozen = gridObject["agent:frozen"]
      object.color = gridObject["agent:color"]
      object.action_success = gridObject["action_success"]
      object.group_id = gridObject["agent:group"]
      object.orientation = gridObject["agent:orientation"]
      object.hp = gridObject["agent:hp"]
      object.current_reward = gridObject["agent:reward"]
      object.total_reward = gridObject["agent:total_reward"]

      let action_id = []
      let action_param = []
      gridObject["action"] = expandSequenceV2(gridObject["action"], replayData.max_steps)
      for (let step = 0; step < replayData.max_steps; step++) {
        let action = getAttrV1(gridObject, 'action', step)
        if (action != null) {
          action_id.push([step, action[0]])
          action_param.push([step, action[1]])
        }
      }
      object.action_id = action_id
      object.action_param = action_param
    }
    data.objects.push(object)
  }
  console.log('Converted replay data: ', data)
  return data
}

/** Loads a replay from a JSON object. */
function loadReplayJson(url: string, replayJson: any) {
  let replayData = replayJson

  // If the replay is version 1, convert it to version 2.
  if (replayData.version === 1) {
    replayData = convertReplayV1ToV2(replayData)
  }

  if (replayData.version !== 2) {
    Common.showModal('error', 'Error loading replay', `Unsupported replay version: ${replayData.version}`)
    return
  }

  state.replay = replayData

  // Go through each grid object and expand its key sequence.
  for (const gridObject of state.replay.objects) {
    for (const key in gridObject) {
      if (Array.isArray(gridObject[key]) && gridObject[key][0].length == 2) {
        gridObject[key] = expandSequence(gridObject[key], state.replay.max_steps)
      }
    }
  }

  // Find all agents for faster access.
  state.replay.agents = []
  for (let i = 0; i < state.replay.num_agents; i++) {
    state.replay.agents.push({})
    for (const gridObject of state.replay.objects) {
      if (gridObject.agent_id === i) {
        state.replay.agents[i] = gridObject
      }
    }
  }

  fixReplay()

  if (state.replay.file_name) {
    html.fileName.textContent = state.replay.file_name
  } else {
    html.fileName.textContent = url.split('/').pop() || 'unknown'
  }

  Common.closeModal()
  focusFullMap(ui.mapPanel)
  updateAgentTable()
  onResize()
  requestFrame()
}

/** Loads a single step of a replay. */
export function loadReplayStep(replayStep: any) {
  // This gets us a simple replay step that we can overwrite.

  // Update the grid objects.
  const step = replayStep.step

  state.replay.max_steps = Math.max(state.replay.max_steps, step + 1)

  for (const gridObject of replayStep.objects) {
    // Grid objects are 1-indexed.
    const index = gridObject.id - 1
    for (const key in gridObject) {
      const value = gridObject[key]
      // Ensure that the grid object exists.
      while (state.replay.objects.length <= index) {
        state.replay.objects.push({})
      }
      // Ensure that the key exists.
      if (state.replay.objects[index][key] === undefined || state.replay.objects[index][key] === null) {
        state.replay.objects[index][key] = []
        while (state.replay.objects[index][key].length <= step) {
          state.replay.objects[index][key].push(null)
        }
      }

      state.replay.objects[index][key][step] = value

      if (key === 'agent_id') {
        // Update the agent.
        while (state.replay.agents.length <= value) {
          state.replay.agents.push({})
        }
        state.replay.agents[value] = state.replay.objects[index]
      }
    }
    // Make sure that the keys that don't exist in the update are set to null too.
    for (const key in state.replay.objects[index]) {
      if (gridObject[key] === undefined) {
        state.replay.objects[index][key][step] = null
      }
    }
  }

  fixReplay()

  updateStep(step)

  requestFrame()
}

/** Initializes the WebSocket connection. */
export function initWebSocket(wsUrl: string) {
  state.ws = new WebSocket(wsUrl)
  state.ws.onmessage = (event) => {
    const data = JSON.parse(event.data)
    if (data.type === 'replay') {
      loadReplayJson(wsUrl, data.replay)
      Common.closeModal()
      html.actionButtons.classList.remove('hidden')
    } else if (data.type === 'replay_step') {
      loadReplayStep(data.replay_step)
    } else if (data.type === 'memory_copied') {
      navigator.clipboard.writeText(JSON.stringify(data.memory))
    }
  }
  state.ws.onopen = () => {
    Common.showModal('info', 'Starting environment', 'Please wait while live environment is starting...')
  }
  state.ws.onclose = () => {
    Common.showModal('error', 'WebSocket closed', 'Please check your connection and refresh this page.')
  }
  state.ws.onerror = (event) => {
    Common.showModal('error', 'WebSocket error', `Websocket error: ${event}`)
  }
}

/** Sends an action to the server. */
export function sendAction(actionName: string, actionParam: number) {
  if (state.ws === null) {
    console.error('WebSocket is not connected')
    return
  }
  const agentId = getAttr(state.selectedGridObject, 'agent_id')
  if (agentId != null) {
    const actionId = state.replay.action_names.indexOf(actionName)
    if (actionId === -1) {
      console.error('Action not found: ', actionName)
      return
    }
    state.ws.send(
      JSON.stringify({
        type: 'action',
        agent_id: agentId,
        action_id: actionId,
        action_param: actionParam,
      })
    )
  } else {
    console.error('No selected grid object')
  }
}

/**
 * Capitalize the first letter of every word in a string.
 * Example: "hello world" -> "Hello World"
 */
function capitalize(str: string) {
  return str
    .split(' ')
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ')
}

/** Gets a nice english name of a resource, type or any other property. */
export function propertyName(key: string) {
  return capitalize(key.replace('inv:', '').replace('agent:', '').replace('.', ' ').replace('_', ' '))
}

/** Gets the icon of a resource, type or any other property. */
export function propertyIcon(key: string) {
  if (state.replay.type_names.includes(key)) {
    const idx = state.replay.type_names.indexOf(key)
    return `data/atlas/${state.replay.object_images[idx][0]}`
  }
  if (key.startsWith('inv:') || key.startsWith('agent:inv:')) {
    return `data/atlas/resources/${key.replace('inv:', '').replace('agent:', '')}.png`
  }
  return `data/ui/table/${key.replace('agent:', '')}.png`
}
