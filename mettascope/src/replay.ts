import { updateAgentTable } from './agentpanel.js'
import * as Common from './common.js'
import { ctx, html, state, ui } from './common.js'
import { onResize, requestFrame, updateStep } from './main.js'
import { focusFullMap } from './worldmap.js'

/** This represents a sequence of values sort of like a movie timeline. */
export class Sequence<T> {
  private value: T | null = null
  private sequence: T[] | null = null
  constructor(defaultValue: T) {
    this.value = defaultValue
  }

  /** Expands a sequence of values. */
  expand(data: any, numSteps: number, defaultValue: T) {
    if (data == null || data == undefined) {
      // Use the default value.
      this.value = defaultValue
      return
    } else if (data instanceof Array) {
      // For coordinates, we need to expand the sequence.
      if (data.length == 0) {
        this.value = defaultValue
        return
      } else if (Array.isArray(this.value) && data.length > 0 && !Array.isArray(data[1])) {
        // Its just a single array like value.
        this.value = data as T
      } else {
        // Expand the sequence.
        // A sequence of pairs is expanded to a sequence of values.
        var expanded: any[] = []
        var i = 0
        var j = 0
        var v: any = null
        for (i = 0; i < numSteps; i++) {
          if (j < data.length && data[j][0] == i) {
            v = data[j][1]
            if (v == null || v == undefined) {
              v = defaultValue
            }
            j++
          }
          expanded.push(v)
        }
        this.sequence = expanded
      }
    } else {
      // A single value is a valid sequence.
      this.value = data as T
    }
  }

  /** Gets a value from the sequence at current or specified step. */
  get(atStep: number = -1): T {
    if (atStep == -1) {
      atStep = state.step
    }
    if (this.sequence == null) {
      return this.value as T
    } else {
      return this.sequence[atStep] as T
    }
  }

  /** Adds a value to the sequence. */
  add(value: T) {
    if (this.sequence == null) {
      this.value = value as T
    } else {
      this.sequence.push(value)
    }
  }

  /** Checks if the sequence is a sequence of values. */
  isSequence(): boolean {
    return this.sequence != null
  }

  /** Checks if the sequence is a single value . */
  isValue(): boolean {
    return this.sequence == null
  }
}

// Entity and replay conform version 2 of the replay_spec.md.
export class Entity {
  // Common keys.
  id: number = 0
  typeId: number = 0
  groupId: number = 0
  agentId: number = 0
  location: Sequence<[number, number, number]> = new Sequence([0, 0, 0])
  orientation: Sequence<number> = new Sequence(0)
  inventory: Sequence<[number, number][]> = new Sequence<[number, number][]>([])
  inventoryMax: number = 0
  color: Sequence<number> = new Sequence(0)

  // Agent specific keys.
  actionId: Sequence<number> = new Sequence(0)
  actionParameter: Sequence<number> = new Sequence(0)
  actionSuccess: Sequence<boolean> = new Sequence(false)
  currentReward: Sequence<number> = new Sequence(0)
  totalReward: Sequence<number> = new Sequence(0)
  isFrozen: Sequence<boolean> = new Sequence(false)
  frozenProgress: Sequence<number> = new Sequence(0)
  frozenTime: number = 0
  visionSize: number = 0

  // Building specific keys.
  recipeInput: [number, number][] = []
  recipeOutput: [number, number][] = []
  recipeMax: number = 0
  productionProgress: Sequence<number> = new Sequence(0)
  productionTime: number = 0
  cooldownProgress: Sequence<number> = new Sequence(0)
  cooldownTime: number = 0

  // Gain map for the agent.
  gainMap: Map<number, number>[] = []
  isAgent: boolean = false
}

export class Replay {
  version: number = 0
  numAgents: number = 0
  maxSteps: number = 0
  mapSize: [number, number] = [0, 0]
  fileName: string = ''
  typeNames: string[] = []
  actionNames: string[] = []
  itemNames: string[] = []
  groupNames: string[] = []
  objects: Entity[] = []
  rewardSharingMatrix: number[][] = []
  agents: Entity[] = []

  // Generated data.
  typeImages: string[] = []
  actionImages: string[] = []
  resourceImages: string[] = []
  objectImages: string[] = []
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
  console.log('Fixing replay...')

  // Create action image mappings for faster access.
  state.replay.actionImages = []
  for (const actionName of state.replay.actionNames) {
    const path = `trace/${actionName}.png`
    if (ctx.hasImage(path)) {
      state.replay.actionImages.push(path)
    } else {
      console.warn('Action not supported: ', path)
      state.replay.actionImages.push('trace/unknown.png')
    }
  }

  // Create object image mappings for faster access.
  state.replay.objectImages = []
  for (const typeName of state.replay.typeNames) {
    const path = `objects/${typeName}.png`
    if (ctx.hasImage(path)) {
      state.replay.objectImages.push(path)
    } else {
      console.warn('Object not supported: ', path)
      state.replay.objectImages.push('objects/unknown.png')
    }
  }

  // Create  resource image mappings for faster access.
  state.replay.resourceImages = []
  for (const resourceName of state.replay.itemNames) {
    const path = `resources/${resourceName}.png`
    if (ctx.hasImage(path)) {
      state.replay.resourceImages.push(path)
    } else {
      console.warn('Resource not supported: ', path)
      state.replay.resourceImages.push('resources/unknown.png')
    }
  }

  // Find all agents for faster access.
  state.replay.agents = []
  for (let i = 0; i < state.replay.numAgents; i++) {
    state.replay.agents.push(new Entity())
    for (const gridObject of state.replay.objects) {
      const typeId = gridObject.typeId
      const typeName = state.replay.typeNames[typeId]
      if (typeName === 'agent' && gridObject.agentId === i) {
        state.replay.agents[i] = gridObject
        gridObject.isAgent = true
      }
    }
  }

  // Compute gain/loss of for agents.
  for (const agent of state.replay.agents) {
    // Gain map exists for the duration of the replay.
    agent.gainMap = []
    // Gain map for step 0 is empty.
    {
      const inventory = agent.inventory.get(0)
      const gainMap = new Map<number, number>()
      if (inventory.length > 0) {
        for (const inventoryPair of inventory) {
          const inventoryId = inventoryPair[0]
          const inventoryAmount = inventoryPair[1]
          gainMap.set(inventoryId, inventoryAmount)
        }
      }
      agent.gainMap.push(gainMap)
    }
    // We compute the gain map for each step > 1.
    for (let step = 1; step < state.replay.maxSteps; step++) {
      const inventory = agent.inventory.get(step)
      const prevInventory = agent.inventory.get(step - 1)
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
  console.info('Converting replay from version 1 to version 2...')
  console.info('Replay data: ', replayData)
  let data: any = {
    version: 2,
  }
  data.action_names = replayData.action_names
  data.action_names = data.action_names.map((name: string) => {
    if (name === "put_recipe_items") {
      return "put_items"
    }
    if (name === "get_output") {
      return "get_items"
    }
    return name
  })
  if (replayData.inventory_items != null && replayData.inventory_items != undefined && replayData.inventory_items.length > 0) {
    data.item_names = replayData.inventory_items
  } else {
    data.item_names = ['ore.red', 'ore.blue', 'ore.green', 'battery', 'heart', 'armor', 'laser', 'blueprint']
  }
  data.type_names = replayData.object_types
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
  let maxX = 0
  let maxY = 0
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
      maxX = Math.max(maxX, x)
      maxY = Math.max(maxY, y)
    }

    let inventory = []
    for (let inventoryId = 0; inventoryId < data.item_names.length; inventoryId++) {
      let inventoryName = data.item_names[inventoryId]
      if ("inv:" + inventoryName in gridObject) {
        gridObject["inv:" + inventoryName] = expandSequenceV2(gridObject["inv:" + inventoryName], replayData.max_steps)
      }
      if ("agent:inv:" + inventoryName in gridObject) {
        gridObject["inv:" + inventoryName] = expandSequenceV2(gridObject["agent:inv:" + inventoryName], replayData.max_steps)
      }
    }
    for (let step = 0; step < replayData.max_steps; step++) {
      let inventoryList = []
      for (let inventoryId = 0; inventoryId < data.item_names.length; inventoryId++) {
        let inventoryName = data.item_names[inventoryId]
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
      object.is_object = true
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

  data.map_size = [maxX + 1, maxY + 1]
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

  state.replay = new Replay()
  state.replay.version = replayData.version
  state.replay.actionNames = replayData.action_names
  state.replay.itemNames = replayData.item_names
  state.replay.typeNames = replayData.type_names
  state.replay.numAgents = replayData.num_agents
  state.replay.maxSteps = replayData.max_steps
  state.replay.mapSize = replayData.map_size
  state.replay.fileName = replayData.file_name

  // Go through each grid object and expand its key sequence.
  for (const gridObject of replayData.objects) {
    let object = new Entity()
    object.id = gridObject.id
    object.typeId = gridObject.type_id
    object.location.expand(gridObject.location, replayData.max_steps, [0, 0, 0])
    object.orientation.expand(gridObject.orientation, replayData.max_steps, 0)
    console.log('gridObject.inventory: ', gridObject.inventory)
    object.inventory.expand(gridObject.inventory, replayData.max_steps, [])
    object.inventoryMax = gridObject.inventory_max
    object.color.expand(gridObject.color, replayData.max_steps, 0)

    if ("agent_id" in gridObject) {
      object.agentId = gridObject.agent_id
      object.groupId = gridObject.group_id
      object.isFrozen.expand(gridObject.is_frozen, replayData.max_steps, false)
      object.actionId.expand(gridObject.action_id, replayData.max_steps, 0)
      object.actionParameter.expand(gridObject.action_param, replayData.max_steps, 0)
      object.actionSuccess.expand(gridObject.action_success, replayData.max_steps, false)
      object.currentReward.expand(gridObject.current_reward, replayData.max_steps, 0)
      object.totalReward.expand(gridObject.total_reward, replayData.max_steps, 0)
      object.frozenProgress.expand(gridObject.frozen_progress, replayData.max_steps, 0)
      object.frozenTime = gridObject.frozen_time
      object.visionSize = Common.DEFAULT_VISION_SIZE // TODO Fix this
    }

    if ("recipe_input" in gridObject) {
      object.recipeInput = gridObject.recipe_input
      object.recipeOutput = gridObject.recipe_output
      object.recipeMax = gridObject.recipe_max
      object.productionProgress.expand(gridObject.production_progress, replayData.max_steps, 0)
      object.productionTime = gridObject.production_time
      object.cooldownProgress.expand(gridObject.cooldown_progress, replayData.max_steps, 0)
      object.cooldownTime = gridObject.cooldown_time
    }

    state.replay.objects.push(object)
  }

  fixReplay()


  console.log('Replay data: ', state.replay)

  if (state.replay.fileName) {
    html.fileName.textContent = state.replay.fileName
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

  state.replay.maxSteps = Math.max(state.replay.maxSteps, step + 1)

  for (const gridObject of replayStep.objects) {
    // Grid objects are 1-indexed.
    const index = gridObject.id - 1
    // Ensure that the grid object exists.
    while (state.replay.objects.length <= index) {
      state.replay.objects.push(new Entity())
    }

    const object = state.replay.objects[index]
    object.id = gridObject.id
    object.typeId = gridObject.type_id
    object.groupId = gridObject.group_id
    object.agentId = gridObject.agent_id
    object.visionSize = gridObject.vision_size
    object.isFrozen.add(gridObject.is_frozen)
    object.location.add(gridObject.location)
    object.orientation.add(gridObject.orientation)
    object.inventory.add(gridObject.inventory)
    object.color.add(gridObject.color)
    if ("agent_id" in gridObject) {
      object.actionId.add(gridObject.action_id)
      object.actionParameter.add(gridObject.action_param)
      object.actionSuccess.add(gridObject.action_success)
      object.currentReward.add(gridObject.current_reward)
      object.totalReward.add(gridObject.total_reward)
      object.isFrozen.add(gridObject.isFrozen)
      object.frozenProgress.add(gridObject.frozen_progress)
    }
    if ("recipe_input" in gridObject) {
      object.productionProgress.add(gridObject.production_progress)
      object.cooldownProgress.add(gridObject.cooldown_progress)
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
  const agentId = state.selectedGridObject.agentId
  if (agentId != null) {
    const actionId = state.replay.actionNames.indexOf(actionName)
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
  if (state.replay.typeNames.includes(key)) {
    const idx = state.replay.typeNames.indexOf(key)
    return `data/atlas/${state.replay.typeImages[idx]}`
  }
  return `data/ui/table/${key.replace('agent:', '')}.png`
}
