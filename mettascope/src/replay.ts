import * as Common from './common.js'
import { ui, state, html, ctx } from './common.js'
import { focusFullMap } from './worldmap.js'
import { onResize, updateStep, requestFrame } from './main.js'
import { updateAgentTable } from './agentpanel.js'


type Sequence<T> = [number, T][] | [number, T] | null

// Object and replay conform version 2 of the replay_spec.md.
export class Object {
  // Common keys.
  id: Sequence<number> = null
  typeId: Sequence<number> = null
  groupId: Sequence<number> = null
  agentId: Sequence<number> = null
  position: Sequence<[number, number]> = null
  rotation: Sequence<number> = null
  layer: Sequence<number> = null
  inventory: Sequence<number[]> = null
  inventoryMax: Sequence<number> = null
  color: Sequence<number> = null

  // Agent specific keys.
  actionId: Sequence<number> = null
  actionParameter: Sequence<number> = null
  actionSuccess: Sequence<boolean> = null
  currentReward: Sequence<number> = null
  totalReward: Sequence<number> = null
  frozen: Sequence<boolean> = null
  frozenProgress: Sequence<number> = null
  frozenTime: Sequence<number> = null

  // Building specific keys.
  recipeInput: Sequence<number> = null
  recipeOutput: Sequence<number> = null
  recipeMax: Sequence<number> = null
  productionProgress: Sequence<number> = null
  productionTime: Sequence<number> = null
  cooldownProgress: Sequence<number> = null
  cooldownTime: Sequence<number> = null

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
  objects: Object[] = []
  rewardSharingMatrix: number[][] = []
}

// Replay helper has lookups that speed up rendering.
class ReplayHelper {
  actionImages: string[] = []
  typeImages: string[] = []
  itemImages: string[] = []
}


/** Gets an attribute from a grid object, respecting the current step. */
export function getAttr<T>(sequence: Sequence<T>, atStep = -1): T {
  if (atStep == -1) {
    // When the step is not passed in, use the global step.
    atStep = state.step
  }
  if (sequence instanceof Array) {
    return sequence[atStep] as T
  } else {
    return sequence as T
  }
}

/** Decompresses a stream. Used for compressed JSON from fetch or drag-and-drop. */
async function decompressStream(stream: ReadableStream<Uint8Array>): Promise<string> {
  const decompressionStream = new DecompressionStream('deflate')
  const decompressedStream = stream.pipeThrough(decompressionStream)

  const reader = decompressedStream.getReader()
  const chunks: Uint8Array[] = []
  let result
  while (!(result = await reader.read()).done) {
    chunks.push(result.value)
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
      let replayData = await response.text()
      loadReplayText(replayUrl, replayData)
    } else if (contentType === 'application/x-compress' || contentType === 'application/octet-stream') {
      // This is compressed JSON.
      const decompressedData = await decompressStream(response.body)
      loadReplayText(replayUrl, decompressedData)
    } else {
      throw new Error('Unsupported content type: ' + contentType)
    }
  } catch (error) {
    Common.showModal('error', 'Error fetching replay', 'Message: ' + error)
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
    Common.showModal('error', 'Error reading file', 'Message: ' + error)
  }
}

/**
 * Expands a sequence of values.
 * Example: [[0, value1], [2, value2], ...] -> [value1, value1, value2, ...]
 */
// [[0, value1], [2, value2], ...] -> [value1, value1, value2, ...]
function expandSequence(sequence: any[], numSteps: number): any[] {
  var expanded: any[] = []
  var i = 0
  var j = 0
  var v: any = null
  for (i = 0; i < numSteps; i++) {
    if (j < sequence.length && sequence[j][0] == i) {
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
async function loadReplayText(url: string, replayData: string) {
  loadReplayJson(url, JSON.parse(replayData))
}

// Replays can be in many different formats, with stuff missing or broken.
// This function fixes the replay to be in a consistent format,
// adding missing keys, recomputing invalid values, etc.
// It also creates some internal data structures for faster access to images.
function fixReplay() {
  // Create type image mappings for faster access.
  state.replayHelper.typeImages = []
  for (const typeName of state.replay.typeNames) {
    let path = 'trace/' + typeName + '.png'
    if (ctx.hasImage(path)) {
      state.replayHelper.typeImages.push(path)
    } else {
      console.warn('Type not supported: ', path)
      state.replayHelper.typeImages.push('trace/unknown.png')
    }
  }

  // Create action image mappings for faster access.
  state.replayHelper.actionImages = []
  for (const actionName of state.replay.actionNames) {
    let path = 'trace/' + actionName + '.png'
    if (ctx.hasImage(path)) {
      state.replayHelper.actionImages.push(path)
    } else {
      console.warn('Action not supported: ', path)
      state.replayHelper.actionImages.push('trace/unknown.png')
    }
  }

  // Create item image mappings for faster access.
  state.replayHelper.itemImages = []
  for (const itemName of state.replay.itemNames) {
    let path = 'trace/' + itemName + '.png'
    if (ctx.hasImage(path)) {
      state.replayHelper.itemImages.push(path)
    } else {
      console.warn('Item not supported: ', path)
      state.replayHelper.itemImages.push('trace/unknown.png')
    }
  }
}

/** Loads a replay from a JSON object. */
async function loadReplayJson(url: string, replayData: any) {
  state.replay = new Replay()
  state.replayHelper = new ReplayHelper()

  // Assert the replay version.
  if (replayData.version != 2) {
    Common.showModal('error', 'Replay version not supported', 'Replay version ' + replayData.version + ' is not supported.')
    return
  }

  state.replay.version = replayData.version
  state.replay.numAgents = replayData.num_agents
  state.replay.maxSteps = replayData.max_steps
  state.replay.mapSize = replayData.map_size
  state.replay.fileName = replayData.file_name
  state.replay.typeNames = replayData.type_names
  state.replay.actionNames = replayData.action_names
  state.replay.itemNames = replayData.item_names

  // Go through each object and expand only known keys.
  for (const objData of replayData.objects) {
    let obj = new Object()
    let maxSteps = replayData.max_steps
    obj.id = expandSequence(objData['id'], maxSteps)
    obj.typeId = expandSequence(objData['type_id'], maxSteps)
    obj.groupId = expandSequence(objData['group_id'], maxSteps)
    obj.agentId = expandSequence(objData['agent_id'], maxSteps)
    obj.position = expandSequence(objData['position'], maxSteps)
    obj.rotation = expandSequence(objData['rotation'], maxSteps)
    obj.layer = expandSequence(objData['layer'], maxSteps)
    obj.inventory = expandSequence(objData['inventory'], maxSteps)
    obj.inventoryMax = expandSequence(objData['inventory_max'], maxSteps)
    obj.actionId = expandSequence(objData['action_id'], maxSteps)
    obj.actionParameter = expandSequence(objData['action_parameter'], maxSteps)
    obj.actionSuccess = expandSequence(objData['action_success'], maxSteps)
    obj.currentReward = expandSequence(objData['current_reward'], maxSteps)
    obj.totalReward = expandSequence(objData['total_reward'], maxSteps)
    obj.frozen = expandSequence(objData['frozen'], maxSteps)
    obj.frozenProgress = expandSequence(objData['frozen_progress'], maxSteps)
    obj.frozenTime = expandSequence(objData['frozen_time'], maxSteps)
    obj.recipeInput = expandSequence(objData['recipe_input'], maxSteps)
    obj.recipeOutput = expandSequence(objData['recipe_output'], maxSteps)
    obj.recipeMax = expandSequence(objData['recipe_max'], maxSteps)
    obj.productionProgress = expandSequence(objData['production_progress'], maxSteps)
    obj.productionTime = expandSequence(objData['production_time'], maxSteps)
    obj.cooldownProgress = expandSequence(objData['cooldown_progress'], maxSteps)
    obj.cooldownTime = expandSequence(objData['cooldown_time'], maxSteps)
    state.replay.objects.push(obj)
  }

  // Find all agents for faster access.
  state.replayHelper.agents = []
  for (let i = 0; i < state.replay.numAgents; i++) {
    state.replayHelper.agents.push(null)
    for (const obj of state.replay.objects) {
      if (getAttr(obj.agentId) == i) {
        state.replayHelper.agents[i] = obj
      }
    }
  }

  fixReplay()

  if (state.replay.fileName.length > 0) {
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

  for (const obj of replayStep.grid_objects) {
    // Grid objects are 1-indexed.
    const index = obj.id - 1

    // for (const key in obj) {
    //   const value = obj[key]
    //   // Ensure that the grid object exists.
    //   while (state.replay.objects.length <= index) {
    //     state.replay.objects.push({})
    //   }
    //   // Ensure that the key exists.
    //   if (state.replay.objects[index][key] === undefined || state.replay.objects[index][key] === null) {
    //     state.replay.objects[index][key] = []
    //     while (state.replay.objects[index][key].length <= step) {
    //       state.replay.objects[index][key].push(null)
    //     }
    //   }

    //   state.replay.objects[index][key][step] = value

    //   if (key == 'agent_id') {
    //     // Update the agent.
    //     while (state.replay.agents.length <= value) {
    //       state.replay.agents.push({})
    //     }
    //     state.replay.agents[value] = state.replay.objects[index]
    //   }
    // }
    // // Make sure that the keys that don't exist in the update are set to null too.
    // for (const key in state.replay.objects[index]) {
    //   if (obj[key] === undefined) {
    //     state.replay.objects[index][key][step] = null
    //   }
    // }
  }

  // fixReplay()

  // updateStep(step)

  // requestFrame()
}

/** Initializes the WebSocket connection. */
export function initWebSocket(wsUrl: string) {
  state.ws = new WebSocket(wsUrl)
  state.ws.onmessage = (event) => {
    const data = JSON.parse(event.data)
    console.info('Received message: ', data.type)
    if (data.type === 'replay') {
      loadReplayJson(wsUrl, data.replay)
      Common.closeModal()
      html.actionButtons.classList.remove('hidden')
    } else if (data.type === 'replay_step') {
      loadReplayStep(data.replay_step)
    } else if (data.type === 'message') {
      console.info('Received message: ', data.message)
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
    Common.showModal('error', 'WebSocket error', 'Websocket error: ' + event)
  }
}

/** Sends an action to the server. */
export function sendAction(actionName: string, actionParam: number) {
  if (state.ws === null) {
    console.error('WebSocket is not connected')
    return
  }
  const agentId = getAttr(state.selectedGridObject.agentId)
  if (agentId != null) {
    const actionId = state.replay.actionNames.indexOf(actionName)
    if (actionId == -1) {
      console.error('Action not found: ', actionName)
      return
    }
    state.ws.send(
      JSON.stringify({
        type: 'action',
        agent_id: agentId,
        action: [actionId, actionParam],
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
  // FIX ME: The inv: and agent: prefixes don't exist anymore.
  if (state.replay.typeNames.includes(key)) {
    let idx = state.replay.typeNames.indexOf(key)
    return "data/atlas/" + state.replayHelper.typeImages[idx][0]
  } else if (key.startsWith('inv:') || key.startsWith('agent:inv:')) {
    return 'data/atlas/resources/' + key.replace('inv:', '').replace('agent:', '') + '.png'
  } else {
    return 'data/ui/table/' + key.replace('agent:', '') + '.png'
  }
}
