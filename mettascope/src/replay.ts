import * as Common from './common.js'
import { ui, state, html, ctx } from './common.js'
import { focusFullMap } from './worldmap.js'
import { onResize, updateStep, requestFrame } from './main.js'
import { updateAgentTable } from './agentpanel.js'


// type Sequence<T> = [number, T][] | [number, T] | null

export class Sequence<T> {
  private value: T | null = null
  private sequence: T[] | null = null
  constructor(defaultValue: T) {
    this.value = defaultValue
  }

  expand(data: any, numSteps: number) {
    if (data == null || data == undefined) {
      // Use the default value.
      return
    } else if (data instanceof Array) {
      // For coordinates, we need to expand the sequence.
      if (Array.isArray(this.value) && data.length > 0 && !Array.isArray(data[1])) {
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

  isSequence(): boolean {
    return this.sequence != null
  }

  isValue(): boolean {
    return this.sequence == null
  }
}

// Entity and replay conform version 2 of the replay_spec.md.
export class Entity {
  // Common keys.
  id: Sequence<number> = new Sequence(0)
  typeId: Sequence<number> = new Sequence(0)
  groupId: Sequence<number> = new Sequence(0)
  agentId: Sequence<number> = new Sequence(0)
  position: Sequence<[number, number]> = new Sequence([0, 0])
  rotation: Sequence<number> = new Sequence(0)
  layer: Sequence<number> = new Sequence(0)
  inventory: Sequence<number[]> = new Sequence([])
  inventoryMax: Sequence<number> = new Sequence(0)
  color: Sequence<number> = new Sequence(0)

  // Agent specific keys.
  actionId: Sequence<number> = new Sequence(0)
  actionParameter: Sequence<number> = new Sequence(0)
  actionSuccess: Sequence<boolean> = new Sequence(false)
  currentReward: Sequence<number> = new Sequence(0)
  totalReward: Sequence<number> = new Sequence(0)
  frozen: Sequence<boolean> = new Sequence(false)
  frozenProgress: Sequence<number> = new Sequence(0)
  frozenTime: Sequence<number> = new Sequence(0)

  // Building specific keys.
  recipeInput: Sequence<number> = new Sequence(0)
  recipeOutput: Sequence<number> = new Sequence(0)
  recipeMax: Sequence<number> = new Sequence(0)
  productionProgress: Sequence<number> = new Sequence(0)
  productionTime: Sequence<number> = new Sequence(0)
  cooldownProgress: Sequence<number> = new Sequence(0)
  cooldownTime: Sequence<number> = new Sequence(0)

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
}

// Replay helper has lookups that speed up rendering.
export class ReplayHelper {
  actionImages: string[] = []
  typeImages: string[] = []
  itemImages: string[] = []
  agents: (Entity | null)[] = []
}


/** Gets an attribute from a grid Entity, respecting the current step. */
// export function getAttr<T>(sequence: Sequence<T>, atStep = -1): T {
//   if (atStep == -1) {
//     // When the step is not passed in, use the global step.
//     atStep = state.step
//   }
//   if (sequence instanceof Array) {
//     return sequence[atStep] as T
//   } else {
//     return sequence as T
//   }
// }


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

// /**
//  * Expands a sequence of values.
//  * Example: [[0, value1], [2, value2], ...] -> [value1, value1, value2, ...]
//  */
// // [[0, value1], [2, value2], ...] -> [value1, value1, value2, ...]
// function expandSequence(sequence: any, numSteps: number): any {
//   if (sequence == null) {
//     // Null is a valid sequence.
//     return null
//   } else if (sequence instanceof Array) {
//     // A sequence of pairs is expanded to a sequence of values.
//     var expanded: any[] = []
//     var i = 0
//     var j = 0
//     var v: any = null
//     for (i = 0; i < numSteps; i++) {
//       if (j < sequence.length && sequence[j][0] == i) {
//         v = sequence[j][1]
//         j++
//       }
//       expanded.push(v)
//     }
//     return expanded
//   } else {
//     // A single value is a valid sequence.
//     return sequence
//   }
// }

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
  if (!state.replay || !state.replayHelper) return

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

/** Loads a replay from a JSON Entity. */
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

  // Go through each Entity and expand only known keys.
  for (const objData of replayData.objects) {
    let obj = new Entity()
    let maxSteps = replayData.max_steps
    obj.id.expand(objData['id'], maxSteps)
    obj.typeId.expand(objData['type_id'], maxSteps)
    obj.groupId.expand(objData['group_id'], maxSteps)
    obj.agentId.expand(objData['agent_id'], maxSteps)
    obj.position.expand(objData['position'], maxSteps)
    obj.rotation.expand(objData['rotation'], maxSteps)
    obj.layer.expand(objData['layer'], maxSteps)
    obj.inventory.expand(objData['inventory'], maxSteps)
    obj.inventoryMax.expand(objData['inventory_max'], maxSteps)
    obj.actionId.expand(objData['action_id'], maxSteps)
    obj.actionParameter.expand(objData['action_parameter'], maxSteps)
    obj.actionSuccess.expand(objData['action_success'], maxSteps)
    obj.currentReward.expand(objData['current_reward'], maxSteps)
    obj.totalReward.expand(objData['total_reward'], maxSteps)
    obj.frozen.expand(objData['frozen'], maxSteps)
    obj.frozenProgress.expand(objData['frozen_progress'], maxSteps)
    obj.frozenTime.expand(objData['frozen_time'], maxSteps)
    obj.recipeInput.expand(objData['recipe_input'], maxSteps)
    obj.recipeOutput.expand(objData['recipe_output'], maxSteps)
    obj.recipeMax.expand(objData['recipe_max'], maxSteps)
    obj.productionProgress.expand(objData['production_progress'], maxSteps)
    obj.productionTime.expand(objData['production_time'], maxSteps)
    obj.cooldownProgress.expand(objData['cooldown_progress'], maxSteps)
    obj.cooldownTime.expand(objData['cooldown_time'], maxSteps)
    state.replay.objects.push(obj)
  }

  // Find all agents for faster access.
  state.replayHelper.agents = []
  for (let i = 0; i < state.replay.numAgents; i++) {
    state.replayHelper.agents.push(null)
    for (const obj of state.replay.objects) {
      if (obj.agentId.get() == i) {
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
  if (!state.replay) return

  // This gets us a simple replay step that we can overwrite.

  // Update the grid objects.
  const step = replayStep.step

  state.replay.maxSteps = Math.max(state.replay.maxSteps, step + 1)

  for (const obj of replayStep.grid_objects) {
    // Grid objects are 1-indexed.
    const index = obj.id - 1

    // for (const key in obj) {
    //   const value = obj[key]
    //   // Ensure that the grid Entity exists.
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
  if (state.ws === null || !state.replay || !state.selectedGridObject) {
    console.error('WebSocket is not connected or no replay/selected object')
    return
  }
  const agentId = state.selectedGridObject.agentId.get()
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
    console.error('No selected grid Entity')
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
  if (!state.replay || !state.replayHelper) return ''

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
