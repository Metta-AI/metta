/**
 * This file defines the object menu.
 * It is shown when the memory button on the info panel is clicked.
 * It allows the user to set the memory to 0, 1, or random.
 * It also allows the user to copy and paste the memory.
 * The cool thing about copy-and-paste is that it uses the clipboard,
 * so it works across tabs and can be saved to a file or sent across devices.
 */

import { state } from './common.js'
import { find, findAttr, onEvent, showMenu } from './htmlutils.js'

const objectMenu = find('#object-menu')

export function initObjectMenu() {
  objectMenu.classList.add('hidden')
}

/** Shows the object menu when the memory button on the info panel is clicked. */
onEvent('click', '.hover-panel .memory', (target: HTMLElement, _e: Event) => {
  const agentId = findAttr(target, 'data-agent-id')
  objectMenu.setAttribute('data-agent-id', agentId)
  showMenu(target, objectMenu)
})

/** In the object menu, sets the memory to 0. */
onEvent('click', '#object-menu .set-memory-to-0', (target: HTMLElement, _e: Event) => {
  if (state.ws == null) {
    return
  }
  const agentId = Number.parseInt(findAttr(target, 'data-agent-id'))
  state.ws.send(
    JSON.stringify({
      type: 'clear_memory',
      what: '0',
      agent_id: agentId,
    })
  )
})

/** In the object menu, sets the memory to 1. */
onEvent('click', '#object-menu .set-memory-to-1', (target: HTMLElement, _e: Event) => {
  if (state.ws == null) {
    return
  }
  const agentId = Number.parseInt(findAttr(target, 'data-agent-id'))
  state.ws.send(
    JSON.stringify({
      type: 'clear_memory',
      what: '1',
      agent_id: agentId,
    })
  )
})

/** In the object menu, sets the memory to random. */
onEvent('click', '#object-menu .set-memory-to-random', (target: HTMLElement, _e: Event) => {
  if (state.ws == null) {
    return
  }
  const agentId = Number.parseInt(findAttr(target, 'data-agent-id'))
  state.ws.send(
    JSON.stringify({
      type: 'clear_memory',
      what: 'random',
      agent_id: agentId,
    })
  )
})

/** In the object menu, copies the memory. */
onEvent('click', '#object-menu .copy-memory', (target: HTMLElement, _e: Event) => {
  if (state.ws == null) {
    return
  }
  const agentId = Number.parseInt(findAttr(target, 'data-agent-id'))
  // Request memory from the server.
  state.ws.send(
    JSON.stringify({
      type: 'copy_memory',
      agent_id: agentId,
    })
  )
})

/** In the object menu, pastes the memory. */
onEvent('click', '#object-menu .paste-memory', async (target: HTMLElement, _e: Event) => {
  if (state.ws == null) {
    return
  }
  const agentId = Number.parseInt(findAttr(target, 'data-agent-id'))
  try {
    const clipboardText = await navigator.clipboard.readText()
    const memory = JSON.parse(clipboardText) as [number[], number[]]
    state.ws.send(
      JSON.stringify({
        type: 'paste_memory',
        agent_id: agentId,
        memory: memory,
      })
    )
  } catch (err) {
    console.error('Failed to paste from clipboard:', err)
  }
})
