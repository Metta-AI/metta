import { state } from './common.js'
import { find } from './htmlutils.js'
import { requestFrame } from './main.js'
import { sendAction } from './replay.js'

// Small timing window to combine two directional key presses (e.g., W+A -> Northwest).
const COMBO_WINDOW_MS = 100
let comboKeys: Set<'w' | 'a' | 's' | 'd'> = new Set()
let comboTimer: number | null = null

// Queue up a key press for a directional combo.
function pushComboKey(key: 'w' | 'a' | 's' | 'd') {
  comboKeys.add(key)
  if (comboTimer != null) window.clearTimeout(comboTimer)
  comboTimer = window.setTimeout(() => flushComboIfAny(), COMBO_WINDOW_MS)
}

// Flush the combo if any keys are pressed.
function flushComboIfAny() {
  if (comboTimer != null) {
    window.clearTimeout(comboTimer)
    comboTimer = null
  }
  if (comboKeys.size === 0) return

  let param = -1
  if (comboKeys.size >= 2) {
    const has = (k: 'w' | 'a' | 's' | 'd') => comboKeys.has(k)
    if (has('w') && has('a'))
      param = 4 // NW
    else if (has('w') && has('d'))
      param = 5 // NE
    else if (has('s') && has('d'))
      param = 7 // SE
    else if (has('s') && has('a')) param = 6 // SW
  }
  if (param === -1) {
    // Single key or opposing keys fallback: prefer last in insertion order.
    let lastKey: 'w' | 'a' | 's' | 'd' | null = null
    for (const k of comboKeys) lastKey = k
    if (lastKey === 'w') param = 0  // NORTH
    else if (lastKey === 's') param = 1  // SOUTH
    else if (lastKey === 'a') param = 2  // WEST
    else if (lastKey === 'd') param = 3  // EAST
  }
  if (param !== -1) {
    sendAction('move', param)
  }
  comboKeys.clear()
}

/** Initializes the action buttons. */
export function initActionButtons() {
  find('#action-buttons .north').addEventListener('click', () => {
    sendAction('move', 0)  // Move North = 0
  })

  find('#action-buttons .south').addEventListener('click', () => {
    sendAction('move', 1)  // Move South = 1
  })

  find('#action-buttons .west').addEventListener('click', () => {
    sendAction('move', 2)  // Move West = 2
  })

  find('#action-buttons .east').addEventListener('click', () => {
    sendAction('move', 3)  // Move East = 3
  })

  // Note: Forward/backward buttons removed - they don't make sense with unified movement
  // where each move specifies an absolute direction

  find('#action-buttons .put-recipe-items').addEventListener('click', () => {
    sendAction('put_items', 0)
  })

  find('#action-buttons .get-output').addEventListener('click', () => {
    sendAction('get_items', 0)
  })

  find('#action-buttons .noop').addEventListener('click', () => {
    sendAction('noop', 0)
  })

  find('#action-buttons .attack').addEventListener('click', () => {
    state.showAttackMode = !state.showAttackMode
  })

  find('#action-buttons .change-color').addEventListener('click', () => {
    sendAction('change_color', 0)
  })

  find('#action-buttons .swap').addEventListener('click', () => {
    sendAction('swap', 0)
  })
}

/** Processes keyboard actions. */
export function processActions(event: KeyboardEvent) {
  // Smart navigation, where pressing a key rotates the agent in the
  // direction of the key, but if the agent is already facing in that
  // direction, it moves forward.

  if (state.ws == null) {
    return
  }

  if (state.selectedGridObject != null) {
    const agent = state.selectedGridObject
    const orientation = agent.orientation.get()
    // Support WASD, arrow keys, and all numpad keys for movement/rotation.
    const key = event.key
    const code = event.code
    const supportsMove = state.replay.actionNames.includes('move')
    const supportsMove8 = state.replay.MettaGridConfig.game?.allow_diagonals ?? false

    // Movement handling - unified movement system
    if (key === 'w' || key === 'ArrowUp') {
      if (supportsMove8) {
        pushComboKey('w')
      } else {
        sendAction('move', 0) // Move North
      }
    }
    if (key === 'a' || key === 'ArrowLeft') {
      if (supportsMove8) {
        pushComboKey('a')
      } else {
        sendAction('move', 2) // Move West
      }
    }
    if (key === 's' || key === 'ArrowDown') {
      if (supportsMove8) {
        pushComboKey('s')
      } else {
        sendAction('move', 1) // Move South
      }
    }
    if (key === 'd' || key === 'ArrowRight') {
      if (supportsMove8) {
        pushComboKey('d')
      } else {
        sendAction('move', 3) // Move East
      }
    }

    // Numpad movement (immediate, no combo buffering)
    if (code === 'Numpad8') {
      sendAction('move', 0) // Move North
    }
    if (code === 'Numpad4') {
      sendAction('move', 2) // Move West
    }
    if (code === 'Numpad2') {
      sendAction('move', 1) // Move South
    }
    if (code === 'Numpad6') {
      sendAction('move', 3) // Move East
    }
    // Forward/backward relative to current orientation
    if (event.key === 'f') {
      // Move in current facing direction
      sendAction('move', orientation)
    }
    if (event.key === 'r') {
      // Move backward (opposite of current facing)
      const opposites: { [key: number]: number } = { 0: 1, 1: 0, 2: 3, 3: 2 }
      sendAction('move', opposites[orientation] || 0)
    }
    if (event.key === 'q') {
      // Put recipe items.
      sendAction('put_items', 0)
    }
    if (event.key === 'b') {
      // Place box.
      sendAction('place_box', 0)
    }
    if (event.key === 'e') {
      // Get the output.
      sendAction('get_items', 0)
    }
    // Diagonal numpad
    if (event.code === 'Numpad7') {
      if (supportsMove8) {
        sendAction('move', 4) // Northwest
      } else {
        // For 4-directional movement, move North then West
        sendAction('move', 0) // North
        setTimeout(() => sendAction('move', 2), 50) // West after small delay
      }
    }
    if (event.code === 'Numpad9') {
      if (supportsMove8) {
        sendAction('move', 5) // Northeast
      } else {
        // For 4-directional movement, move North then East
        sendAction('move', 0) // North
        setTimeout(() => sendAction('move', 3), 50) // East after small delay
      }
    }
    if (event.code === 'Numpad1') {
      if (supportsMove8) {
        sendAction('move', 6) // Southwest
      } else {
        // For 4-directional movement, move South then West
        sendAction('move', 1) // South
        setTimeout(() => sendAction('move', 2), 50) // West after small delay
      }
    }
    if (event.code === 'Numpad3') {
      if (supportsMove8) {
        sendAction('move', 7) // Southeast
      } else {
        // For 4-directional movement, move South then East
        sendAction('move', 1) // South
        setTimeout(() => sendAction('move', 3), 50) // East after small delay
      }
    }

    if (event.key === 'x' || event.code === 'Numpad5') {
      // No-op / wait.
      sendAction('noop', 0)
    }
    if (event.key >= '1' && event.key <= '9') {
      // Keys 1-9 are the attack matrix.
      sendAction('attack', Number.parseInt(event.key) - 1)
    }
    if (event.key === 'Z') {
      // Show attack mode menu (a bunch of little circles that you can click on).
      state.showAttackMode = !state.showAttackMode
    }
    if (event.key === 'c') {
      // Change color.
      sendAction('change_color', 0)
    }
    if (event.key === 'g') {
      // Swap.
      sendAction('swap', 0)
    }
  }
}

// Gamepad support

// You can test your gamepad here: https://hardwaretester.com/gamepad

// Remember the previous pressed state of each button so we can detect
// rising edges (button just pressed) and avoid sending the same command
// repeatedly every animation frame.

const prevButtonPressed: boolean[] = []

/**
 * Processes the first connected gamepad (if any) and converts gamepad input
 * into the same logical keyboard events that `processActions` already
 * handles. This allows us to reuse the existing keyboard-driven action logic
 * without duplicating it.
 */
export function processGamepad() {
  const gamepads = navigator.getGamepads ? navigator.getGamepads() : []
  if (!gamepads) {
    return false
  }

  const gp = gamepads[0]
  if (!gp) {
    // gamepads do not show up until you press a button
    return false
  }

  let inputDetected = false

  // Helper to dispatch a synthetic keyboard event so that we can reuse the
  // existing `processActions` logic.
  const dispatchKey = (key: string) => {
    // Construct a minimal KeyboardEvent carrying the `key` information.
    // The event is not fired on the DOM; instead we pass it directly to
    // `processActions`.
    const evt = new KeyboardEvent('gamepad', { key })
    processActions(evt)
    inputDetected = true
  }

  // Map of gamepad button indices to associated keys.
  const buttonKeyMap: Record<number, string> = {
    0: 'q', // A – get items
    1: 'e', // B – put items
    2: '', // X – unmapped
    3: '', // Y – unmapped
    6: 'g', // LT – swap
    7: 'c', // RT – change color
  }

  // Process face/shoulder buttons.
  gp.buttons.forEach((btn, index) => {
    const pressed = typeof btn === 'object' ? btn.pressed : btn === 1.0
    const prev = prevButtonPressed[index] || false
    // Log button state for debugging
    if (pressed && !prev) {
      const key = buttonKeyMap[index]
      if (key) {
        dispatchKey(key)
      }
    }
    prevButtonPressed[index] = pressed
  })

  // Helper function to convert directional input into keys. We trigger the
  // key event whenever the direction crosses a threshold from the neutral
  // position. We treat both the left stick (axes 0 and 1) and the d-pad
  // buttons (12-15).

  const axisThreshold = 0.5

  const directionActives: Record<string, boolean> = {
    w: false, // up
    s: false, // down
    a: false, // left
    d: false, // right
  }

  // Set from D-pad buttons.
  if (gp.buttons[12]?.pressed) {
    directionActives.w = true
  }
  if (gp.buttons[13]?.pressed) {
    directionActives.s = true
  }
  if (gp.buttons[14]?.pressed) {
    directionActives.a = true
  }
  if (gp.buttons[15]?.pressed) {
    directionActives.d = true
  }

  // Set from left stick axes.
  if (gp.axes.length >= 2) {
    const x = gp.axes[0]
    const y = gp.axes[1]
    if (y < -axisThreshold) {
      directionActives.w = true
    }
    if (y > axisThreshold) {
      directionActives.s = true
    }
    if (x < -axisThreshold) {
      directionActives.a = true
    }
    if (x > axisThreshold) {
      directionActives.d = true
    }
  }

  // Trigger key events for newly active directions.
  Object.entries(directionActives).forEach(([key, active]) => {
    const idx = 100 + key.charCodeAt(0) // offset to avoid collision
    const prev = prevButtonPressed[idx] || false
    if (active && !prev) {
      dispatchKey(key)
    }
    prevButtonPressed[idx] = active
  })

  return inputDetected
}

/**
 * Starts the gamepad polling loop that runs independently of the main render loop.
 * This ensures gamepad input is detected even when the game is idle.
 */
export function startGamepadPolling() {
  const pollGamepad = () => {
    try {
      const inputDetected = processGamepad()
      if (inputDetected) {
        requestFrame()
      }
    } catch (e) {
      console.error('Error processing gamepad input:', e)
    }

    // Continue polling
    requestAnimationFrame(pollGamepad)
  }

  // Start the polling loop
  requestAnimationFrame(pollGamepad)
}
