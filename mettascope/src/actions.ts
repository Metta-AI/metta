import { state } from './common.js'
import { find } from './htmlutils.js'
import { requestFrame } from './main.js'
import { getAttr, sendAction } from './replay.js'

/** Initializes the action buttons. */
export function initActionButtons() {
  find('#action-buttons .north').addEventListener('click', () => {
    sendAction('rotate', 0)
  })

  find('#action-buttons .west').addEventListener('click', () => {
    sendAction('rotate', 2)
  })

  find('#action-buttons .south').addEventListener('click', () => {
    sendAction('rotate', 1)
  })

  find('#action-buttons .east').addEventListener('click', () => {
    sendAction('rotate', 3)
  })

  find('#action-buttons .forward').addEventListener('click', () => {
    sendAction('move', 0)
  })

  find('#action-buttons .backward').addEventListener('click', () => {
    sendAction('move', 1)
  })

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
    const orientation = getAttr(agent, 'agent:orientation')
    // Support WASD, arrow keys, and all numpad keys for movement/rotation.
    const key = event.key
    const code = event.code

    if (key === 'w' || key === 'ArrowUp' || code === 'Numpad8') {
      if (orientation !== 0) {
        // Rotate up.
        sendAction('rotate', 0)
      } else {
        // Move forward (up).
        sendAction('move', 0)
      }
    }
    if (key === 'a' || key === 'ArrowLeft' || code === 'Numpad4') {
      if (orientation !== 2) {
        // Rotate left.
        sendAction('rotate', 2)
      } else {
        // Move forward (left).
        sendAction('move', 0)
      }
    }
    if (key === 's' || key === 'ArrowDown' || code === 'Numpad2') {
      if (orientation !== 1) {
        // Rotate down.
        sendAction('rotate', 1)
      } else {
        // Move forward (down).
        sendAction('move', 0)
      }
    }
    if (key === 'd' || key === 'ArrowRight' || code === 'Numpad6') {
      if (orientation !== 3) {
        // Rotate right.
        sendAction('rotate', 3)
      } else {
        // Move forward (right).
        sendAction('move', 0)
      }
    }
    if (event.key === 'f') {
      // Just move forward.
      sendAction('move', 0)
    }
    if (event.key === 'r') {
      // Just move backward/reverse.
      sendAction('move', 1)
    }
    if (event.key === 'q') {
      // Put recipe items.
      sendAction('put_items', 0)
    }
    if (event.key === 'e') {
      // Get the output.
      sendAction('get_items', 0)
    }
    // Diagonal movement with numpad (4-action approach for true diagonal in one press)
    if (event.code === 'Numpad7') {
      // Up-Left
      sendAction('rotate', 0) // Rotate up
      sendAction('move', 0) // Move up
      sendAction('rotate', 2) // Rotate left
      sendAction('move', 0) // Move left
    }
    if (event.code === 'Numpad9') {
      // Up-Right
      sendAction('rotate', 0) // Rotate up
      sendAction('move', 0) // Move up
      sendAction('rotate', 3) // Rotate right
      sendAction('move', 0) // Move right
    }
    if (event.code === 'Numpad1') {
      // Down-Left
      sendAction('rotate', 1) // Rotate down
      sendAction('move', 0) // Move down
      sendAction('rotate', 2) // Rotate left
      sendAction('move', 0) // Move left
    }
    if (event.code === 'Numpad3') {
      // Down-Right
      sendAction('rotate', 1) // Rotate down
      sendAction('move', 0) // Move down
      sendAction('rotate', 3) // Rotate right
      sendAction('move', 0) // Move right
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

// ---------------------------------------------------------------------------
// Gamepad support
// ---------------------------------------------------------------------------

// You can test your gamepad here: https://hardwaretester.com/gamepad

// Remember the previous pressed state of each button so we can detect
// rising edges (button just pressed) and avoid sending the same command
// repeatedly every animation frame.

const prevButtonPressed: Array<boolean> = []

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
