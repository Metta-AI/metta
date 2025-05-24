import { state } from './common.js';
import { getAttr, sendAction } from './replay.js';
import { find } from './htmlutils.js';


export function initActionButtons() {
  find("#action-buttons .north").addEventListener('click', () => {
    sendAction("rotate", 0)
  })

  find("#action-buttons .west").addEventListener('click', () => {
    sendAction("rotate", 2)
  })

  find("#action-buttons .south").addEventListener('click', () => {
    sendAction("rotate", 1)
  })

  find("#action-buttons .east").addEventListener('click', () => {
    sendAction("rotate", 3)
  })

  find("#action-buttons .forward").addEventListener('click', () => {
    sendAction("move", 0)
  })

  find("#action-buttons .backward").addEventListener('click', () => {
    sendAction("move", 1)
  })

  find("#action-buttons .put-recipe-items").addEventListener('click', () => {
    sendAction("put_recipe_items", 0)
  })

  find("#action-buttons .get-output").addEventListener('click', () => {
    sendAction("get_output", 0)
  })

  find("#action-buttons .noop").addEventListener('click', () => {
    sendAction("noop", 0)
  })

  find("#action-buttons .attack").addEventListener('click', () => {
    state.showAttackMode = true;
  })

  find("#action-buttons .attack-nearest").addEventListener('click', () => {
    sendAction("attack_nearest", 0)
  })

  find("#action-buttons .change-color").addEventListener('click', () => {
    sendAction("change_color", 0)
  })

  find("#action-buttons .swap").addEventListener('click', () => {
    sendAction("swap", 0)
  })
}

export function processActions(event: KeyboardEvent) {
  // Smart navigation, where pressing key rotations the agent in the
  // direction of the key, but if the agent is already facing in that
  // direction, it moves forward.
  if (state.ws != null && state.selectedGridObject != null) {
    const agent = state.selectedGridObject;
    const orientation = getAttr(agent, "agent:orientation");
    if (event.key == "w") {
      if (orientation != 0) {
        // Rotate up.
        sendAction("rotate", 0)
      } else {
        // Move forward (up).
        sendAction("move", 0)
      }
    }
    if (event.key == "a") {
      if (orientation != 2) {
        // Rotate left.
        sendAction("rotate", 2)
      } else {
        // Move forward (left).
        sendAction("move", 0)
      }
    }
    if (event.key == "s") {
      if (orientation != 1) {
        // Rotate down.
        sendAction("rotate", 1)
      } else {
        // Move forward (down).
        sendAction("move", 0)
      }
    }
    if (event.key == "d") {
      if (orientation != 3) {
        // Rotate right.
        sendAction("rotate", 3)
      } else {
        // Move forward (right).
        sendAction("move", 0)
      }
    }
    if (event.key == "f") {
      // Just move forward.
      sendAction("move", 0)
    }
    if (event.key == "r") {
      // Just move backwards/reverse.
      sendAction("move", 1)
    }
    if (event.key == "q") {
      // Put recipe items.
      sendAction("put_recipe_items", 0)
    }
    if (event.key == "e") {
      // Get output.
      sendAction("get_output", 0)
    }
    if (event.key == "x") {
      // Noop.
      sendAction("noop", 0)
    }
    if (event.key >= "1" && event.key <= "9") {
      // Keys 1-9 is the attack matrix.
      sendAction("attack", parseInt(event.key))
    }
    if (event.key == "z") {
      // Attack nearest.
      sendAction("attack_nearest", 0)
    }
    if (event.key == "c") {
      // Change color.
      sendAction("change_color", 0)
    }
    if (event.key == "g") {
      // Swap.
      sendAction("swap", 0)
    }
  }
}
