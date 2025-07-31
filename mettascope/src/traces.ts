import * as Common from './common.js'
import { ctx, setFollowSelection, state, ui } from './common.js'
import { parseHtmlColor } from './htmlutils.js'
import { updateSelection, updateStep } from './main.js'
import type { PanelInfo } from './panels.js'
import { Vec2f } from './vector_math.js'

// Cache tracking.
const lastCachedState = {
  step: -1,
  selection: null as any,
  zoomLevel: -1,
  panPos: null as Vec2f | null,
  width: -1,
  height: -1,
}

/** Invalidate the trace cache to force regeneration on next call to drawTrace. */
export function invalidateTrace() {
  lastCachedState.step = -1
}

/** Draws the trace panel. */
export function drawTrace(panel: PanelInfo) {
  if (state.replay === null || ctx === null || ctx.ready === false) {
    return
  }

  let shouldRegenerate = false

  // The trace mesh should be cached and only updated when needed.
  ctx.setCacheable(true)

  // Handle mouse events for the trace panel.
  if (ui.mouseTargets.includes('#trace-panel')) {
    if (ui.mouseDoubleClick) {
      // Toggle followSelection on double-click.
      console.info('Trace double click - following selection')
      setFollowSelection(true)
      panel.zoomLevel = Common.DEFAULT_TRACE_ZOOM_LEVEL
      ui.mapPanel.zoomLevel = Common.DEFAULT_ZOOM_LEVEL
    } else if (ui.mouseClick) {
      // A trace click is likely a drag/pan.
      console.info('Trace click - clearing trace follow selection')
      setFollowSelection(false)
    } else if (ui.mouseUp && ui.mouseDownPos.sub(ui.mousePos).length() < 10) {
      // Check if we are clicking on an action/step.
      console.info('Trace up without dragging - selecting trace object')
      const localMousePos = panel.transformOuter(ui.mousePos)
      if (localMousePos != null) {
        const mapX = localMousePos.x()
        const selectedStep = Math.floor(mapX / Common.TRACE_WIDTH)
        const agentId = Math.floor(localMousePos.y() / Common.TRACE_HEIGHT)
        if (
          mapX > 0 &&
          mapX < state.replay.maxSteps * Common.TRACE_WIDTH &&
          localMousePos.y() > 0 &&
          localMousePos.y() < state.replay.numAgents * Common.TRACE_HEIGHT &&
          selectedStep >= 0 &&
          selectedStep < state.replay.maxSteps &&
          agentId >= 0 &&
          agentId < state.replay.numAgents
        ) {
          updateSelection(state.replay.agents[agentId])
          if (state.selectedGridObject != null) {
            console.info('Selected an agent on a trace:', state.selectedGridObject)
            const location = state.selectedGridObject.location.get()
            ui.mapPanel.focusPos(
              location[0] * Common.TILE_SIZE,
              location[1] * Common.TILE_SIZE,
              Common.DEFAULT_ZOOM_LEVEL
            )
            // Update the step to the clicked step.
            updateStep(selectedStep)
          }
        }
      }
    }
  }

  // If we're following a selection, center the trace panel on it.
  if (state.followSelection && state.selectedGridObject !== null) {
    const x = state.step * Common.TRACE_WIDTH + Common.TRACE_WIDTH / 2
    const y = state.selectedGridObject.agentId * Common.TRACE_HEIGHT + Common.TRACE_HEIGHT / 2
    panel.panPos = new Vec2f(-x, -y)
    shouldRegenerate = true
  }

  // if any state has changed, we need to regenerate the trace mesh.
  if (
    state.step !== lastCachedState.step ||
    state.selectedGridObject !== lastCachedState.selection ||
    panel.zoomLevel !== lastCachedState.zoomLevel ||
    panel.width !== lastCachedState.width ||
    panel.height !== lastCachedState.height ||
    lastCachedState.panPos === null ||
    panel.panPos.x() !== lastCachedState.panPos.x() ||
    panel.panPos.y() !== lastCachedState.panPos.y()
  ) {
    shouldRegenerate = true
    lastCachedState.step = state.step
    lastCachedState.selection = state.selectedGridObject
    lastCachedState.zoomLevel = panel.zoomLevel
    lastCachedState.width = panel.width
    lastCachedState.height = panel.height
    lastCachedState.panPos = new Vec2f(panel.panPos.x(), panel.panPos.y())
  }

  // Clear and regenerate ALL content only when needed
  if (!shouldRegenerate) {
    return
  }

  ctx.save()
  ctx.clearMesh()

  const rect = panel.rectInner()
  ctx.setScissorRect(rect.x, rect.y, rect.width, rect.height)

  const fullSize = new Vec2f(state.replay.maxSteps * Common.TRACE_WIDTH, state.replay.numAgents * Common.TRACE_HEIGHT)

  // Draw the background.
  ctx.drawSolidRect(
    rect.x,
    rect.y,
    rect.width,
    rect.height,
    //[0.08, 0.08, 0.08, 1.0] // Dark background
    parseHtmlColor('#141B23')
  )

  ctx.translate(rect.x + rect.width / 2, rect.y + rect.height / 2)
  ctx.scale(panel.zoomLevel, panel.zoomLevel)
  ctx.translate(panel.panPos.x(), panel.panPos.y())

  // Draw a rectangle around the selected agent.
  if (state.selectedGridObject !== null && state.selectedGridObject.agentId !== undefined) {
    const agentId = state.selectedGridObject.agentId

    // Draw the selection rectangle.
    ctx.drawSolidRect(0, agentId * Common.TRACE_HEIGHT, fullSize.x(), Common.TRACE_HEIGHT, [0.3, 0.3, 0.3, 1])
  }

  // Draw a line for the current step that goes through all the traces.
  ctx.drawSolidRect(
    state.step * Common.TRACE_WIDTH,
    0,
    Common.TRACE_WIDTH,
    fullSize.y(),
    [0.5, 0.5, 0.5, 0.5] // White with 50% opacity.
  )

  // Draw the agent traces.
  for (let i = 0; i < state.replay.numAgents; i++) {
    const agent = state.replay.agents[i]
    for (let j = 0; j < state.replay.maxSteps; j++) {
      const actionId = agent.actionId.get(j)
      const actionParam = agent.actionParameter.get(j)
      const actionSuccess = agent.actionSuccess.get(j)

      if (agent.isFrozen.get(j)) {
        // Draw the frozen state.
        ctx.drawSprite(
          'trace/frozen.png',
          j * Common.TRACE_WIDTH + Common.TRACE_WIDTH / 2,
          i * Common.TRACE_HEIGHT + Common.TRACE_HEIGHT / 2
        )
      } else if (actionSuccess && actionId >= 0 && actionId < state.replay.actionImages.length) {
        // Draw the action.
        ctx.drawSprite(
          state.replay.actionImages[actionId],
          j * Common.TRACE_WIDTH + Common.TRACE_WIDTH / 2,
          i * Common.TRACE_HEIGHT + Common.TRACE_HEIGHT / 2
        )
      } else if (actionId != null && actionId >= 0 && actionId < state.replay.actionImages.length) {
        // Draw the invalid action.
        ctx.drawSprite(
          'trace/invalid.png',
          j * Common.TRACE_WIDTH + Common.TRACE_WIDTH / 2,
          i * Common.TRACE_HEIGHT + Common.TRACE_HEIGHT / 2
        )
      }

      const reward = agent.currentReward.get(j)
      // If there is a reward, draw a coin.
      if (reward > 0) {
        ctx.drawSprite(
          'resources/reward.png',
          j * Common.TRACE_WIDTH + Common.TRACE_WIDTH / 2,
          i * Common.TRACE_HEIGHT + Common.TRACE_HEIGHT - 32,
          [1.0, 1.0, 1.0, 1.0],
          1 / 4
        )
      }

      // Draw resource gain/loss.
      // if (state.showResources && j > 0) {
      //   const inventory = agent.inventory.get(j)
      //   const prevInventory = agent.inventory.get(j - 1)
      //   let y = 0
      //   const step = 32
      //   for (const inventoryPair of inventory) {
      //     const inventoryId = inventoryPair[0]
      //     const inventoryAmount = inventoryPair[1]
      //     let diff = inventoryAmount
      //     // If the inventory has changed, draw the sprite.
      //     for (const prevInventoryPair of prevInventory) {
      //       if (prevInventoryPair[0] === inventoryPair[0]) {
      //         diff = inventoryAmount - prevInventoryPair[1]
      //       }
      //     }
      //     for (let k = 0; k < diff; k++) {
      //       const inventoryName = state.replay.itemNames[inventoryId]
      //       const inventoryImage = `resources/${inventoryName}.png`
      //       ctx.drawSprite(
      //         inventoryImage,
      //         j * Common.TRACE_WIDTH + Common.TRACE_WIDTH / 2,
      //         i * Common.TRACE_HEIGHT + y,
      //         [1, 1, 1, 1],
      //         1 / 4
      //       )
      //       y += step
      //     }
      //   }
      // }
    }
  }

  ctx.restore()
}
