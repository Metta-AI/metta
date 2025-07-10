import { Vec2f } from './vector_math.js'
import * as Common from './common.js'
import { ui, state, ctx, setFollowSelection } from './common.js'
import { PanelInfo } from './panels.js'
import { updateStep, updateSelection } from './main.js'
import { parseHtmlColor } from './htmlutils.js'

/** Draws the trace panel. */
export function drawTraces(panel: PanelInfo) {
  if (!state.replay || !state.replayHelper) return

  // Handle mouse events for the trace panel.
  if (ui.mouseTargets.includes('#trace-panel')) {
    if (ui.dragging == '') {
      // Check if we are clicking on a trace Entity.
      const localMousePos = panel.transformOuter(ui.mousePos)
      if (localMousePos != null) {
        const mapX = localMousePos.x()
        const mapY = localMousePos.y()
        if (
          mapX >= 0 &&
          mapX < state.replay.maxSteps * Common.TRACE_WIDTH &&
          mapY >= 0 &&
          localMousePos.y() < state.replay.numAgents * Common.TRACE_HEIGHT &&
          ui.mouseUp
        ) {
          const selectedStep = Math.floor(mapX / Common.TRACE_WIDTH)
          const agentId = Math.floor(mapY / Common.TRACE_HEIGHT)
          if (
            selectedStep >= 0 &&
            selectedStep < state.replay.maxSteps &&
            agentId >= 0 &&
            agentId < state.replay.numAgents
          ) {
            updateSelection(state.replayHelper.agents[agentId])
            updateStep(selectedStep)
            const position = state.selectedGridObject ? state.selectedGridObject.position.get() : null
            if (position) {
              ui.mapPanel.focusPos(
                position[0] * Common.TILE_SIZE,
                position[1] * Common.TILE_SIZE,
                Common.DEFAULT_ZOOM_LEVEL
              )
            }
          }
        }
      }
    }
  }

  // If we're following a selection, center the trace panel on it.
  if (state.followSelection && state.selectedGridObject !== null) {
    const x = state.step * Common.TRACE_WIDTH + Common.TRACE_WIDTH / 2
    const agentId = state.selectedGridObject.agentId.get()
    if (agentId !== null) {
      const y = (agentId as number) * Common.TRACE_HEIGHT + Common.TRACE_HEIGHT / 2
      panel.panPos = new Vec2f(-x, -y)
    }
  }

  ctx.save()
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
  if (state.selectedGridObject !== null && state.selectedGridObject.agentId !== null) {
    const agentId = state.selectedGridObject.agentId.get()

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
    const agent = state.replayHelper.agents[i]
    if (!agent) continue

    for (let j = 0; j < state.replay.maxSteps; j++) {
      const action = agent.actionId.get(j)
      const action_success = agent.actionSuccess.get(j)

      if (agent.frozen.get(j)) {
        // Draw the frozen state.
        ctx.drawSprite(
          'trace/frozen.png',
          j * Common.TRACE_WIDTH + Common.TRACE_WIDTH / 2,
          i * Common.TRACE_HEIGHT + Common.TRACE_HEIGHT / 2
        )
      } else if (action_success && action != null && (action as number) >= 0 && (action as number) < state.replayHelper.actionImages.length) {
        ctx.drawSprite(
          'trace/success.png',
          j * Common.TRACE_WIDTH,
          i * Common.TRACE_HEIGHT,
          parseHtmlColor('#B8DBB8')
        )
        ctx.drawSprite(
          state.replayHelper.actionImages[action as number],
          j * Common.TRACE_WIDTH + Common.TRACE_WIDTH / 2,
          i * Common.TRACE_HEIGHT + Common.TRACE_HEIGHT / 2
        )
      } else if (action != null && (action as number) >= 0 && (action as number) < state.replayHelper.actionImages.length) {
        // Draw the invalid action.
        ctx.drawSprite(
          'trace/invalid.png',
          j * Common.TRACE_WIDTH + Common.TRACE_WIDTH / 2,
          i * Common.TRACE_HEIGHT + Common.TRACE_HEIGHT / 2
        )
      }

      const reward = agent.currentReward.get(j)
      // If there is a reward, draw a coin.
      if ((reward as number) > 0) {
        ctx.drawSprite(
          'resources/reward.png',
          j * Common.TRACE_WIDTH + Common.TRACE_WIDTH / 2,
          i * Common.TRACE_HEIGHT + Common.TRACE_HEIGHT - 32,
          [1.0, 1.0, 1.0, 1.0],
          1 / 4
        )
      }

      // FIX ME: resource_inventory doesn't exist on Replay type
      // Draw the resources gained or lost.
      /*
      let number = 0
      for (const [key, [image, color]] of state.replay.resource_inventory) {
        number += Math.abs(getAttr(agent, key, j + 1) - getAttr(agent, key, j))
      }
      if (number > 0) {
        ctx.drawNumber(j * Common.TRACE_WIDTH + 36, i * Common.TRACE_HEIGHT + 36, number, parseHtmlColor('#FFF'))
      }
      */

      // Also commented out this resource tracking section:
      /*
      for (const [key, [image, color]] of state.replay.resource_inventory) {
        const prevResources = getAttr(agent, key, j - 1)
        const nextResources = getAttr(agent, key, j)
        const absGain = Math.abs(nextResources - prevResources)
        if (absGain > 0) {
          // ...
        }
      }
      */
    }
  }

  ctx.restore()
}
