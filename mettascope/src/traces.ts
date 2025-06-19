import { Vec2f } from './vector_math.js'
import * as Common from './common.js'
import { ui, state, ctx, setFollowSelection } from './common.js'
import { getAttr } from './replay.js'
import { PanelInfo } from './panels.js'
import { updateStep, updateSelection } from './main.js'
import { parseHtmlColor } from './htmlutils.js'

/** Draws the trace panel. */
export function drawTrace(panel: PanelInfo) {
  if (state.replay === null || ctx === null || ctx.ready === false) {
    return
  }

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
          mapX < state.replay.max_steps * Common.TRACE_WIDTH &&
          localMousePos.y() > 0 &&
          localMousePos.y() < state.replay.num_agents * Common.TRACE_HEIGHT &&
          selectedStep >= 0 &&
          selectedStep < state.replay.max_steps &&
          agentId >= 0 &&
          agentId < state.replay.num_agents
        ) {
          updateSelection(state.replay.agents[agentId])
          console.info('Selected an agent on a trace:', state.selectedGridObject)
          ui.mapPanel.focusPos(
            getAttr(state.selectedGridObject, 'c') * Common.TILE_SIZE,
            getAttr(state.selectedGridObject, 'r') * Common.TILE_SIZE,
            Common.DEFAULT_ZOOM_LEVEL
          )
          // Update the step to the clicked step.
          updateStep(selectedStep)
        }
      }
    }
  }

  // If we're following a selection, center the trace panel on it.
  if (state.followSelection && state.selectedGridObject !== null) {
    const x = state.step * Common.TRACE_WIDTH + Common.TRACE_WIDTH / 2
    const y = getAttr(state.selectedGridObject, 'agent_id') * Common.TRACE_HEIGHT + Common.TRACE_HEIGHT / 2
    panel.panPos = new Vec2f(-x, -y)
  }

  ctx.save()
  const rect = panel.rectInner()
  ctx.setScissorRect(rect.x, rect.y, rect.width, rect.height)

  const fullSize = new Vec2f(state.replay.max_steps * Common.TRACE_WIDTH, state.replay.num_agents * Common.TRACE_HEIGHT)

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
  if (state.selectedGridObject !== null && state.selectedGridObject.agent_id !== undefined) {
    const agentId = state.selectedGridObject.agent_id

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
  for (let i = 0; i < state.replay.num_agents; i++) {
    const agent = state.replay.agents[i]
    for (let j = 0; j < state.replay.max_steps; j++) {
      const action = getAttr(agent, 'action', j)
      const action_success = getAttr(agent, 'action_success', j)

      if (getAttr(agent, 'agent:frozen', j) > 0) {
        // Draw the frozen state.
        ctx.drawSprite(
          'trace/frozen.png',
          j * Common.TRACE_WIDTH + Common.TRACE_WIDTH / 2,
          i * Common.TRACE_HEIGHT + Common.TRACE_HEIGHT / 2
        )
      } else if (action_success && action != null && action[0] >= 0 && action[0] < state.replay.action_images.length) {
        // Draw the action.
        ctx.drawSprite(
          state.replay.action_images[action[0]],
          j * Common.TRACE_WIDTH + Common.TRACE_WIDTH / 2,
          i * Common.TRACE_HEIGHT + Common.TRACE_HEIGHT / 2
        )
      } else if (action != null && action[0] >= 0 && action[0] < state.replay.action_images.length) {
        // Draw the invalid action.
        ctx.drawSprite(
          'trace/invalid.png',
          j * Common.TRACE_WIDTH + Common.TRACE_WIDTH / 2,
          i * Common.TRACE_HEIGHT + Common.TRACE_HEIGHT / 2
        )
      }

      const reward = getAttr(agent, 'reward', j)
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
      if (state.showResources && j > 0) {
        // Figure out how many resources to draw.
        var number = 0
        for (const [key, [image, color]] of state.replay.resource_inventory) {
          number += Math.abs(getAttr(agent, key, j + 1) - getAttr(agent, key, j))
        }
        // Draw the resources.
        var y = 32
        // Compress the resources if there are too many so that they fit.
        var step = Math.min(32, (Common.TRACE_HEIGHT - 64) / number)
        for (const [key, [image, color]] of state.replay.resource_inventory) {
          const prevResources = getAttr(agent, key, j - 1)
          const nextResources = getAttr(agent, key, j)
          const absGain = Math.abs(nextResources - prevResources)
          for (let k = 0; k < absGain; k++) {
            ctx.drawSprite(
              image,
              j * Common.TRACE_WIDTH + Common.TRACE_WIDTH / 2,
              i * Common.TRACE_HEIGHT + y,
              color,
              1 / 4
            )
            y += step
          }
        }
      }
    }
  }

  ctx.restore()
}
