/**
 * This file handles the timeline scrubber.
 * Its main feature is that clicking on the scrubber updates the step.
 * It shows the current step as a counter.
 * It can show key actions on the timeline.
 * It has a box that shows what the traces view is looking at, so it acts as a
 * minimap for the traces.
 */

import { PanelInfo } from './panels.js'
import * as Common from './common.js'
import { ui, state, html, ctx, setFollowSelection } from './common.js'
import { onEvent } from './htmlutils.js'
import { updateStep, requestFrame } from './main.js'
import { clamp } from './context3d.js'
import { getAttr } from './replay.js'
import { search, searchMatch, hasSearchTerm } from './search.js'
import { Vec2f } from './vector_math.js'

/** Initializes the timeline. */
export function initTimeline() {
  console.info('Initializing timeline')
  // Move the step counter off-screen for now.
  html.stepCounter.parentElement!.style.left = '-1000px'
}

function getStepFromX(x: number) {
  let scrubberWidth = ui.timelinePanel.width - 32
  let s = Math.floor(((x - 16) / scrubberWidth) * state.replay.max_steps)
  return clamp(s, 0, state.replay.max_steps - 1)
}

/** Updates the scrubber. */
export function onScrubberChange(event: MouseEvent) {
  let mouseX = event.clientX
  let step = getStepFromX(mouseX)
  updateStep(step)
}

export function onTraceMinimapChange(event: MouseEvent) {
  let mouseX = event.clientX
  let step = getStepFromX(mouseX)
  ui.tracePanel.panPos.setX(-step * Common.TRACE_WIDTH)
}

// /** Handles a mouse down on the timeline, which updates the step. */
// onEvent('mousedown', '#timeline-panel', (target: HTMLElement, e: Event) => {
//   // Are we clicking on the scrubber or behind it (trace window) or event?
//   let event = e as MouseEvent
//   let mouseY = event.clientY - target.getBoundingClientRect().top
//   let mouseX = event.clientX
//   if (mouseY > 34 && mouseY < 51) {
//     ui.mainScrubberDown = true
//     onScrubberChange(event)
//   } else {
//     // Click on the trace window.
//     ui.mainTraceMinimapDown = true
//     onTraceMinimapChange(event)
//   }
//   requestFrame()
// })

/** Updates the timeline. */
export function updateTimeline() {
  if (state.replay === null) {
    return
  }

  let scrubberWidth = ui.timelinePanel.width - 32
  let fullSteps = state.replay.max_steps - 1
  html.stepCounter.textContent = state.step.toString()
  html.stepCounter.parentElement!.style.left =
    (16 + (state.step / fullSteps) * scrubberWidth - 46 / 2).toString() + 'px'
}

function inside(point: Vec2f, x: number, y: number, w: number, h: number): boolean {
  return (
    point.x() >= x && point.x() < x + w && point.y() >= y && point.y() < y + h
  )
}

/** Draws the timeline. */
export function drawTimeline(panel: PanelInfo) {
  if (state.replay === null || ctx === null || ctx.ready === false) {
    return
  }

  let localMousePos = new Vec2f(ui.mousePos.x(), ui.mousePos.y() - ui.timelinePanel.y)

  if (ui.mouseDown) {
    let step = getStepFromX(localMousePos.x())
    if (localMousePos.y() > 34 && localMousePos.y() < 51) {
      ui.mainScrubberDown = true
      updateStep(step)
    } else {
      // Click on the trace window.
      ui.mainTraceMinimapDown = true
      ui.tracePanel.panPos.setX(-step * Common.TRACE_WIDTH)
    }
  }

  if (ui.mouseDoubleClick) {
    let step = getStepFromX(ui.mousePos.x())
    ui.tracePanel.panPos.setX(-step * Common.TRACE_WIDTH)
    requestFrame()
  }

  ctx.save()
  const rect = panel.rectInner()
  ctx.setScissorRect(rect.x, rect.y, rect.width, rect.height)
  ctx.translate(rect.x, rect.y)
  ctx.scale(ui.dpr, ui.dpr)

  let scrubberWidth = rect.width - 32
  let fullSteps = state.replay.max_steps - 1

  // Draw the background of the scrubber.
  ctx.drawSolidRect(
    16,
    34,
    scrubberWidth,
    16,
    [0.5, 0.5, 0.5, 0.5] // White with 50% opacity
  )

  // Draw the foreground of the scrubber.
  ctx.drawSolidRect(16, 34, (scrubberWidth * state.step) / fullSteps / ui.dpr, 16, [1, 1, 1, 1])

  if (state.showTraces) {
    // Draw the position of the traces view.
    let scrubberTileSize = scrubberWidth / fullSteps
    let tracesX = (-ui.tracePanel.panPos.x() / Common.TRACE_WIDTH) * scrubberTileSize
    let zoomLevel = ui.tracePanel.zoomLevel
    let tracesW = (ui.tracePanel.width / zoomLevel / Common.TRACE_WIDTH) * scrubberTileSize
    ctx.drawStrokeRect(16 + tracesX - tracesW / 2 - 1, 0, tracesW, 64, 1, [1, 1, 1, 1])
  }

  if (search.active) {
    // Draw searched states, actions or items on the timeline.

    // console.log('localMousePos', localMousePos.x(), localMousePos.y())

    // Draw frozen special state.
    if (hasSearchTerm("frozen")) {
      for (let agent of state.replay.agents) {
        let prevFrozen = 0
        for (let j = 0; j < state.replay.max_steps; j++) {
          // Draw the frozen state.
          let frozen = getAttr(agent, 'agent:frozen', j)
          if (frozen > 0 && prevFrozen == 0) {
            let x = 16 + (j / fullSteps) * scrubberWidth
            if (inside(localMousePos, x - 10, 2, 20, 20)) {
              ctx.drawSolidRect(x - 10, 2, 20, 20, [1, 1, 1, 1])
              if (ui.mouseClick) {
                state.selectedGridObject = agent
                setFollowSelection(true)
                updateStep(j)
              }
            }
            ctx.drawSprite('agents/frozen.png', x, 12, [1, 1, 1, 1], 0.1, 0)
            ctx.drawSolidRect(x - 1, 24, 2, 8, [1, 1, 1, 1])
          }
          prevFrozen = frozen
        }
      }
    }

    // Search is active, so we need to look for actions.
    for (let agent of state.replay.agents) {
      for (let j = 0; j < state.replay.max_steps; j++) {
        let action = getAttr(agent, 'action', j)
        let action_success = getAttr(agent, 'action_success', j)
        if (action !== undefined && action_success) {
          let actionName = state.replay.action_names[action[0]]
          if (searchMatch(actionName)) {
            let x = 16 + (j / fullSteps) * scrubberWidth
            if (inside(localMousePos, x - 10, 2, 20, 20)) {
              ctx.drawSolidRect(x - 10, 2, 20, 20, [1, 1, 1, 1])
              if (ui.mouseClick) {
                state.selectedGridObject = agent
                setFollowSelection(true)
                updateStep(j)
              }
            }
            ctx.drawSprite('actions/' + actionName + '.png', x, 12, [1, 1, 1, 1], 0.05, 0)
            ctx.drawSolidRect(x - 1, 24, 2, 8, [1, 1, 1, 1])
          }
        }
      }
    }

    // Draw resources.
    for (let [key, value] of state.replay.resource_inventory) {
      if (searchMatch(key)) {
        for (let agent of state.replay.agents) {
          let prevAmount = 0
          for (let j = 0; j < state.replay.max_steps; j++) {
            let amount = getAttr(agent, key, j)
            if (amount > prevAmount) {
              let x = 16 + (j / fullSteps) * scrubberWidth
              // Do click on the item icon.
              if (inside(localMousePos, x - 10, 2, 20, 20)) {
                ctx.drawSolidRect(x - 10, 2, 20, 20, [1, 1, 1, 1])
                if (ui.mouseClick) {
                  //console.log('clicked on resource', "agent:", agent.id, "step:", j, "resource:", key)
                  state.selectedGridObject = agent
                  setFollowSelection(true)
                  updateStep(j)
                }
              }
              ctx.drawSprite(value[0], x, 12, [1, 1, 1, 1], 0.1, 0)
              ctx.drawSolidRect(x - 1, 24, 2, 8, [1, 1, 1, 1])
            }
            prevAmount = amount
          }
        }
      }
    }
  }

  ctx.restore()
}
