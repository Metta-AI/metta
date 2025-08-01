/**
 * This file handles the timeline scrubber.
 * Its main feature is that clicking on the scrubber updates the step.
 * It shows the current step as a counter.
 * It can show key actions on the timeline.
 * It has a box that shows what the traces view is looking at, so it acts as a
 * minimap for the traces.
 */

import * as Common from './common.js'
import { ctx, html, state, ui } from './common.js'
import { clamp } from './context3d.js'
import { onEvent } from './htmlutils.js'
import { requestFrame, updateStep } from './main.js'
import type { PanelInfo } from './panels.js'

/** Initializes the timeline. */
export function initTimeline() {
  console.info('Initializing timeline')
  // Move the step counter off-screen for now.
  html.stepCounter.parentElement!.style.left = '-1000px'
}

function getStepFromX(x: number) {
  const scrubberWidth = ui.timelinePanel.width - 32
  const s = Math.floor(((x - 16) / scrubberWidth) * state.replay.maxSteps)
  return clamp(s, 0, state.replay.maxSteps - 1)
}

/** Updates the scrubber. */
export function onScrubberChange(event: PointerEvent) {
  const mouseX = event.clientX
  const step = getStepFromX(mouseX)
  updateStep(step)
}

export function onTraceMinimapChange(event: PointerEvent) {
  const mouseX = event.clientX
  const step = getStepFromX(mouseX)
  ui.tracePanel.panPos.setX(-step * Common.TRACE_WIDTH)
}

/** Handles a pointer down on the timeline, which updates the step. */
onEvent('pointerdown', '#timeline-panel', (target: HTMLElement, e: Event) => {
  // Are we clicking on the scrubber or behind it (trace window) or event?
  const event = e as PointerEvent
  const mouseY = event.clientY - target.getBoundingClientRect().top
  const _mouseX = event.clientX
  if (mouseY > 34 && mouseY < 51) {
    ui.mainScrubberDown = true
    onScrubberChange(event)
  } else {
    // Click on the trace window.
    ui.mainTraceMinimapDown = true
    onTraceMinimapChange(event)
  }
})

/** Updates the timeline. */
export function updateTimeline() {
  if (state.replay === null) {
    return
  }

  const scrubberWidth = ui.timelinePanel.width - 32
  const fullSteps = state.replay.maxSteps - 1
  html.stepCounter.textContent = state.step.toString()
  html.stepCounter.parentElement!.style.left = `${(16 + (state.step / fullSteps) * scrubberWidth - 46 / 2).toString()}px`
}

/** Draws the timeline. */
export function drawTimeline(panel: PanelInfo) {
  if (state.replay === null || ctx === null || ctx.ready === false) {
    return
  }

  if (ui.mouseDoubleClick) {
    const step = getStepFromX(ui.mousePos.x())
    ui.tracePanel.panPos.setX(-step * Common.TRACE_WIDTH)
    requestFrame()
  }

  ctx.save()
  const rect = panel.rectInner()
  ctx.setScissorRect(rect.x, rect.y, rect.width, rect.height)
  ctx.translate(rect.x, rect.y)
  ctx.scale(ui.dpr, ui.dpr)

  const scrubberWidth = rect.width - 32
  const fullSteps = state.replay.maxSteps - 1

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
    const scrubberTileSize = scrubberWidth / fullSteps
    const tracesX = (-ui.tracePanel.panPos.x() / Common.TRACE_WIDTH) * scrubberTileSize
    const zoomLevel = ui.tracePanel.zoomLevel
    const tracesW = (ui.tracePanel.width / zoomLevel / Common.TRACE_WIDTH) * scrubberTileSize
    ctx.drawStrokeRect(16 + (tracesX - tracesW / 2) / ui.dpr - 1, 0, tracesW, 64, 1, [1, 1, 1, 1])
  }

  // Draw key actions on the timeline.
  for (const agent of state.replay.agents) {
    let prevFrozen = false
    for (let j = 0; j < state.replay.maxSteps; j++) {
      // Draw the frozen state.
      const isFrozen = agent.isFrozen.get(j)
      if (isFrozen && !prevFrozen) {
        const x = 16 + (j / fullSteps) * scrubberWidth
        ctx.drawSprite('agents/frozen.png', x, 12, [1, 1, 1, 1], 0.1, 0)
        ctx.drawSolidRect(x - 1, 24, 2, 8, [1, 1, 1, 1])
      }
      prevFrozen = isFrozen
    }
  }

  ctx.restore()
}
