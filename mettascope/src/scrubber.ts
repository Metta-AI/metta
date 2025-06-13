/**
 * This file handles the timeline scrubber.
 * Main features is that clicking on the scrubber will update the step.
 * It shows the current step as a counter.
 * It can show key actions on the timeline.
 * It has a box which is that the traces view is looking at, so it sort of
 * acts like a traces minimap.
*/

import { PanelInfo } from "./panels.js"
import * as Common from './common.js';
import { ui, state, html, ctx, setFollowSelection } from './common.js';
import { onEvent } from "./htmlutils.js";
import { updateStep } from './main.js';
import { clamp } from "./context3d.js";
import { getAttr } from "./replay.js";

/** Initialize the scrubber. */
export function initScrubber() {
  console.log("Initializing scrubber");
}

/** Update the scrubber. */
export function onScrubberChange(event: MouseEvent) {
  let mouseX = event.clientX;
  let scrubberWidth = ui.scrubberPanel.width - 32;
  let s = Math.floor((mouseX - 16) / scrubberWidth * state.replay.max_steps);
  let step = clamp(s, 0, state.replay.max_steps - 1);
  updateStep(step);
}

/** Handle mouse down on the scrubber, which will update the step. */
onEvent("mousedown", "#scrubber-panel", (target: HTMLElement, event: Event) => {
  console.log("Scrubber clicked");
  ui.mainScrubberDown = true;
  onScrubberChange(event as MouseEvent);
});

/** Update the scrubber. */
export function updateScrubber() {
  if (state.replay === null) {
    return;
  }

  let scrubberWidth = ui.scrubberPanel.width - 32;
  html.stepCounter.textContent = state.step.toString();
  html.stepCounter.parentElement!.style.left = (16 + state.step / state.replay.max_steps * scrubberWidth - 46 / 2).toString() + "px";
}

/** Draw the scrubber. */
export function drawScrubber(panel: PanelInfo) {
  if (state.replay === null || ctx === null || ctx.ready === false) {
    return;
  }

  ctx.save();
  ctx.setScissorRect(panel.x, panel.y, panel.width, panel.height);
  ctx.translate(panel.x, panel.y);

  let scrubberWidth = panel.width - 32;

  // Draw the background of the scrubber.
  ctx.drawSolidRect(
    16, 34,
    scrubberWidth, 16,
    [0.5, 0.5, 0.5, 0.5] // White with 50% opacity
  );

  // Draw the foreground of the scrubber.
  ctx.drawSolidRect(
    16, 34,
    scrubberWidth * state.step / state.replay.max_steps, 16,
    [1, 1, 1, 1]
  );

  // The the position of the traces view.
  let scrubberTileSize = scrubberWidth / state.replay.max_steps
  let tracesX = -ui.tracePanel.panPos.x() / Common.TRACE_WIDTH * scrubberTileSize
  let zoomLevel = ui.tracePanel.zoomLevel;
  let tracesW = (ui.tracePanel.width / zoomLevel) / Common.TRACE_WIDTH * scrubberTileSize;
  ctx.drawStrokeRect(
    16 + tracesX - tracesW / 2 - 1, 0,
    tracesW, 64,
    1,
    [1, 1, 1, 1]
  );

  // Draw key actions on the scrubber.
  for (let agent of state.replay.agents) {
    let prevFrozen = 0;
    for (let j = 0; j < state.replay.max_steps; j++) {

      // Draw frozen state.
      let frozen = getAttr(agent, "agent:frozen", j);
      if (frozen > 0 && prevFrozen == 0) {
        let x = j / state.replay.max_steps * scrubberWidth;
        ctx.drawSprite(
          "agents/frozen.png",
          x,
          12,
          [1, 1, 1, 1],
          0.1,
          0
        );
        ctx.drawSolidRect(
          x,
          24,
          1,
          8,
          [1, 1, 1, 1]
        );
      }
      prevFrozen = frozen;

    }
  }

  ctx.restore();
}
