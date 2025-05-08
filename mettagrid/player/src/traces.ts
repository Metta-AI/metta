import { Vec2f } from './vector_math.js';
import * as Common from './common.js';
import { ui, state, ctx, setFollowSelection } from './common.js';
import { getAttr } from './replay.js';
import { PanelInfo } from './panels.js';
import { updateStep } from './main.js';

export function drawTrace(panel: PanelInfo) {
  if (state.replay === null || ctx === null || ctx.ready === false) {
    return;
  }

  const localMousePos = panel.transformPoint(ui.mousePos);

  if (state.followTraceSelection && state.selectedGridObject !== null) {
    panel.focusPos(
      state.step * Common.TRACE_WIDTH + Common.TRACE_WIDTH / 2,
      getAttr(state.selectedGridObject, "agent_id") * Common.TRACE_HEIGHT + Common.TRACE_HEIGHT / 2
    );
  }

  if (ui.mouseClick && panel.inside(ui.mousePos)) {
    if (localMousePos != null) {
      const mapX = localMousePos.x();
      if (mapX > 0 && mapX < state.replay.max_steps * Common.TRACE_WIDTH &&
        localMousePos.y() > 0 && localMousePos.y() < state.replay.num_agents * Common.TRACE_HEIGHT) {
        const agentId = Math.floor(localMousePos.y() / Common.TRACE_HEIGHT);
        if (agentId >= 0 && agentId < state.replay.num_agents) {
          setFollowSelection(true, null);
          state.selectedGridObject = state.replay.agents[agentId];
          console.log("selectedGridObject on a trace:", state.selectedGridObject);
          ui.mapPanel.focusPos(
            getAttr(state.selectedGridObject, "c") * Common.TILE_SIZE,
            getAttr(state.selectedGridObject, "r") * Common.TILE_SIZE
          );
          // Update the step to the clicked step.
          updateStep(Math.floor(mapX / Common.TRACE_WIDTH));

          if (ui.mouseDoubleClick) {
            setFollowSelection(null, true);
            panel.zoomLevel = 1;
          }
        }
      }
    }
  }

  ctx.save();
  ctx.setScissorRect(panel.x, panel.y, panel.width, panel.height);

  const fullSize = new Vec2f(
    state.replay.max_steps * Common.TRACE_WIDTH,
    state.replay.num_agents * Common.TRACE_HEIGHT
  );

  // Draw background
  ctx.drawSolidRect(
    panel.x, panel.y, panel.width, panel.height,
    [0.08, 0.08, 0.08, 1.0] // Dark background
  );

  ctx.translate(panel.x + panel.width / 2, panel.y + panel.height / 2);
  ctx.scale(panel.zoomLevel, panel.zoomLevel);
  ctx.translate(panel.panPos.x(), panel.panPos.y());

  // Draw rectangle around the selected agent
  if (state.selectedGridObject !== null && state.selectedGridObject.agent_id !== undefined) {
    const agentId = state.selectedGridObject.agent_id;

    // Draw selection rectangle
    ctx.drawSolidRect(
      0, agentId * Common.TRACE_HEIGHT, fullSize.x(), Common.TRACE_HEIGHT,
      [.3, .3, .3, 1]
    );
  }

  // Draw current step line that goes through all of the traces
  ctx.drawSolidRect(
    state.step * Common.TRACE_WIDTH, 0,
    Common.TRACE_WIDTH, fullSize.y(),
    [0.5, 0.5, 0.5, 0.5] // White with 50% opacity
  );

  // Draw agent traces
  for (let i = 0; i < state.replay.num_agents; i++) {
    const agent = state.replay.agents[i];
    for (let j = 0; j < state.replay.max_steps; j++) {
      const action = getAttr(agent, "action", j);
      const action_success = getAttr(agent, "action_success", j);

      if (action_success && action != null && action[0] > 0 && action[0] < state.replay.action_images.length) {
        ctx.drawSprite(
          state.replay.action_images[action[0]],
          j * Common.TRACE_WIDTH + Common.TRACE_WIDTH / 2,
          i * Common.TRACE_HEIGHT + Common.TRACE_HEIGHT / 2,
        );
      } else if (action != null && action[0] > 0 && action[0] < state.replay.action_images.length) {
        ctx.drawSprite(
          state.replay.action_images[action[0]],
          j * Common.TRACE_WIDTH + Common.TRACE_WIDTH / 2,
          i * Common.TRACE_HEIGHT + Common.TRACE_HEIGHT / 2,
          [0.01, 0.01, 0.01, 0.01],
        );
      }

      if (getAttr(agent, "agent:frozen", j) > 0) {
        ctx.drawSprite(
          "trace/frozen.png",
          j * Common.TRACE_WIDTH + Common.TRACE_WIDTH / 2,
          i * Common.TRACE_HEIGHT + Common.TRACE_HEIGHT / 2,
        );
      }

      const reward = getAttr(agent, "reward", j);
      // If there is reward, draw a star.
      if (reward > 0) {
        ctx.drawSprite(
          "resources/reward.png",
          j * Common.TRACE_WIDTH + Common.TRACE_WIDTH / 2,
          i * Common.TRACE_HEIGHT + 256 - 32,
          [1.0, 1.0, 1.0, 1.0],
          1 / 8
        );
      }
    }
  }

  ctx.restore();
}
