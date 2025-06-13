import { Vec2f } from './vector_math.js';
import * as Common from './common.js';
import { ui, state, ctx } from './common.js';
import { getAttr } from './replay.js';
import { PanelInfo } from './panels.js';
import { parseHtmlColor } from './htmlutils.js';

/** Draw the mini map. */
export function drawMiniMap(panel: PanelInfo) {
  if (state.replay === null || ctx === null || ctx.ready === false) {
    return;
  }

  if (ui.mouseDown && panel.inside(ui.mousePos)) {
    const localMousePos = panel.transformOuter(ui.mousePos);
    // Pan the main map to the mini map's mouse position.
    const miniMapMousePos = new Vec2f(
      Math.round(localMousePos.x() / Common.MINI_MAP_TILE_SIZE),
      Math.round(localMousePos.y() / Common.MINI_MAP_TILE_SIZE)
    );
    ui.mapPanel.panPos = new Vec2f(
      -miniMapMousePos.x() * Common.TILE_SIZE - state.replay.map_size[0] * Common.TILE_SIZE / 2,
      -miniMapMousePos.y() * Common.TILE_SIZE - state.replay.map_size[1] * Common.TILE_SIZE / 2
    );
    state.followSelection = false;
  }

  // Mini map is always drawn as colored rectangles.
  ctx.save();
  ctx.setScissorRect(panel.x, panel.y, panel.width, panel.height);
  ctx.translate(panel.x, panel.y);

  // Draw background rect thats the size of the map.
  ctx.drawSolidRect(
    0,
    0,
    state.replay.map_size[0] * Common.MINI_MAP_TILE_SIZE,
    state.replay.map_size[1] * Common.MINI_MAP_TILE_SIZE,
    parseHtmlColor("#E7D4B7")
  );

  // Draw grid objects on the mini map.
  for (const gridObject of state.replay.grid_objects) {
    const x = getAttr(gridObject, "c");
    const y = getAttr(gridObject, "r");
    const type = getAttr(gridObject, "type");
    const typeName = state.replay.object_types[type];
    var color = parseHtmlColor("#FFFFFF");
    if (typeName === "wall") {
      color = parseHtmlColor("#61574B");
    } else if (typeName === "agent") {
      continue;
    }
    ctx.drawSolidRect(
      x * Common.MINI_MAP_TILE_SIZE,
      y * Common.MINI_MAP_TILE_SIZE,
      Common.MINI_MAP_TILE_SIZE,
      Common.MINI_MAP_TILE_SIZE,
      color
    );
  }

  // Draw agent pips on top
  for (const gridObject of state.replay.grid_objects) {
    const x = getAttr(gridObject, "c");
    const y = getAttr(gridObject, "r");
    const type = getAttr(gridObject, "type");
    const typeName = state.replay.object_types[type];
    if (typeName === "agent") {
      ctx.drawSprite(
        "minimapPip.png",
        x * Common.MINI_MAP_TILE_SIZE + 1,
        y * Common.MINI_MAP_TILE_SIZE + 1,
        [1, 0, 0, 1],
        1,
        0);
      continue;
    }
  }

  // Draw where the screen is on the mini map.
  const pos = new Vec2f(
    -ui.mapPanel.panPos.x() / Common.TILE_SIZE * Common.MINI_MAP_TILE_SIZE,
    -ui.mapPanel.panPos.y() / Common.TILE_SIZE * Common.MINI_MAP_TILE_SIZE
  );
  const width = ui.mapPanel.width / ui.mapPanel.zoomLevel / Common.TILE_SIZE * Common.MINI_MAP_TILE_SIZE;
  const height = ui.mapPanel.height / ui.mapPanel.zoomLevel / Common.TILE_SIZE * Common.MINI_MAP_TILE_SIZE;

  ctx.drawStrokeRect(
    pos.x() - width / 2,
    pos.y() - height / 2,
    width,
    height,
    1,
    [1, 1, 1, 1]
  );

  ctx.restore();
}
