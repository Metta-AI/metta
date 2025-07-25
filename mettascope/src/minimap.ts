import * as Common from './common.js'
import { ctx, state, ui } from './common.js'
import { parseHtmlColor } from './htmlutils.js'
import type { PanelInfo } from './panels.js'
import { getAttr } from './replay.js'
import { Vec2f } from './vector_math.js'

/** Core minimap rendering logic that can be shared between minimap and macromap rendering. */
export function renderMinimapObjects(offset: Vec2f) {
  if (state.replay === null || ctx === null || ctx.ready === false) {
    return
  }

  // Draw a background rect that's the size of the map.
  ctx.drawSolidRect(
    offset.x(),
    offset.y(),
    state.replay.map_size[0],
    state.replay.map_size[1],
    parseHtmlColor('#E7D4B7')
  )

  // Draw the grid objects on the minimap.
  for (const gridObject of state.replay.grid_objects) {
    const x = getAttr(gridObject, 'c')
    const y = getAttr(gridObject, 'r')
    const type = getAttr(gridObject, 'type')
    const typeName = state.replay.object_types[type]
    let color = parseHtmlColor('#FFFFFF')
    if (typeName === 'wall') {
      color = parseHtmlColor('#61574B')
    } else if (typeName === 'agent') {
      continue // Draw agents separately on top.
    }
    ctx.drawSolidRect(x + offset.x(), y + offset.y(), 1, 1, color)
  }

  // Draw the agent pips on top.
  for (const gridObject of state.replay.grid_objects) {
    const x = getAttr(gridObject, 'c')
    const y = getAttr(gridObject, 'r')
    const type = getAttr(gridObject, 'type')
    const typeName = state.replay.object_types[type]
    const pipScale = 0.3
    if (typeName === 'agent') {
      const agent_id = getAttr(gridObject, 'agent_id')
      ctx.drawSprite(
        'minimapPip.png',
        x + offset.x() + 0.5,
        y + offset.y() + 0.5,
        Common.colorFromId(agent_id),
        pipScale,
        0
      )
    }
  }
}

/** Draws the minimap. */
export function drawMiniMap(panel: PanelInfo) {
  if (state.replay === null || ctx === null || ctx.ready === false) {
    return
  }

  if (ui.mouseDown && panel.inside(ui.mousePos)) {
    const localMousePos = new Vec2f(ui.mousePos.x() - panel.x, ui.mousePos.y() - panel.y)
    // Pan the main map to the minimap's mouse position.
    const miniMapMousePos = new Vec2f(
      Math.round(localMousePos.x() / Common.MINI_MAP_TILE_SIZE),
      Math.round(localMousePos.y() / Common.MINI_MAP_TILE_SIZE)
    )
    ui.mapPanel.panPos = new Vec2f(-miniMapMousePos.x() * Common.TILE_SIZE, -miniMapMousePos.y() * Common.TILE_SIZE)
    state.followSelection = false
  }

  // The minimap is always drawn as colored rectangles.
  ctx.save()
  const rect = panel.rectInner()
  ctx.setScissorRect(rect.x, rect.y, rect.width, rect.height)
  ctx.translate(rect.x, rect.y)
  ctx.scale(ui.dpr, ui.dpr)

  // Use the shared rendering logic.
  ctx.scale(Common.MINI_MAP_TILE_SIZE, Common.MINI_MAP_TILE_SIZE)
  renderMinimapObjects(new Vec2f(0, 0))

  // Draw where the screen is on the minimap.
  const pos = new Vec2f(
    (-ui.mapPanel.panPos.x() / Common.TILE_SIZE) * Common.MINI_MAP_TILE_SIZE,
    (-ui.mapPanel.panPos.y() / Common.TILE_SIZE) * Common.MINI_MAP_TILE_SIZE
  )
  const width = (ui.mapPanel.width / ui.mapPanel.zoomLevel / Common.TILE_SIZE) * Common.MINI_MAP_TILE_SIZE
  const height = (ui.mapPanel.height / ui.mapPanel.zoomLevel / Common.TILE_SIZE) * Common.MINI_MAP_TILE_SIZE

  ctx.drawStrokeRect(pos.x() - width / 2, pos.y() - height / 2, width, height, 1, [1, 1, 1, 1])

  ctx.restore()
}
