import * as Common from './common.js'
import { ctx, setFollowSelection, state, ui, HEATMAP_MIN_OPACITY, HEATMAP_MAX_OPACITY } from './common.js'
import { Grid } from './grid.js'
import { renderHeatmapTiles } from './heatmap.js'
import { type HoverBubble, updateHoverBubble, updateReadout } from './hoverbubbles.js'
import { parseHtmlColor } from './htmlutils.js'
import { updateSelection } from './main.js'
import { renderMinimapObjects, renderMinimapVisualRanges } from './minimap.js'
import type { PanelInfo } from './panels.js'
import { Entity, sendAction } from './replay.js'
import { search, searchMatch } from './search.js'
import { Vec2f } from './vector_math.js'

/**
 * Clamps the map panel's pan position so that the world map always remains at
 * least partially visible within the panel.
 */
function clampMapPan(panel: PanelInfo) {
  if (state.replay === null) {
    return
  }

  // The bounds of the world map in world-space coordinates. Tiles are drawn
  // starting at (−TILE_SIZE/2, −TILE_SIZE/2).
  const mapMinX = -Common.TILE_SIZE / 2
  const mapMinY = -Common.TILE_SIZE / 2
  const mapMaxX = state.replay.mapSize[0] * Common.TILE_SIZE - Common.TILE_SIZE / 2
  const mapMaxY = state.replay.mapSize[1] * Common.TILE_SIZE - Common.TILE_SIZE / 2

  // Dimensions of the visible area in world-space coordinates.
  const rect = panel.rectInner()
  const viewHalfWidth = rect.width / (2 * panel.zoomLevel)
  const viewHalfHeight = rect.height / (2 * panel.zoomLevel)

  // Current viewport centre in world-space.
  let cx = -panel.panPos.x()
  let cy = -panel.panPos.y()

  const mapWidth = mapMaxX - mapMinX
  const mapHeight = mapMaxY - mapMinY

  // Minimum number of pixels of the map that must remain visible.
  const minVisiblePixels = 500

  // Convert to world coordinates based on current zoom level.
  const minVisibleWorldUnits = minVisiblePixels / panel.zoomLevel

  // Ensure the required visible area doesn't exceed the actual map size.
  const maxVisibleUnitsX = Math.min(minVisibleWorldUnits, mapWidth / 2)
  const maxVisibleUnitsY = Math.min(minVisibleWorldUnits, mapHeight / 2)

  // Clamp horizontally.
  const minCenterX = mapMinX + maxVisibleUnitsX - viewHalfWidth
  const maxCenterX = mapMaxX - maxVisibleUnitsX + viewHalfWidth
  cx = Math.max(minCenterX, Math.min(cx, maxCenterX))

  // Clamp vertically.
  const minCenterY = mapMinY + maxVisibleUnitsY - viewHalfHeight
  const maxCenterY = mapMaxY - maxVisibleUnitsY + viewHalfHeight
  cy = Math.max(minCenterY, Math.min(cy, maxCenterY))

  panel.panPos = new Vec2f(-cx, -cy)
}

/** Checks to see if an object has any inventory. */
function hasInventory(obj: Entity) {
  const inventory = obj.inventory.get()
  return inventory.length > 0
}

/** Focus the screen on a specific area of the map. */
export function focusMap(x: number, y: number, w: number, h: number) {
  ui.mapPanel.focusPos(x, y, Math.min(ui.mapPanel.width / w, ui.mapPanel.height / h))
}

/** Makes the panel focus on the full map; used at the start of the replay. */
export function focusFullMap(_panel: PanelInfo) {
  if (state.replay === null) {
    return
  }
  const width = state.replay.mapSize[0] * Common.TILE_SIZE
  const height = state.replay.mapSize[1] * Common.TILE_SIZE
  focusMap(width / 2, height / 2, width, height)
}

/** Draws the floor. */
function drawFloor() {
  const floorColor = parseHtmlColor('#CFA970')
  ctx.drawSolidRect(
    -Common.TILE_SIZE / 2,
    -Common.TILE_SIZE / 2,
    state.replay.mapSize[0] * Common.TILE_SIZE,
    state.replay.mapSize[1] * Common.TILE_SIZE,
    floorColor
  )
}

const wallSprites = [
  'objects/wall.0.png',
  'objects/wall.e.png',
  'objects/wall.s.png',
  'objects/wall.se.png',
  'objects/wall.w.png',
  'objects/wall.we.png',
  'objects/wall.ws.png',
  'objects/wall.wse.png',
  'objects/wall.n.png',
  'objects/wall.ne.png',
  'objects/wall.ns.png',
  'objects/wall.nse.png',
  'objects/wall.nw.png',
  'objects/wall.nwe.png',
  'objects/wall.nws.png',
  'objects/wall.nwse.png',
]

const enum WallTile {
  None = 0,
  E = 1,
  S = 2,
  W = 4,
  N = 8,

  SE = S | E,
  NW = N | W,
}

let wallCells: Uint16Array | null = null
let wallFills: Uint16Array | null = null
let wallMap: Grid | null = null

/** Draws the walls, based on the adjacency map, and fills any holes. */
function drawWalls() {
  const width  = state.replay.mapSize[0]
  const height = state.replay.mapSize[1]
  const totalCells = width * height

  let numFills = 0
  let numWalls = 0
  if (wallCells === null || wallCells.length < totalCells) {
    wallCells = new Uint16Array(totalCells * 2)
    wallFills = new Uint16Array(totalCells * 2)
    wallMap = new Grid(width, height)
  }

  // Construct a wall adjacency map.
  for (const gridObject of state.replay.objects) {
    const type = gridObject.typeId
    const typeName = state.replay.typeNames[type]
    if (typeName !== 'wall') {
      continue
    }
    const location = gridObject.location.get()
    const x = location[0]
    const y = location[1]
    wallMap!.set(x, y, true)
    wallCells![numWalls++] = x
    wallCells![numWalls++] = y
  }

  // Draw the walls, following the adjacency map.
  for (let i = 0; i < numWalls; i += 2) {
    const x = wallCells![i]
    const y = wallCells![i + 1]

    let tile = WallTile.None
    if (wallMap!.get(x, y + 1)) tile |= WallTile.S
    if (wallMap!.get(x + 1, y)) tile |= WallTile.E
    if (wallMap!.get(x, y - 1)) tile |= WallTile.N
    if (wallMap!.get(x - 1, y)) tile |= WallTile.W

    if ((tile & WallTile.SE) == WallTile.SE && wallMap!.get(x + 1, y + 1)) {
      wallFills![numFills++] = x
      wallFills![numFills++] = y

      if ((tile & WallTile.NW) == WallTile.NW && wallMap!.get(x + 1, y - 1) && wallMap!.get(x - 1, y - 1) && wallMap!.get(x - 1, y + 1)) {
        continue
      }
    }

    ctx.drawSprite(wallSprites[tile], x * Common.TILE_SIZE, y * Common.TILE_SIZE)
  }

  // Draw the wall infills.
  for (let i = 0; i < numFills; i += 2) {
    const x = wallFills![i]
    const y = wallFills![i + 1]
    ctx.drawSprite(
      "objects/wall.fill.png",
      x * Common.TILE_SIZE + Common.TILE_SIZE / 2,
      y * Common.TILE_SIZE + Common.TILE_SIZE / 2 - 42
    )
  }
}

function drawObject(gridObject: Entity) {
  const type: number = gridObject.typeId
  const typeName: string = state.replay.typeNames[type]
  if (typeName === 'wall') {
    // Walls are drawn in a different way.
    return
  }
  const location = gridObject.location.get()
  const x = location[0]
  const y = location[1]

  if (gridObject.isAgent) {
    // Respect the orientation of an object, usually an agent.
    const orientation = gridObject.orientation.get()
    let suffix = ''
    if (orientation === 0) {
      suffix = 'n'
    } else if (orientation === 1) {
      suffix = 's'
    } else if (orientation === 2) {
      suffix = 'w'
    } else if (orientation === 3) {
      suffix = 'e'
    }

    const agentId = gridObject.agentId

    ctx.drawSprite(
      `agents/agent.${suffix}.png`,
      x * Common.TILE_SIZE,
      y * Common.TILE_SIZE,
      Common.colorFromId(agentId)
    )
  } else {
    // Draw regular objects.
    ctx.drawSprite(state.replay.objectImages[type], x * Common.TILE_SIZE, y * Common.TILE_SIZE)
  }
}

/** Draws all objects on the map (that are not walls). */
function drawObjects() {
  for (const gridObject of state.replay.objects) {
    drawObject(gridObject)
  }
}

/** Draws actions above the objects. */
function drawActions() {
  for (const gridObject of state.replay.objects) {
    const location = gridObject.location.get()
    const x = location[0]
    const y = location[1]

    // Do agent actions.
    if (gridObject.actionId.isSequence()) {
      // Draw the action.
      const actionId = gridObject.actionId.get()
      const actionParam = gridObject.actionParameter.get()
      const actionSuccess = gridObject.actionSuccess.get()
      if (actionSuccess && actionId != null) {
        const action_name = state.replay.actionNames[actionId]
        const orientation = gridObject.orientation.get()
        let rotation = 0
        if (orientation === 0) {
          rotation = Math.PI / 2 // North
        } else if (orientation === 1) {
          rotation = -Math.PI / 2 // South
        } else if (orientation === 2) {
          rotation = Math.PI // West
        } else if (orientation === 3) {
          rotation = 0 // East
        }
        if (action_name === 'attack' && actionParam >= 1 && actionParam <= 9) {
          ctx.drawSprite(
            `actions/attack${actionParam}.png`,
            x * Common.TILE_SIZE,
            y * Common.TILE_SIZE,
            [1, 1, 1, 1],
            1,
            rotation
          )
        } else if (action_name === 'attack_nearest') {
          ctx.drawSprite(
            'actions/attack_nearest.png',
            x * Common.TILE_SIZE,
            y * Common.TILE_SIZE,
            [1, 1, 1, 1],
            1,
            rotation
          )
        } else if (action_name === 'put_items') {
          ctx.drawSprite(
            'actions/put_recipe_items.png',
            x * Common.TILE_SIZE,
            y * Common.TILE_SIZE,
            [1, 1, 1, 1],
            1,
            rotation
          )
        } else if (action_name === 'get_items') {
          ctx.drawSprite(
            'actions/get_output.png',
            x * Common.TILE_SIZE,
            y * Common.TILE_SIZE,
            [1, 1, 1, 1],
            1,
            rotation
          )
        } else if (action_name === 'swap') {
          ctx.drawSprite('actions/swap.png', x * Common.TILE_SIZE, y * Common.TILE_SIZE, [1, 1, 1, 1], 1, rotation)
        }
      }
    }

    // Do building actions.
    if (gridObject.productionProgress.get() > 0) {
      ctx.drawSprite(
        'actions/converting.png',
        x * Common.TILE_SIZE,
        y * Common.TILE_SIZE - 100,
        [1, 1, 1, 1],
        1,
        // Apply a gentle rotation.
        -state.step * 0.1
      )
    }

    // Do states.
    if (gridObject.isFrozen.get()) {
      ctx.drawSprite('agents/frozen.png', x * Common.TILE_SIZE, y * Common.TILE_SIZE)
    }
  }
}

/** Draws the object's inventory. */
function drawInventory(useSearch = false) {
  if (!state.showResources) {
    return
  }

  for (const gridObject of state.replay.objects) {
    const location = gridObject.location.get()
    const x = location[0]
    const y = location[1]

    // Sum up the object's inventory in case we need to condense it.
    let inventoryX = Common.INVENTORY_PADDING
    let numItems = 0
    const inventory: [number, number][] = gridObject.inventory.get()
    for (const inventoryPair of inventory) {
      const num = inventoryPair[1]
      numItems += num
    }
    // If object has one output resource, and its in the inventory,
    // draw it in the center. This is used to draw the heart over an altar.
    if (
      gridObject.outputResources.length === 1 &&
      inventory.length === 1 &&
      inventory[0][1] === 1 &&
      inventory[0][0] === gridObject.outputResources[0][0]
    ) {
      const itemName = state.replay.itemNames[gridObject.outputResources[0][0]]
      if (useSearch) {
        if (searchMatch(itemName)) {
          // Draw halo behind the icon.
          ctx.drawSprite(
            'effects/halo.png',
            x * Common.TILE_SIZE + inventoryX - Common.TILE_SIZE / 2,
            y * Common.TILE_SIZE - Common.TILE_SIZE / 2 + 16,
            [1, 1, 1, 1],
            0.25,
            0
          )
        }
      }
      const icon = `resources/${itemName}.png`
      ctx.drawSprite(
        icon,
        x * Common.TILE_SIZE,
        y * Common.TILE_SIZE - Common.TILE_SIZE / 2 + 8,
        [1, 1, 1, 1],
        1 / 2,
        0
      )
      continue
    }

    // Draw the actual inventory icons.
    const advanceX = Math.min(32, (Common.TILE_SIZE - Common.INVENTORY_PADDING * 2) / numItems)
    for (const inventoryPair of inventory) {
      const inventoryId = inventoryPair[0]
      const num = inventoryPair[1]
      const key = state.replay.itemNames[inventoryId]
      const icon = `resources/${key}.png`
      const color = [1, 1, 1, 1]
      if (num > 0) {
        for (let i = 0; i < num; i++) {
          if (useSearch) {
            if (!searchMatch(key)) {
              inventoryX += advanceX
              continue
            }
            // Draw halo behind the icon.
            ctx.drawSprite(
              'effects/halo.png',
              x * Common.TILE_SIZE + inventoryX - Common.TILE_SIZE / 2,
              y * Common.TILE_SIZE - Common.TILE_SIZE / 2 + 16,
              [1, 1, 1, 1],
              0.25,
              0
            )
          }
          ctx.drawSprite(
            icon,
            x * Common.TILE_SIZE + inventoryX - Common.TILE_SIZE / 2,
            y * Common.TILE_SIZE - Common.TILE_SIZE / 2 + 16,
            color,
            1 / 8,
            0
          )
          inventoryX += advanceX
        }
      }
    }
  }
}

/** Draws the rewards on the bottom of the object. */
function drawRewards() {
  for (const gridObject of state.replay.objects) {
    const location = gridObject.location.get()
    const x = location[0]
    const y = location[1]
    if (gridObject.totalReward !== undefined) {
      const totalReward = gridObject.totalReward.get()
      let rewardX = 0
      const advanceX = Math.min(32, Common.TILE_SIZE / totalReward)
      for (let i = 0; i < totalReward; i++) {
        ctx.save()
        ctx.translate(
          x * Common.TILE_SIZE + rewardX - Common.TILE_SIZE / 2,
          y * Common.TILE_SIZE + Common.TILE_SIZE / 2 - 16
        )
        ctx.scale(1 / 8, 1 / 8)
        ctx.drawSprite('resources/reward.png', 0, 0)
        ctx.restore()
        rewardX += advanceX
      }
    }
  }
}

/** Draws the selection of the selected object. */
function drawSelection() {
  if (state.selectedGridObject === null) {
    return
  }

  const location = state.selectedGridObject.location.get()
  const x = location[0]
  const y = location[1]
  ctx.drawSprite('selection.png', x * Common.TILE_SIZE, y * Common.TILE_SIZE)
}

/** Draws the trajectory of the selected object, with footprints or a future arrow. */
function drawTrajectory() {
  if (state.selectedGridObject != null && state.selectedGridObject!.location.length() > 1) {
    // Draw both past and future trajectories.
    for (let i = 1; i < state.replay.maxSteps; i++) {
      const location0 = state.selectedGridObject.location.get(i - 1)
      const cx0 = location0[0]
      const cy0 = location0[1]
      const location1 = state.selectedGridObject.location.get(i)
      const cx1 = location1[0]
      const cy1 = location1[1]

      if (cx0 !== cx1 || cy0 !== cy1) {
        const a = 1 - Math.abs(i - state.step) / 200
        if (a > 0) {
          let color = [0, 0, 0, a]
          let image = ''
          if (state.step >= i) {
            // The past trajectory is black.
            color = [0, 0, 0, a]
            if (state.selectedGridObject.isAgent) {
              image = 'agents/footprints.png'
            } else {
              image = 'agents/past_arrow.png'
            }
          } else {
            // The future trajectory is white.
            color = [a, a, a, a]
            if (state.selectedGridObject.isAgent) {
              image = 'agents/path.png'
            } else {
              image = 'agents/future_arrow.png'
            }
          }

          // Calculate movement direction and rotation for both cardinal and diagonal movements.
          const dx = cx1 - cx0
          const dy = cy1 - cy0
          let rotation = 0
          let scale: number | [number, number] = 1

          if (dx > 0 && dy === 0) {
            // Movement is due east.
            rotation = 0
          } else if (dx < 0 && dy === 0) {
            // Movement is due west.
            rotation = Math.PI
          } else if (dx === 0 && dy > 0) {
            // Movement is due south.
            rotation = -Math.PI / 2
          } else if (dx === 0 && dy < 0) {
            // Movement is due north.
            rotation = Math.PI / 2
          } else if (dx > 0 && dy > 0) {
            // Movement is southeast diagonal.
            rotation = -Math.PI / 4
            scale = [Math.sqrt(2), 1]
          } else if (dx > 0 && dy < 0) {
            // Movement is northeast diagonal.
            rotation = Math.PI / 4
            scale = [Math.sqrt(2), 1]
          } else if (dx < 0 && dy > 0) {
            // Movement is southwest diagonal.
            rotation = (-3 * Math.PI) / 4
            scale = [Math.sqrt(2), 1]
          } else if (dx < 0 && dy < 0) {
            // Movement is northwest diagonal.
            rotation = (3 * Math.PI) / 4
            scale = [Math.sqrt(2), 1]
          }

          ctx.drawSprite(image, cx0 * Common.TILE_SIZE, cy0 * Common.TILE_SIZE + 60, color, scale, rotation)
        }
      }
    }
  }
}

/** Draws the thought bubbles of the selected agent. */
function drawThoughtBubbles() {
  // The idea behind thought bubbles is to show what an agent is thinking.
  // We don't have this directly from the policy yet, so the next best thing
  // is to show a future "key action."
  // It should be a good proxy for what the agent is thinking about.
  if (state.selectedGridObject != null && state.selectedGridObject.isAgent) {
    // We need to find a key action in the future.
    // A key action is a successful action that is not a no-op, rotate, or move.
    // It must not be more than 20 steps in the future.
    let keyAction = null
    let keyActionStep = null
    let actionHasTarget = false
    const actionStepEnd = Math.min(state.replay.maxSteps, state.step + 20)
    for (let actionStep = state.step; actionStep < actionStepEnd; actionStep++) {
      const actionId = state.selectedGridObject.actionId.get(actionStep)
      const actionParam = state.selectedGridObject.actionParameter.get(actionStep)
      if (actionId == null || actionParam == null) {
        continue
      }
      const actionSuccess = state.selectedGridObject.actionSuccess.get(actionStep)
      if (!actionSuccess) {
        continue
      }
      const actionName = state.replay.actionNames[actionId]
      if (
        actionName === 'noop' ||
        actionName === 'rotate' ||
        actionName === 'move' ||
        actionName === 'move_cardinal' ||
        actionName === 'move_8way'
      ) {
        continue
      }
      keyAction = [actionId, actionParam]
      keyActionStep = actionStep
      actionHasTarget = !(actionName === 'attack' || actionName === 'attack_nearest')
      break
    }

    if (keyAction != null && keyActionStep != null) {
      const location = state.selectedGridObject.location.get()
      const x = (location[0] + 0.5) * Common.TILE_SIZE
      const y = (location[1] - 0.5) * Common.TILE_SIZE
      if (actionHasTarget && keyActionStep !== state.step) {
        // Draw an arrow on a circle around the target, pointing at it.
        const targetLocation = state.selectedGridObject.location.get(keyActionStep)
        const [targetGridX, targetGridY] = applyOrientationOffset(
          targetLocation[0],
          targetLocation[1],
          state.selectedGridObject.orientation.get(keyActionStep)
        )
        const targetX = (targetGridX + 0.5) * Common.TILE_SIZE
        const targetY = (targetGridY - 0.5) * Common.TILE_SIZE
        const angle = Math.atan2(targetX - x, targetY - y)
        const r = Common.TILE_SIZE / 3
        const tX = targetX - Math.sin(angle) * r - Common.TILE_SIZE / 2
        const tY = targetY - Math.cos(angle) * r + Common.TILE_SIZE / 2
        ctx.drawSprite('actions/arrow.png', tX, tY, undefined, undefined, angle + Math.PI)
      }
      // We have a key action, so draw the thought bubble.
      // Draw the key action icon with gained or lost resources.
      if (state.step === keyActionStep) {
        ctx.drawSprite('actions/thoughts_lightning.png', x, y)
      } else {
        ctx.drawSprite('actions/thoughts.png', x, y)
      }
      // Draw the action icon.
      const iconName = `actions/icons/${state.replay.actionNames[keyAction[0]]}.png`
      if (ctx.hasImage(iconName)) {
        ctx.drawSprite(iconName, x, y, [1, 1, 1, 1], 1 / 4, 0)
      } else {
        ctx.drawSprite('actions/icons/unknown.png', x, y, [1, 1, 1, 1], 1 / 4, 0)
      }

      // Draw the resources lost on the left and gained on the right.
      let gainX = x + 32
      let lossX = x - 32
      const gainMap = state.selectedGridObject.gainMap[keyActionStep]
      for (const [inventoryId, inventoryAmount] of gainMap) {
        const inventoryName = state.replay.itemNames[inventoryId]
        const inventoryImage = `resources/${inventoryName}.png`
        if (inventoryAmount > 0) {
          ctx.drawSprite(inventoryImage, gainX, y, [1, 1, 1, 1], 1 / 8, 0)
          gainX += 8
        } else {
          ctx.drawSprite(inventoryImage, lossX, y, [1, 1, 1, 1], 1 / 8, 0)
          lossX -= 8
        }
      }
    }
  }
}

/** Draws the visibility map, either agent view ranges or fog of war. */
function drawVisibility() {
  if (state.showVisualRanges || state.showFogOfWar) {
    // Compute the visibility map; each agent contributes to the visibility map.
    const visibilityMap = new Grid(state.replay.mapSize[0], state.replay.mapSize[1])

    // Update the visibility map for a grid object.
    function updateVisibilityMap(gridObject: Entity) {
      const location = gridObject.location.get()
      const x = location[0]
      const y = location[1]
      const visionSize = Math.floor(gridObject.visionSize / 2)
      for (let dx = -visionSize; dx <= visionSize; dx++) {
        for (let dy = -visionSize; dy <= visionSize; dy++) {
          visibilityMap.set(x + dx, y + dy, true)
        }
      }
    }

    if (state.selectedGridObject !== null && state.selectedGridObject.isAgent) {
      // When there is a selected grid object, only update its visibility.
      updateVisibilityMap(state.selectedGridObject)
    } else {
      // When there is no selected grid object, update the visibility map for all agents.
      for (const gridObject of state.replay.objects) {
        const type = gridObject.typeId
        const typeName = state.replay.typeNames[type]
        if (typeName === 'agent') {
          updateVisibilityMap(gridObject)
        }
      }
    }

    let color = [0, 0, 0, 0.25]
    if (state.showFogOfWar) {
      color = [0, 0, 0, 1]
    }
    for (let x = 0; x < state.replay.mapSize[0]; x++) {
      for (let y = 0; y < state.replay.mapSize[1]; y++) {
        if (!visibilityMap.get(x, y)) {
          ctx.drawSolidRect(
            x * Common.TILE_SIZE - Common.TILE_SIZE / 2,
            y * Common.TILE_SIZE - Common.TILE_SIZE / 2,
            Common.TILE_SIZE,
            Common.TILE_SIZE,
            color
          )
        }
      }
    }
  }
}

/** Draws the grid. */
function drawGrid() {
  if (state.showGrid) {
    for (let x = 0; x < state.replay.mapSize[0]; x++) {
      for (let y = 0; y < state.replay.mapSize[1]; y++) {
        ctx.drawSprite('objects/grid.png', x * Common.TILE_SIZE, y * Common.TILE_SIZE)
      }
    }
  }
}

/** Draws the heatmap overlay. */
function drawHeatmap() {
  if (state.showHeatmap) {
    renderHeatmapTiles(state.step, (x: number, y: number, color: [number, number, number, number]) => {
      ctx.drawSolidRect(
        x * Common.TILE_SIZE - Common.TILE_SIZE / 2,
        y * Common.TILE_SIZE - Common.TILE_SIZE / 2,
        Common.TILE_SIZE,
        Common.TILE_SIZE,
        color
      )
    })
  }
}

/** Given a position and an orientation, returns the position offset by the orientation. */
function applyOrientationOffset(x: number, y: number, orientation: number) {
  switch (orientation) {
    case 0:
      return [x, y - 1]
    case 1:
      return [x, y + 1]
    case 2:
      return [x - 1, y]
    case 3:
      return [x + 1, y]
    default:
      return [x, y]
  }
}

/** Given an orientation and an index, returns the grid position. */
function attackGrid(orientation: number, idx: number) {
  //                           North\0
  //                       +---+---+---+
  //                       | 7 | 8 | 9 |
  //                       +---+---+---+
  //                       | 4 | 5 | 6 |
  //                       +---+---+---+
  //                       | 1 | 2 | 3 |
  //                       +---+---+---+
  //                             ●
  //        +---+---+---+                 +---+---+---+
  //        | 9 | 6 | 3 |                 | 1 | 4 | 7 |
  //        +---+---+---+                 +---+---+---+
  // West\2 | 8 | 5 | 2 | ●             ● | 2 | 5 | 8 | East/3
  //        +---+---+---+                 +---+---+---+
  //        | 7 | 4 | 1 |                 | 3 | 6 | 9 |
  //        +---+---+---+                 +---+---+---+
  //                             ●
  //                       +---+---+---+
  //                       | 3 | 2 | 1 |
  //                       +---+---+---+
  //                       | 6 | 5 | 4 |
  //                       +---+---+---+
  //                       | 9 | 8 | 7 |
  //                       +---+---+---+
  //                           South/1
  //

  // Modulo operation.
  function mod(a: number, b: number) {
    return ((a % b) + b) % b
  }

  // Integer division.
  function div(a: number, b: number) {
    return Math.floor(a / b)
  }

  const i = idx - 1
  let dx: number
  let dy: number
  if (orientation === 0) {
    dx = mod(i, 3) - 1
    dy = -div(i, 3) - 1
  } else if (orientation === 1) {
    dx = -mod(i, 3) + 1
    dy = div(i, 3) + 1
  } else if (orientation === 2) {
    dx = -div(i, 3) - 1
    dy = -mod(i, 3) + 1
  } else if (orientation === 3) {
    dx = div(i, 3) + 1
    dy = mod(i, 3) - 1
  } else {
    throw new Error(`Invalid orientation: ${orientation}`)
  }
  return [dx, dy]
}

/** Draw the attack mode. */
function drawAttackMode() {
  // We might be clicking on the map to attack something.
  let gridMousePos: Vec2f | null = null
  if (ui.mouseUp && ui.mouseTargets.includes('#worldmap-panel') && state.showAttackMode) {
    state.showAttackMode = false
    const localMousePos = ui.mapPanel.transformOuter(ui.mousePos)
    if (localMousePos != null) {
      gridMousePos = new Vec2f(
        Math.round(localMousePos.x() / Common.TILE_SIZE),
        Math.round(localMousePos.y() / Common.TILE_SIZE)
      )
    }
  }

  // Draw a selection of 3x3 grid of targets in the direction of the selected agent.
  if (state.selectedGridObject !== null && state.selectedGridObject.isAgent) {
    const location = state.selectedGridObject.location.get()
    const x = location[0]
    const y = location[1]
    const orientation = state.selectedGridObject.orientation.get()

    // Draw a 3x3 grid of targets in the direction of the selected agent.
    for (let attackIndex = 1; attackIndex <= 9; attackIndex++) {
      const [dx, dy] = attackGrid(orientation, attackIndex)
      const targetX = x + dx
      const targetY = y + dy
      ctx.drawSprite('target.png', targetX * Common.TILE_SIZE, targetY * Common.TILE_SIZE)
      if (gridMousePos != null && targetX === gridMousePos.x() && targetY === gridMousePos.y()) {
        // Check if we are clicking this specific tile.
        console.info('Attack mode clicked on:', targetX, targetY)
        sendAction('attack', attackIndex)
      }
    }
  }
}

/** Draw the info line from the object to the info panel. */
function drawInfoLine(bubble: HoverBubble) {
  const location = bubble.object.location.get()
  const x = location[0]
  const y = location[1]
  ctx.drawSprite('info.png', x * Common.TILE_SIZE, y * Common.TILE_SIZE)

  // Compute the bubble position in the world map coordinates.
  const bubbleBounds = bubble.div.getBoundingClientRect()
  const bubbleScreenPos = new Vec2f(bubbleBounds.left + 20, bubbleBounds.top + 20)
  const bubbleWorldPos = ui.mapPanel.transformOuter(bubbleScreenPos)

  // Draw a line from the object to the bubble.
  ctx.drawSpriteLine(
    'dash.png',
    x * Common.TILE_SIZE,
    y * Common.TILE_SIZE,
    bubbleWorldPos.x(),
    bubbleWorldPos.y(),
    60,
    [1, 1, 1, 1],
    2
  )
}

/** Draws the world map. */
export function drawMap(panel: PanelInfo) {
  if (state.replay === null || ctx === null || ctx.ready === false) {
    return
  }
  let objectUnderMouse = null
  // Handle mouse events for the map panel.
  if (ui.mouseTargets.includes('#worldmap-panel')) {
    if (ui.dragging === '' && !state.showAttackMode) {
      // Find the object under the mouse.

      const localMousePos = panel.transformOuter(ui.mousePos)
      if (localMousePos != null) {
        const gridMousePos = new Vec2f(
          Math.round(localMousePos.x() / Common.TILE_SIZE),
          Math.round(localMousePos.y() / Common.TILE_SIZE)
        )
        objectUnderMouse = state.replay.objects.find((obj: Entity) => {
          const location = obj.location.get()
          const x = location[0]
          const y = location[1]
          return x === gridMousePos.x() && y === gridMousePos.y()
        })
      }
    }

    if (ui.mouseDoubleClick) {
      // Toggle followSelection on double-click.
      console.info('Map double click - following selection', ui.mouseTargets)
      setFollowSelection(true)
      panel.zoomLevel = Common.DEFAULT_ZOOM_LEVEL
      ui.tracePanel.zoomLevel = Common.DEFAULT_TRACE_ZOOM_LEVEL
    } else if (ui.mouseClick) {
      // A map click is likely a drag/pan.
      console.info('Map click - clearing follow selection')
      setFollowSelection(false)
    } else if (ui.mouseUp && ui.mouseDownPos.sub(ui.mousePos).length() < 10) {
      // Check if we are clicking on an object.
      if (objectUnderMouse !== undefined && objectUnderMouse !== null) {
        updateSelection(objectUnderMouse)
        console.info('Selected object on the map:', state.selectedGridObject)
        if (state.selectedGridObject !== null && state.selectedGridObject.isAgent) {
          // If selecting an agent, focus the trace panel on the agent.
          ui.tracePanel.focusPos(
            state.step * Common.TRACE_WIDTH + Common.TRACE_WIDTH / 2,
            state.selectedGridObject.agentId * Common.TRACE_HEIGHT + Common.TRACE_HEIGHT / 2,
            Common.DEFAULT_TRACE_ZOOM_LEVEL
          )
        }
      }
    } else {
      // Only reset the hover timer if we moved onto a different object (or off of an object).
      if (objectUnderMouse && ui.hoverObject !== objectUnderMouse) {
        ui.hoverObject = objectUnderMouse
        clearTimeout(ui.hoverTimer)
        ui.hoverTimer = setTimeout(() => {
          if (ui.mouseTargets.includes('#worldmap-panel')) {
            ui.delayedHoverObject = ui.hoverObject
            updateHoverBubble(ui.delayedHoverObject)
          }
        }, Common.INFO_PANEL_POP_TIME)
      } else if (!objectUnderMouse && ui.hoverObject !== null) {
        // Reset hover state when moving to empty space.
        ui.hoverObject = null
        clearTimeout(ui.hoverTimer)
      }
    }
  }

  // If we're following a selection, center the map on it.
  if (state.followSelection && state.selectedGridObject !== null) {
    const location = state.selectedGridObject.location.get()
    const x = location[0]
    const y = location[1]
    panel.panPos = new Vec2f(-x * Common.TILE_SIZE, -y * Common.TILE_SIZE)
  }

  // Ensure that at least a portion of the map remains visible.
  clampMapPan(panel)

  ctx.save()
  const rect = panel.rectInner()
  ctx.setScissorRect(rect.x, rect.y, rect.width, rect.height)

  ctx.translate(rect.x + rect.width / 2, rect.y + rect.height / 2)
  ctx.scale(panel.zoomLevel, panel.zoomLevel)
  ctx.translate(panel.panPos.x(), panel.panPos.y())

  if (panel.zoomLevel < Common.MACROMAP_ZOOM_THRESHOLD) {
    /** Draw the Macromap of the world map instead of objects.
     * The user has zoomed out so far that we should switch to a minimap-style rendering.
     * This is used when the user zooms out far enough that normal
     * sprites would be unreadable. */
    ctx.save()
    ctx.scale(Common.TILE_SIZE, Common.TILE_SIZE)
    renderMinimapObjects(new Vec2f(-0.5, -0.5))
    renderMinimapVisualRanges(new Vec2f(-0.5, -0.5))
    ctx.restore()
    drawSelection()
  } else {
    drawFloor()
    drawHeatmap()
    drawWalls()
    drawTrajectory()
    drawObjects()
    drawActions()
    drawSelection()
    drawInventory()
    drawRewards()
    drawVisibility()
    drawGrid()
    drawThoughtBubbles()
  }

  if (search.active) {
    // Draw the black overlay over the map.
    ctx.drawSolidRect(
      -Common.TILE_SIZE / 2,
      -Common.TILE_SIZE / 2,
      state.replay.mapSize[0] * Common.TILE_SIZE,
      state.replay.mapSize[1] * Common.TILE_SIZE,
      [0, 0, 0, 0.8]
    )

    drawSelection()

    // Draw matching objects on top of the overlay.
    for (const gridObject of state.replay.objects) {
      const typeId = gridObject.typeId
      const typeName = state.replay.typeNames[typeId]
      const location = gridObject.location.get()
      const x = location[0]
      const y = location[1]
      if (searchMatch(typeName)) {
        // Draw halo behind the object.
        ctx.drawSprite('effects/halo.png', x * Common.TILE_SIZE, y * Common.TILE_SIZE, [1, 1, 1, 1], 1.5, 0)
        drawObject(gridObject)
      }
    }

    drawInventory(true)
  }

  if (state.showAttackMode) {
    drawAttackMode()
  }

  updateHoverBubble(ui.delayedHoverObject)
  updateReadout()

  for (const bubble of ui.hoverBubbles) {
    bubble.update()
    drawInfoLine(bubble)
  }

  ctx.restore()
}
