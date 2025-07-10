import { Vec2f } from './vector_math.js'
import { Grid } from './grid.js'
import * as Common from './common.js'
import { ui, state, ctx, setFollowSelection } from './common.js'
import { getAttr, Entity, sendAction } from './replay.js'
import { PanelInfo } from './panels.js'
import { onFrame, updateSelection } from './main.js'
import { parseHtmlColor, find } from './htmlutils.js'
import { updateHoverPanel, updateReadout, HoverPanel } from './hoverpanels.js'
import { search, searchMatch } from './search.js'


/** Generates a color from an agent ID. */
function colorFromId(agentId: number) {
  let n = agentId + Math.PI + Math.E + Math.SQRT2
  return [(n * Math.PI) % 1.0, (n * Math.E) % 1.0, (n * Math.SQRT2) % 1.0, 1.0]
}

/** Checks to see if an Entity has any inventory. */
function hasInventory(obj: Entity) {
  return getAttr(obj.inventory).length > 0
}

/** Focus the screen on a specific area of the map. */
export function focusMap(x: number, y: number, w: number, h: number) {
  ui.mapPanel.focusPos(x, y, Math.min(ui.mapPanel.width / w, ui.mapPanel.height / h))
}

/** Makes the panel focus on the full map; used at the start of the replay. */
export function focusFullMap(panel: PanelInfo) {
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
    state.replay!.mapSize[0] * Common.TILE_SIZE,
    state.replay!.mapSize[1] * Common.TILE_SIZE,
    floorColor
  )
}

/** Draws the walls, based on the adjacency map, and fills any holes. */
function drawWalls() {
  // Construct a wall adjacency map.
  var wallMap = new Grid(state.replay!.mapSize[0], state.replay!.mapSize[1])
  for (const obj of state.replay!.objects) {
    const type = getAttr(obj.typeId)
    const typeName = state.replay!.typeNames[type]
    if (typeName !== 'wall') {
      continue
    }
    const position = getAttr(obj.position)
    wallMap.set(position[0], position[1], true)
  }

  // Draw the walls, following the adjacency map.
  for (const obj of state.replay!.objects) {
    const type = getAttr(obj.typeId)
    const typeName = state.replay!.typeNames[type]
    if (typeName !== 'wall') {
      continue
    }
    const position = getAttr(obj.position)
    const x = position[0]
    const y = position[1]
    var suffix = '0'
    var n = false,
      w = false,
      e = false,
      s = false
    if (wallMap.get(x, y - 1)) {
      n = true
    }
    if (wallMap.get(x - 1, y)) {
      w = true
    }
    if (wallMap.get(x, y + 1)) {
      s = true
    }
    if (wallMap.get(x + 1, y)) {
      e = true
    }
    if (n || w || e || s) {
      suffix = (n ? 'n' : '') + (w ? 'w' : '') + (s ? 's' : '') + (e ? 'e' : '')
    }
    ctx.drawSprite('objects/wall.' + suffix + '.png', x * Common.TILE_SIZE, y * Common.TILE_SIZE)
  }

  // Draw the wall infill, following the adjacency map.
  for (const obj of state.replay!.objects) {
    const type = getAttr(obj.typeId)
    const typeName = state.replay!.typeNames[type]
    if (typeName !== 'wall') {
      continue
    }
    const position = getAttr(obj.position)
    const x = position[0]
    const y = position[1]
    // If walls to the E, S, and SE are filled, draw a wall fill.
    var s = false,
      e = false,
      se = false
    if (wallMap.get(x + 1, y)) {
      e = true
    }
    if (wallMap.get(x, y + 1)) {
      s = true
    }
    if (wallMap.get(x + 1, y + 1)) {
      se = true
    }
    if (e && s && se) {
      ctx.drawSprite(
        'objects/wall.fill.png',
        x * Common.TILE_SIZE + Common.TILE_SIZE / 2,
        y * Common.TILE_SIZE + Common.TILE_SIZE / 2 - 42
      )
    }
  }
}

function drawObject(obj: Entity) {
  const type: number = getAttr(obj.typeId)
  const typeName: string = state.replay!.typeNames[type]
  if (typeName === 'wall') {
    // Walls are drawn in a different way.
    return
  }
  const position = getAttr(obj.position)
  const x = position[0]
  const y = position[1]

  if (getAttr(obj.agentId) !== null) {
    // Respect the orientation of an Entity, usually an agent.
    const orientation = getAttr(obj.rotation)
    var suffix = ''
    if (orientation == 0) {
      suffix = 'n'
    } else if (orientation == 1) {
      suffix = 's'
    } else if (orientation == 2) {
      suffix = 'w'
    } else if (orientation == 3) {
      suffix = 'e'
    }

    const agent_id = getAttr(obj.agentId)

    ctx.drawSprite(
      'agents/agent.' + suffix + '.png',
      x * Common.TILE_SIZE,
      y * Common.TILE_SIZE,
      colorFromId(agent_id)
    )
  } else {
    // Draw regular objects.

    // Draw the base layer.
    ctx.drawSprite(state.replayHelper.typeImages[type][0], x * Common.TILE_SIZE, y * Common.TILE_SIZE)

    // FIX ME: We now do color differently.
    // Draw the color layer.
    // var colorIdx = getAttr(obj.color)
    // if (colorIdx >= 0 && colorIdx < Common.COLORS.length) {
    //   ctx.drawSprite(
    //     state.replayHelper.typeImages[type][2],
    //     x * Common.TILE_SIZE,
    //     y * Common.TILE_SIZE,
    //     Common.COLORS[colorIdx][1]
    //   )
    // }

    // // Draw the item layer.
    // if (hasInventory(obj)) {
    //   ctx.drawSprite(state.replay.object_images[type][1], x * Common.TILE_SIZE, y * Common.TILE_SIZE)
    // }
  }
}

/** Draws all objects on the map (that are not walls). */
function drawObjects() {
  for (const obj of state.replay!.objects) {
    drawObject(obj)
  }
}

/** Draws actions above the objects. */
function drawActions() {
  for (const obj of state.replay!.objects) {
    const position = getAttr(obj.position)
    const x = position[0]
    const y = position[1]

    // Do agent actions.
    if (getAttr(obj.actionId) !== null) {
      // Draw the action.
      const actionId = getAttr(obj.actionId)
      const actionParameter = getAttr(obj.actionParameter)
      const actionSuccess = getAttr(obj.actionSuccess)
      if (actionSuccess && actionId != null) {
        const action_name = state.replay!.actionNames[actionId]
        const orientation = getAttr(obj.rotation)
        var rotation = 0
        if (orientation == 0) {
          rotation = Math.PI / 2 // North
        } else if (orientation == 1) {
          rotation = -Math.PI / 2 // South
        } else if (orientation == 2) {
          rotation = Math.PI // West
        } else if (orientation == 3) {
          rotation = 0 // East
        }
        if (action_name == 'attack' && actionParameter >= 1 && actionParameter <= 9) {
          ctx.drawSprite(
            'actions/attack' + actionParameter + '.png',
            x * Common.TILE_SIZE,
            y * Common.TILE_SIZE,
            [1, 1, 1, 1],
            1,
            rotation
          )
        } else if (action_name == 'attack_nearest') {
          ctx.drawSprite(
            'actions/attack_nearest.png',
            x * Common.TILE_SIZE,
            y * Common.TILE_SIZE,
            [1, 1, 1, 1],
            1,
            rotation
          )
        } else if (action_name == 'put_recipe_items') {
          ctx.drawSprite(
            'actions/put_recipe_items.png',
            x * Common.TILE_SIZE,
            y * Common.TILE_SIZE,
            [1, 1, 1, 1],
            1,
            rotation
          )
        } else if (action_name == 'get_output') {
          ctx.drawSprite(
            'actions/get_output.png',
            x * Common.TILE_SIZE,
            y * Common.TILE_SIZE,
            [1, 1, 1, 1],
            1,
            rotation
          )
        } else if (action_name == 'swap') {
          ctx.drawSprite('actions/swap.png', x * Common.TILE_SIZE, y * Common.TILE_SIZE, [1, 1, 1, 1], 1, rotation)
        }
      }
    }

    // Do building actions.
    if (getAttr(obj.cooldownProgress) > 0) {
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
    if (getAttr(obj.frozen)) {
      ctx.drawSprite('agents/frozen.png', x * Common.TILE_SIZE, y * Common.TILE_SIZE)
    }
  }
}

/** Draws the Entity's inventory. */
function drawInventory(useSearch = false) {
  if (!state.showResources) {
    return
  }

  for (const obj of state.replay!.objects) {
    const position = getAttr(obj.position)
    const x = position[0]
    const y = position[1]

    // Sum up the Entity's inventory in case we need to condense it.
    let inventoryX = Common.INVENTORY_PADDING
    let numItems = getAttr(obj.inventory).length
    // Draw the actual inventory icons.
    let advanceX = Math.min(32, (Common.TILE_SIZE - Common.INVENTORY_PADDING * 2) / numItems)

    for (const itemId of getAttr(obj.inventory)) {
      let itemName = state.replay!.itemNames[itemId]
      let icon = state.replayHelper.itemImages[itemId]
      if (useSearch) {
        if (!searchMatch(itemName)) {
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
        [1, 1, 1, 1],
        1 / 8,
        0
      )
      inventoryX += advanceX
    }
  }
}

/** Draws the rewards on the bottom of the Entity. */
function drawRewards() {
  for (const obj of state.replay!.objects) {
    const position = getAttr(obj.position)
    const x = position[0]
    const y = position[1]
    if (getAttr(obj.totalReward) !== null) {
      const totalReward = getAttr(obj.totalReward)
      let rewardX = 0
      let advanceX = Math.min(32, Common.TILE_SIZE / totalReward)
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

/** Draws the selection of the selected Entity. */
function drawSelection() {
  if (state.selectedGridObject === null) {
    return
  }

  const position = getAttr(state.selectedGridObject.position)
  const x = position[0]
  const y = position[1]
  ctx.drawSprite('selection.png', x * Common.TILE_SIZE, y * Common.TILE_SIZE)
}

/** Draws the trajectory of the selected Entity, with footprints or a future arrow. */
function drawTrajectory() {
  if (state.selectedGridObject === null) {
    return
  }
  if (state.selectedGridObject.position!.length > 0) {
    // Draw both past and future trajectories.
    for (let i = 1; i < state.replay!.maxSteps; i++) {
      const position0 = getAttr(state.selectedGridObject.position, i - 1)
      const position1 = getAttr(state.selectedGridObject.position, i)
      const cx0 = position0[0]
      const cy0 = position0[1]
      const cx1 = position1[0]
      const cy1 = position1[1]
      if (cx0 !== cx1 || cy0 !== cy1) {
        const a = 1 - Math.abs(i - state.step) / 200
        if (a > 0) {
          let color = [0, 0, 0, a]
          let image = ''
          if (state.step >= i) {
            // The past trajectory is black.
            color = [0, 0, 0, a]
            if (state.selectedGridObject.agentId !== null) {
              image = 'agents/footprints.png'
            } else {
              image = 'agents/past_arrow.png'
            }
          } else {
            // The future trajectory is white.
            color = [a, a, a, a]
            if (state.selectedGridObject.agentId !== null) {
              image = 'agents/path.png'
            } else {
              image = 'agents/future_arrow.png'
            }
          }

          if (cx1 > cx0) {
            // East
            ctx.drawSprite(image, cx0 * Common.TILE_SIZE, cy0 * Common.TILE_SIZE + 60, color, 1, 0)
          } else if (cx1 < cx0) {
            // West
            ctx.drawSprite(image, cx0 * Common.TILE_SIZE, cy0 * Common.TILE_SIZE + 60, color, 1, Math.PI)
          } else if (cy1 > cy0) {
            // South
            ctx.drawSprite(image, cx0 * Common.TILE_SIZE, cy0 * Common.TILE_SIZE + 60, color, 1, -Math.PI / 2)
          } else if (cy1 < cy0) {
            // North
            ctx.drawSprite(image, cx0 * Common.TILE_SIZE, cy0 * Common.TILE_SIZE + 60, color, 1, Math.PI / 2)
          }
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
  if (state.selectedGridObject != null && state.selectedGridObject.agentId != null) {
    // We need to find a key action in the future.
    // A key action is a successful action that is not a no-op, rotate, or move.
    // It must not be more than 20 steps in the future.
    var keyAction = null
    var keyActionStep = null
    for (
      var actionStep = state.step;
      actionStep < state.replay!.maxSteps && actionStep < state.step + 20;
      actionStep++
    ) {
      const actionId = getAttr(state.selectedGridObject.actionId, actionStep)
      if (actionId == null) {
        continue
      }
      const actionName = state.replay!.actionNames[actionId]
      const actionSuccess = getAttr(state.selectedGridObject.actionSuccess, actionStep)
      if (actionName == 'noop' || actionName == 'rotate' || actionName == 'move') {
        continue
      }
      if (actionSuccess) {
        keyAction = actionId
        keyActionStep = actionStep
        break
      }
    }

    if (keyAction != null) {
      // We have a key action, so draw the thought bubble.
      // Draw the key action icon with gained or lost resources.
      const position = getAttr(state.selectedGridObject.position)
      const x = position[0]
      const y = position[1]
      if (state.step == keyActionStep) {
        ctx.drawSprite(
          'actions/thoughts_lightning.png',
          x * Common.TILE_SIZE + Common.TILE_SIZE / 2,
          y * Common.TILE_SIZE - Common.TILE_SIZE / 2
        )
      } else {
        ctx.drawSprite(
          'actions/thoughts.png',
          x * Common.TILE_SIZE + Common.TILE_SIZE / 2,
          y * Common.TILE_SIZE - Common.TILE_SIZE / 2
        )
      }
      // Draw the action icon.
      var iconName = 'actions/icons/' + state.replay!.actionNames[keyAction] + '.png'
      if (ctx.hasImage(iconName)) {
        ctx.drawSprite(
          iconName,
          x * Common.TILE_SIZE + Common.TILE_SIZE / 2,
          y * Common.TILE_SIZE - Common.TILE_SIZE / 2,
          [1, 1, 1, 1],
          1 / 4,
          0
        )
      } else {
        ctx.drawSprite(
          'actions/icons/unknown.png',
          x * Common.TILE_SIZE + Common.TILE_SIZE / 2,
          y * Common.TILE_SIZE - Common.TILE_SIZE / 2,
          [1, 1, 1, 1],
          1 / 4,
          0
        )
      }

      // FIX ME:
      // // Draw the resources lost on the left and gained on the right.
      // for (const itemId of getAttr(state.selectedGridObject.inventory)) {
      //   const itemName = state.replay!.itemNames[itemId]
      //   const icon = state.replayHelper.itemImages[itemId]
      //   const prevResources = getAttr(state.selectedGridObject.inventory, actionStep - 1)
      //   const nextResources = getAttr(state.selectedGridObject.inventory, actionStep)
      //   const gained = nextResources - prevResources
      //   var resourceX = x * Common.TILE_SIZE + Common.TILE_SIZE / 2
      //   var resourceY = y * Common.TILE_SIZE - Common.TILE_SIZE / 2
      //   if (gained > 0) {
      //     resourceX += 32
      //   } else {
      //     resourceX -= 32
      //   }
      //   for (let i = 0; i < Math.abs(gained); i++) {
      //     ctx.drawSprite(icon, resourceX, resourceY, [1, 1, 1, 1], 1 / 8, 0)
      //     if (gained > 0) {
      //       resourceX += 8
      //     } else {
      //       resourceX -= 8
      //     }
      //   }
      // }
    }
  }
}

/** Draws the visibility map, either agent view ranges or fog of war. */
function drawVisibility() {
  if (state.showVisualRanges || state.showFogOfWar) {
    // Compute the visibility map; each agent contributes to the visibility map.
    const visibilityMap = new Grid(state.replay!.mapSize[0], state.replay!.mapSize[1])

    // Update the visibility map for a grid Entity.
    function updateVisibilityMap(obj: any) {
      const x = getAttr(obj, 'c')
      const y = getAttr(obj, 'r')
      var visionSize = Math.floor(getAttr(obj, 'agent:vision_size', state.step, Common.DEFAULT_VISION_SIZE) / 2)
      for (let dx = -visionSize; dx <= visionSize; dx++) {
        for (let dy = -visionSize; dy <= visionSize; dy++) {
          visibilityMap.set(x + dx, y + dy, true)
        }
      }
    }

    if (state.selectedGridObject !== null && state.selectedGridObject.agent_id !== undefined) {
      // When there is a selected grid Entity, only update its visibility.
      updateVisibilityMap(state.selectedGridObject)
    } else {
      // When there is no selected grid Entity, update the visibility map for all agents.
      for (const obj of state.replay.objects) {
        const type = getAttr(obj, 'type')
        const typeName = state.replay.type_names[type]
        if (typeName == 'agent') {
          updateVisibilityMap(obj)
        }
      }
    }

    var color = [0, 0, 0, 0.25]
    if (state.showFogOfWar) {
      color = [0, 0, 0, 1]
    }
    for (let x = 0; x < state.replay.map_size[0]; x++) {
      for (let y = 0; y < state.replay.map_size[1]; y++) {
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
    for (let x = 0; x < state.replay.map_size[0]; x++) {
      for (let y = 0; y < state.replay.map_size[1]; y++) {
        ctx.drawSprite('objects/grid.png', x * Common.TILE_SIZE, y * Common.TILE_SIZE)
      }
    }
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
  let dx, dy
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
  }
  return [dx, dy]
}

/** Draw the attack mode. */
function drawAttackMode() {
  // We might be clicking on the map to attack something.
  var gridMousePos: Vec2f | null = null
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
  if (state.selectedGridObject !== null && state.selectedGridObject.agent_id !== undefined) {
    const x = getAttr(state.selectedGridObject, 'c')
    const y = getAttr(state.selectedGridObject, 'r')
    const orientation = getAttr(state.selectedGridObject, 'agent:orientation')

    // Draw a 3x3 grid of targets in the direction of the selected agent.
    for (let attackIndex = 1; attackIndex <= 9; attackIndex++) {
      const [dx, dy] = attackGrid(orientation, attackIndex)
      const targetX = x + dx
      const targetY = y + dy
      ctx.drawSprite('target.png', targetX * Common.TILE_SIZE, targetY * Common.TILE_SIZE)
      if (gridMousePos != null && targetX == gridMousePos.x() && targetY == gridMousePos.y()) {
        // Check if we are clicking this specific tile.
        console.info('Attack mode clicked on:', targetX, targetY)
        sendAction('attack', attackIndex)
      }
    }
  }
}

/** Draw the info line from the Entity to the info panel. */
function drawInfoLine(panel: HoverPanel) {
  const x = getAttr(panel.Entity, 'c')
  const y = getAttr(panel.Entity, 'r')
  ctx.drawSprite('info.png', x * Common.TILE_SIZE, y * Common.TILE_SIZE)

  // Compute the panel position in the world map coordinates.
  const panelBounds = panel.div.getBoundingClientRect()
  const panelScreenPos = new Vec2f(panelBounds.left + 20, panelBounds.top + 20)
  const panelWorldPos = ui.mapPanel.transformOuter(panelScreenPos)

  // Draw a line from the Entity to the panel.
  ctx.drawSpriteLine(
    'dash.png',
    x * Common.TILE_SIZE,
    y * Common.TILE_SIZE,
    panelWorldPos.x(),
    panelWorldPos.y(),
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

  // Handle mouse events for the map panel.
  if (ui.mouseTargets.includes('#worldmap-panel')) {
    if (ui.dragging == '' && !state.showAttackMode) {
      // Find the Entity under the mouse.
      var objectUnderMouse = null
      const localMousePos = panel.transformOuter(ui.mousePos)
      if (localMousePos != null) {
        const gridMousePos = new Vec2f(
          Math.round(localMousePos.x() / Common.TILE_SIZE),
          Math.round(localMousePos.y() / Common.TILE_SIZE)
        )
        objectUnderMouse = state.replay.objects.find((obj: any) => {
          const x: number = getAttr(obj, 'c')
          const y: number = getAttr(obj, 'r')
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
      // Check if we are clicking on an Entity.
      if (objectUnderMouse !== undefined) {
        updateSelection(objectUnderMouse)
        console.info('Selected Entity on the map:', state.selectedGridObject)
        if (state.selectedGridObject.agent_id !== undefined) {
          // If selecting an agent, focus the trace panel on the agent.
          ui.tracePanel.focusPos(
            state.step * Common.TRACE_WIDTH + Common.TRACE_WIDTH / 2,
            getAttr(state.selectedGridObject, 'agent_id') * Common.TRACE_HEIGHT + Common.TRACE_HEIGHT / 2,
            Common.DEFAULT_TRACE_ZOOM_LEVEL
          )
        }
      }
    } else {
      ui.hoverObject = objectUnderMouse
      clearTimeout(ui.hoverTimer)
      ui.hoverTimer = setTimeout(() => {
        if (ui.mouseTargets.includes('#worldmap-panel')) {
          ui.delayedHoverObject = ui.hoverObject
          updateHoverPanel(ui.delayedHoverObject)
        }
      }, Common.INFO_PANEL_POP_TIME)
    }
  }

  // If we're following a selection, center the map on it.
  if (state.followSelection && state.selectedGridObject !== null) {
    const x = getAttr(state.selectedGridObject, 'c')
    const y = getAttr(state.selectedGridObject, 'r')
    panel.panPos = new Vec2f(-x * Common.TILE_SIZE, -y * Common.TILE_SIZE)
  }

  ctx.save()
  const rect = panel.rectInner()
  ctx.setScissorRect(rect.x, rect.y, rect.width, rect.height)

  ctx.translate(rect.x + rect.width / 2, rect.y + rect.height / 2)
  ctx.scale(panel.zoomLevel, panel.zoomLevel)
  ctx.translate(panel.panPos.x(), panel.panPos.y())

  drawFloor()
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

  if (search.active) {
    // Draw the black overlay over the map.
    ctx.drawSolidRect(
      -Common.TILE_SIZE / 2,
      -Common.TILE_SIZE / 2,
      state.replay.map_size[0] * Common.TILE_SIZE,
      state.replay.map_size[1] * Common.TILE_SIZE,
      [0, 0, 0, 0.80]
    )

    drawSelection()

    // Draw matching objects on top of the overlay.
    for (const obj of state.replay.objects) {
      const typeName = state.replay.type_names[getAttr(obj, 'type')]
      let x = getAttr(obj, 'c')
      let y = getAttr(obj, 'r')
      if (searchMatch(typeName)) {
        // Draw halo behind the Entity.
        ctx.drawSprite(
          'effects/halo.png',
          x * Common.TILE_SIZE,
          y * Common.TILE_SIZE,
          [1, 1, 1, 1],
          1.5,
          0
        )
        drawObject(obj)
      }
    }

    drawInventory(true)
  }



  if (state.showAttackMode) {
    drawAttackMode()
  }

  updateHoverPanel(ui.delayedHoverObject)
  updateReadout()

  for (const panel of ui.hoverPanels) {
    panel.update()
    drawInfoLine(panel)
  }

  ctx.restore()
}
