import { Vec2f } from './vector_math.js';
import { Grid } from './grid.js';
import * as Common from './common.js';
import { ui, state, ctx, setFollowSelection } from './common.js';
import { getAttr, sendAction } from './replay.js';
import { PanelInfo } from './panels.js';
import { onFrame } from './main.js';
import { parseHtmlColor, find } from './htmlutils.js';


// Flag to prevent multiple calls to requestAnimationFrame
let frameRequested = false;

// Function to safely request animation frame
export function requestFrame() {
  if (!frameRequested) {
    frameRequested = true;
    requestAnimationFrame((time) => {
      frameRequested = false;
      onFrame();
    });
  }
}

// Generate a color from an agent id.
function colorFromId(agentId: number) {
  let n = agentId + Math.PI + Math.E + Math.SQRT2;
  return [
    n * Math.PI % 1.0,
    n * Math.E % 1.0,
    n * Math.SQRT2 % 1.0,
    1.0
  ]
}

// Checks to see of object has any inventory.
function hasInventory(obj: any) {
  for (const [key, [icon, color]] of state.replay.resource_inventory) {
    if (getAttr(obj, key) > 0) {
      return true;
    }
  }
  return false;
}

// Make the panel focus on the full map, used at the start of the replay.
export function focusFullMap(panel: PanelInfo) {
  if (state.replay === null) {
    return;
  }
  const width = state.replay.map_size[0] * Common.TILE_SIZE;
  const height = state.replay.map_size[1] * Common.TILE_SIZE;
  panel.focusPos(width / 2, height / 2, Math.min(panel.width / width, panel.height / height));
}

// Draw the floor.
function drawFloor() {
  const floorColor = parseHtmlColor("#CFA970");
  ctx.drawSolidRect(
    -Common.TILE_SIZE / 2,
    -Common.TILE_SIZE / 2,
    state.replay.map_size[0] * Common.TILE_SIZE,
    state.replay.map_size[1] * Common.TILE_SIZE,
    floorColor
  );
}

// Draw the walls, based on the adjacency map, and fill any holes.
function drawWalls() {
  // Construct wall adjacency map.
  var wallMap = new Grid(state.replay.map_size[0], state.replay.map_size[1]);
  for (const gridObject of state.replay.grid_objects) {
    const type = getAttr(gridObject, "type");
    const typeName = state.replay.object_types[type];
    if (typeName !== "wall") {
      continue;
    }
    const x = getAttr(gridObject, "c");
    const y = getAttr(gridObject, "r");
    wallMap.set(x, y, true);
  }

  // Draw the walls following the adjacency map.
  for (const gridObject of state.replay.grid_objects) {
    const type = getAttr(gridObject, "type");
    const typeName = state.replay.object_types[type];
    if (typeName !== "wall") {
      continue;
    }
    const x = getAttr(gridObject, "c");
    const y = getAttr(gridObject, "r");
    var suffix = "0";
    var n = false, w = false, e = false, s = false;
    if (wallMap.get(x, y - 1)) {
      n = true;
    }
    if (wallMap.get(x - 1, y)) {
      w = true;
    }
    if (wallMap.get(x, y + 1)) {
      s = true;
    }
    if (wallMap.get(x + 1, y)) {
      e = true;
    }
    if (n || w || e || s) {
      suffix = (n ? "n" : "") + (w ? "w" : "") + (s ? "s" : "") + (e ? "e" : "");
    }
    ctx.drawSprite('objects/wall.' + suffix + '.png', x * Common.TILE_SIZE, y * Common.TILE_SIZE);
  }

  // Draw the wall in-fill following the adjacency map.
  for (const gridObject of state.replay.grid_objects) {
    const type = getAttr(gridObject, "type");
    const typeName = state.replay.object_types[type];
    if (typeName !== "wall") {
      continue;
    }
    const x = getAttr(gridObject, "c");
    const y = getAttr(gridObject, "r");
    // If walls to E, S and SE is filled, draw a wall fill.
    var s = false, e = false, se = false;
    if (wallMap.get(x + 1, y)) {
      e = true;
    }
    if (wallMap.get(x, y + 1)) {
      s = true;
    }
    if (wallMap.get(x + 1, y + 1)) {
      se = true;
    }
    if (e && s && se) {
      ctx.drawSprite(
        'objects/wall.fill.png',
        x * Common.TILE_SIZE + Common.TILE_SIZE / 2,
        y * Common.TILE_SIZE + Common.TILE_SIZE / 2 - 42
      );
    }
  }
}

// Draw all objects on the map (that are not walls).
function drawObjects() {
  for (const gridObject of state.replay.grid_objects) {
    const type: number = getAttr(gridObject, "type");
    const typeName: string = state.replay.object_types[type];
    if (typeName === "wall") {
      // Walls are drawn in a different way.
      continue;
    }
    const x = getAttr(gridObject, "c")
    const y = getAttr(gridObject, "r")

    if (gridObject["agent_id"] !== undefined) {
      // Respect orientation of an object usually an agent.
      const orientation = getAttr(gridObject, "agent:orientation");
      var suffix = "";
      if (orientation == 0) {
        suffix = "n";
      } else if (orientation == 1) {
        suffix = "s";
      } else if (orientation == 2) {
        suffix = "w";
      } else if (orientation == 3) {
        suffix = "e";
      }

      const agent_id = getAttr(gridObject, "agent_id");

      ctx.drawSprite(
        "agents/agent." + suffix + ".png",
        x * Common.TILE_SIZE,
        y * Common.TILE_SIZE,
        colorFromId(agent_id)
      );
    } else {
      // Draw regular objects.

      // Draw the base layer.
      ctx.drawSprite(
        state.replay.object_images[type][0],
        x * Common.TILE_SIZE,
        y * Common.TILE_SIZE
      );

      // Draw the color layer.
      var colorIdx = getAttr(gridObject, "color");
      if (colorIdx >= 0 && colorIdx < Common.COLORS.length) {
        ctx.drawSprite(
          state.replay.object_images[type][2],
          x * Common.TILE_SIZE,
          y * Common.TILE_SIZE,
          Common.COLORS[colorIdx][1]
        );
      }

      // Draw the item layer.
      if (hasInventory(gridObject)) {
        ctx.drawSprite(
          state.replay.object_images[type][1],
          x * Common.TILE_SIZE,
          y * Common.TILE_SIZE
        );
      }
    }
  }
}

// Draw actions above the objects.
function drawActions() {
  for (const gridObject of state.replay.grid_objects) {
    const x = getAttr(gridObject, "c")
    const y = getAttr(gridObject, "r")

    // Do agent actions.
    if (gridObject["action"] !== undefined) {
      // Draw the action:
      const action = getAttr(gridObject, "action");
      const action_success = getAttr(gridObject, "action_success");
      if (action_success && action != null) {
        const action_name = state.replay.action_names[action[0]];
        const orientation = getAttr(gridObject, "agent:orientation");
        var rotation = 0;
        if (orientation == 0) {
          rotation = Math.PI / 2; // North
        } else if (orientation == 1) {
          rotation = -Math.PI / 2; // South
        } else if (orientation == 2) {
          rotation = Math.PI; // West
        } else if (orientation == 3) {
          rotation = 0; // East
        }
        if (action_name == "attack" && action[1] >= 1 && action[1] <= 9) {
          ctx.drawSprite(
            "actions/attack" + action[1] + ".png",
            x * Common.TILE_SIZE,
            y * Common.TILE_SIZE,
            [1, 1, 1, 1],
            1,
            rotation
          );
        } else if (action_name == "attack_nearest") {
          ctx.drawSprite(
            "actions/attack_nearest.png",
            x * Common.TILE_SIZE,
            y * Common.TILE_SIZE,
            [1, 1, 1, 1],
            1,
            rotation
          );
        } else if (action_name == "put_recipe_items") {
          ctx.drawSprite(
            "actions/put_recipe_items.png",
            x * Common.TILE_SIZE,
            y * Common.TILE_SIZE,
            [1, 1, 1, 1],
            1,
            rotation
          );
        } else if (action_name == "get_output") {
          ctx.drawSprite(
            "actions/get_output.png",
            x * Common.TILE_SIZE,
            y * Common.TILE_SIZE,
            [1, 1, 1, 1],
            1,
            rotation
          );
        } else if (action_name == "swap") {
          ctx.drawSprite(
            "actions/swap.png",
            x * Common.TILE_SIZE,
            y * Common.TILE_SIZE,
            [1, 1, 1, 1],
            1,
            rotation
          );
        }
      }
    }

    // Do building actions.
    if (getAttr(gridObject, "converting") > 0) {
      ctx.drawSprite(
        "actions/converting.png",
        x * Common.TILE_SIZE,
        y * Common.TILE_SIZE - 100,
        [1, 1, 1, 1],
        1,
        // Apply the gentle rotation.
        -state.step * 0.1
      );
    }

    // Do states
    if (getAttr(gridObject, "agent:frozen") > 0) {
      ctx.drawSprite(
        "agents/frozen.png",
        x * Common.TILE_SIZE,
        y * Common.TILE_SIZE,
      );
    }
  }
}

// Draw the object's inventory.
function drawInventory() {
  for (const gridObject of state.replay.grid_objects) {
    const x = getAttr(gridObject, "c")
    const y = getAttr(gridObject, "r")

    // Sum up the objects inventory, in case we need to condense it.
    let inventoryX = Common.INVENTORY_PADDING;
    let numItems = 0;
    for (const [key, [icon, color]] of state.replay.resource_inventory) {
      const num = getAttr(gridObject, key);
      numItems += num;
    }
    // Draw the actual inventory icons.
    let advanceX = Math.min(32, (Common.TILE_SIZE - Common.INVENTORY_PADDING * 2) / numItems);
    for (const [key, [icon, color]] of state.replay.resource_inventory) {
      const num = getAttr(gridObject, key);
      for (let i = 0; i < num; i++) {
        ctx.drawSprite(
          icon,
          x * Common.TILE_SIZE + inventoryX - Common.TILE_SIZE / 2,
          y * Common.TILE_SIZE - Common.TILE_SIZE / 2 + 16,
          color,
          1 / 8,
          0
        );
        inventoryX += advanceX;
      }
    }
  }
}

// Draw the rewards on the bottom of the object.
function drawRewards() {
  for (const gridObject of state.replay.grid_objects) {
    const x = getAttr(gridObject, "c")
    const y = getAttr(gridObject, "r")
    if (gridObject["total_reward"] !== undefined) {
      const totalReward = getAttr(gridObject, "total_reward");
      let rewardX = 0;
      let advanceX = Math.min(32, Common.TILE_SIZE / totalReward);
      for (let i = 0; i < totalReward; i++) {
        ctx.save()
        ctx.translate(
          x * Common.TILE_SIZE + rewardX - Common.TILE_SIZE / 2,
          y * Common.TILE_SIZE + Common.TILE_SIZE / 2 - 16
        );
        ctx.scale(1 / 8, 1 / 8);
        ctx.drawSprite("resources/reward.png", 0, 0);
        ctx.restore()
        rewardX += advanceX;
      }
    }
  }
}

// Draw the selection of the selected object.
function drawSelection() {
  if (state.selectedGridObject === null) {
    return;
  }

  const x = getAttr(state.selectedGridObject, "c")
  const y = getAttr(state.selectedGridObject, "r")
  ctx.drawSprite("selection.png", x * Common.TILE_SIZE, y * Common.TILE_SIZE);
}

// Draw the trajectory of the selected object, footprints or future arrow.
function drawTrajectory() {
  if (state.selectedGridObject === null) {
    return;
  }
  if (state.selectedGridObject.c.length > 0 || state.selectedGridObject.r.length > 0) {

    // Draw both past and future trajectories.
    for (let i = 1; i < state.replay.max_steps; i++) {
      const cx0 = getAttr(state.selectedGridObject, "c", i - 1);
      const cy0 = getAttr(state.selectedGridObject, "r", i - 1);
      const cx1 = getAttr(state.selectedGridObject, "c", i);
      const cy1 = getAttr(state.selectedGridObject, "r", i);
      if (cx0 !== cx1 || cy0 !== cy1) {
        const a = 1 - Math.abs(i - state.step) / 200;
        if (a > 0) {
          let color = [0, 0, 0, a];
          let image = "";
          if (state.step >= i) {
            // Past trajectory is black.
            color = [0, 0, 0, a];
            if (state.selectedGridObject.agent_id !== undefined) {
              image = "agents/footprints.png";
            } else {
              image = "agents/past_arrow.png";
            }
          } else {
            // Future trajectory is white.
            color = [a, a, a, a];
            if (state.selectedGridObject.agent_id !== undefined) {
              image = "agents/path.png";
            } else {
              image = "agents/future_arrow.png";
            }
          }

          if (cx1 > cx0) { // east
            ctx.drawSprite(
              image,
              cx0 * Common.TILE_SIZE,
              cy0 * Common.TILE_SIZE + 60,
              color,
              1,
              0
            );
          } else if (cx1 < cx0) { // west
            ctx.drawSprite(
              image,
              cx0 * Common.TILE_SIZE,
              cy0 * Common.TILE_SIZE + 60,
              color,
              1,
              Math.PI
            );
          } else if (cy1 > cy0) { // south
            ctx.drawSprite(
              image,
              cx0 * Common.TILE_SIZE,
              cy0 * Common.TILE_SIZE + 60,
              color,
              1,
              -Math.PI / 2
            );
          } else if (cy1 < cy0) { // north
            ctx.drawSprite(
              image,
              cx0 * Common.TILE_SIZE,
              cy0 * Common.TILE_SIZE + 60,
              color,
              1,
              Math.PI / 2
            );
          }
        }
      }
    }
  }
}

// Draw the thought bubbles of the selected agent.
function drawThoughtBubbles() {
  // The idea behind thought bubbles is to show what the agent is thinking.
  // We don't have this directly from the policy yet,
  // so the next best thing is to show future "key action".
  // It should be a good proxy for what the agent is thinking about.
  if (state.selectedGridObject != null && state.selectedGridObject.agent_id != null) {
    // We need to find a key action in the future.
    // A key action is a successful action that is not a noop, rotate or move.
    // Must not be more then 20 steps in the future.
    var keyAction = null;
    var keyActionStep = null;
    for (var actionStep = state.step; actionStep < state.replay.max_steps && actionStep < state.step + 20; actionStep++) {
      const action = getAttr(state.selectedGridObject, "action", actionStep);
      if (action == null || action[0] == null || action[1] == null) {
        continue;
      }
      const actionName = state.replay.action_names[action[0]];
      const actionSuccess = getAttr(state.selectedGridObject, "action_success", actionStep);
      if (actionName == "noop" || actionName == "rotate" || actionName == "move") {
        continue;
      }
      if (actionSuccess) {
        keyAction = action;
        keyActionStep = actionStep;
        break;
      }
    }

    if (keyAction != null) {
      // We have a key action, draw the thought bubble.
      // Draw the key action icon with gained or lost resources.
      const x = getAttr(state.selectedGridObject, "c");
      const y = getAttr(state.selectedGridObject, "r");
      if (state.step == keyActionStep) {
        ctx.drawSprite(
          "actions/thoughts_lightning.png",
          x * Common.TILE_SIZE + Common.TILE_SIZE / 2,
          y * Common.TILE_SIZE - Common.TILE_SIZE / 2
        );
      } else {
        ctx.drawSprite(
          "actions/thoughts.png",
          x * Common.TILE_SIZE + Common.TILE_SIZE / 2,
          y * Common.TILE_SIZE - Common.TILE_SIZE / 2
        );
      }
      // Draw the action icon.
      var iconName = "actions/icons/" + state.replay.action_names[keyAction[0]] + ".png";
      if (ctx.hasImage(iconName)) {
        ctx.drawSprite(
          iconName,
          x * Common.TILE_SIZE + Common.TILE_SIZE / 2,
          y * Common.TILE_SIZE - Common.TILE_SIZE / 2,
          [1, 1, 1, 1],
          1 / 4,
          0
        );
      } else {
        ctx.drawSprite(
          "actions/icons/unknown.png",
          x * Common.TILE_SIZE + Common.TILE_SIZE / 2,
          y * Common.TILE_SIZE - Common.TILE_SIZE / 2,
          [1, 1, 1, 1],
          1 / 4,
          0
        );
      }

      // Draw resources lost on the left and gained on the right.
      for (const [key, [image, color]] of state.replay.resource_inventory) {
        const prevResources = getAttr(state.selectedGridObject, key, actionStep - 1);
        const nextResources = getAttr(state.selectedGridObject, key, actionStep);
        const gained = nextResources - prevResources;
        var resourceX = x * Common.TILE_SIZE + Common.TILE_SIZE / 2;
        var resourceY = y * Common.TILE_SIZE - Common.TILE_SIZE / 2;
        if (gained > 0) {
          resourceX += 32;
        } else {
          resourceX -= 32;
        }
        for (let i = 0; i < Math.abs(gained); i++) {
          ctx.drawSprite(
            image,
            resourceX,
            resourceY,
            color,
            1 / 8,
            0
          );
          if (gained > 0) {
            resourceX += 8;
          } else {
            resourceX -= 8;
          }
        }
      }
    }
  }
}

// Draw the visibility map either agent view ranges or fog of war.
function drawVisibility() {

  if (state.showViewRanges || state.showFogOfWar) {
    // Compute the visibility map, each agent contributes to the visibility map.
    const visibilityMap = new Grid(state.replay.map_size[0], state.replay.map_size[1]);

    // Update the visibility map for a grid object.
    function updateVisibilityMap(gridObject: any) {
      const x = getAttr(gridObject, "c");
      const y = getAttr(gridObject, "r");
      var visionSize = Math.floor(getAttr(
        gridObject,
        "agent:vision_size",
        state.step,
        Common.DEFAULT_VISION_SIZE
      ) / 2);
      for (let dx = -visionSize; dx <= visionSize; dx++) {
        for (let dy = -visionSize; dy <= visionSize; dy++) {
          visibilityMap.set(
            x + dx,
            y + dy,
            true
          );
        }
      }
    }

    if (state.selectedGridObject !== null && state.selectedGridObject.agent_id !== undefined) {
      // When there is a selected grid object only update its visibility.
      updateVisibilityMap(state.selectedGridObject);
    } else {
      // When there is no selected grid object update the visibility map for all agents.
      for (const gridObject of state.replay.grid_objects) {
        const type = getAttr(gridObject, "type");
        const typeName = state.replay.object_types[type];
        if (typeName == "agent") {
          updateVisibilityMap(gridObject);
        }
      }
    }

    var color = [0, 0, 0, 0.25];
    if (state.showFogOfWar) {
      color = [0, 0, 0, 1];
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
          );
        }
      }
    }
  }
}

// Draw the grid.
function drawGrid() {
  if (state.showGrid) {
    for (let x = 0; x < state.replay.map_size[0]; x++) {
      for (let y = 0; y < state.replay.map_size[1]; y++) {
        ctx.drawSprite('objects/grid.png', x * Common.TILE_SIZE, y * Common.TILE_SIZE);
      }
    }
  }
}


function attackGrid(orientation: number, idx: number) {
  // Given an orientation and an index, return the grid position.

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
    return ((a % b) + b) % b;
  }

  // Integer division.
  function div(a: number, b: number) {
    return Math.floor(a / b);
  }
  const i = idx - 1;
  let dx, dy;
  if (orientation === 0) {
      dx = mod(i, 3) - 1;
      dy = -div(i, 3) - 1;
  } else if (orientation === 1) {
      dx = -mod(i, 3) + 1;
      dy = div(i, 3) + 1;
  } else if (orientation === 2) {
      dx = -div(i, 3) - 1;
      dy = -mod(i, 3) + 1;
  } else if (orientation === 3) {
      dx = div(i, 3) + 1;
      dy = mod(i, 3) - 1;
  }
  return [dx, dy];
}

function drawAttackMode() {
  // Draw the attack mode.

  // We might be clicking on the map to attack something.
  var gridMousePos: Vec2f | null = null;
  if (ui.mouseUp && ui.mouseTarget == "worldmap-panel" && state.showAttackMode) {
    state.showAttackMode = false;
    const localMousePos = ui.mapPanel.transformPoint(ui.mousePos);
    if (localMousePos != null) {
      gridMousePos = new Vec2f(
        Math.round(localMousePos.x() / Common.TILE_SIZE),
        Math.round(localMousePos.y() / Common.TILE_SIZE)
      );
    }
  }

  // Draw a selection of 3x3 grid of targets in the direction of the selected agent.
  if (state.selectedGridObject !== null && state.selectedGridObject.agent_id !== undefined) {
    const x = getAttr(state.selectedGridObject, "c");
    const y = getAttr(state.selectedGridObject, "r");
    const orientation = getAttr(state.selectedGridObject, "agent:orientation");

    // Draw a 3x3 grid of targets in the direction of the selected agent.
    for(let attackIndex = 1; attackIndex <= 9; attackIndex++) {
        const [dx, dy] = attackGrid(orientation, attackIndex);
        const targetX = x + dx;
        const targetY = y + dy;
        ctx.drawSprite('target.png', targetX * Common.TILE_SIZE, targetY * Common.TILE_SIZE);
        if (gridMousePos != null && targetX == gridMousePos.x() && targetY == gridMousePos.y()) {
          // Check if we are clicking this specific tile.
          console.log("Attack mode clicked on:", targetX, targetY);
          sendAction("attack", attackIndex)
        }
    }
  }
}

export function drawMap(panel: PanelInfo) {
  if (state.replay === null || ctx === null || ctx.ready === false) {
    return;
  }

  // Handle mouse events for the map panel.
  if (ui.mouseTarget == "worldmap-panel" && !state.showAttackMode) {
    if (ui.mouseDoubleClick) {
      // Toggle followSelection on double-click
      console.log("Map double click - following selection");
      setFollowSelection(true);
      panel.zoomLevel = Common.DEFAULT_ZOOM_LEVEL;
      ui.tracePanel.zoomLevel = Common.DEFAULT_TRACE_ZOOM_LEVEL;
    } else if (ui.mouseClick) {
      // Map click - likely a drag/pan
      console.log("Map click - clearing follow selection");
      setFollowSelection(false);
    } else if (ui.mouseUp &&
      ui.mouseDownPos.sub(ui.mousePos).length() < 10
    ) {
      // Check if we are clicking on an object.
      console.log("Map up without dragging - selecting object");
      const localMousePos = panel.transformPoint(ui.mousePos);
      if (localMousePos != null) {
        const gridMousePos = new Vec2f(
          Math.round(localMousePos.x() / Common.TILE_SIZE),
          Math.round(localMousePos.y() / Common.TILE_SIZE)
        );
        const gridObject = state.replay.grid_objects.find((obj: any) => {
          const x: number = getAttr(obj, "c");
          const y: number = getAttr(obj, "r");
          return x === gridMousePos.x() && y === gridMousePos.y();
        });
        if (gridObject !== undefined) {
          state.selectedGridObject = gridObject;
          console.log("Selected object on the map:", state.selectedGridObject);
          if (state.selectedGridObject.agent_id !== undefined) {
            // If selecting an agent, focus the trace panel on the agent.
            ui.tracePanel.focusPos(
              state.step * Common.TRACE_WIDTH + Common.TRACE_WIDTH / 2,
              getAttr(state.selectedGridObject, "agent_id") * Common.TRACE_HEIGHT + Common.TRACE_HEIGHT / 2,
              Common.DEFAULT_TRACE_ZOOM_LEVEL
            );
          }
        }
      }
    }
  }

  // If we're following a selection, center the map on it
  if (state.followSelection && state.selectedGridObject !== null) {
    const x = getAttr(state.selectedGridObject, "c");
    const y = getAttr(state.selectedGridObject, "r");
    panel.panPos = new Vec2f(-x * Common.TILE_SIZE, -y * Common.TILE_SIZE);
  }

  ctx.save();
  ctx.setScissorRect(panel.x, panel.y, panel.width, panel.height);

  ctx.translate(panel.x + panel.width / 2, panel.y + panel.height / 2);
  ctx.scale(panel.zoomLevel, panel.zoomLevel);
  ctx.translate(panel.panPos.x(), panel.panPos.y());

  drawFloor();
  drawWalls();
  drawTrajectory();
  drawObjects();
  drawSelection();
  drawActions();
  drawInventory();
  drawRewards();
  drawVisibility();
  drawGrid();
  drawThoughtBubbles();
  if (state.showAttackMode) {
    drawAttackMode();
  }

  ctx.restore();
}

// Updates the readout of the selected object or replay info.
export function updateReadout() {
  var readout = ""
  if (state.selectedGridObject !== null) {
    if (state.followSelection) {
      readout += "FOLLOWING SELECTION (double-click to unfollow)\n\n";
    }
    for (const key in state.selectedGridObject) {
      var value = getAttr(state.selectedGridObject, key);
      if (key == "type") {
        value = state.replay.object_types[value] + " (" + value + ")";
      } else if (key == "color" && value >= 0 && value < Common.COLORS.length) {
        value = Common.COLORS[value][0] + " (" + value + ")";
      }
      readout += key + ": " + value + "\n";
    }
  } else {
    readout += "Step: " + state.step + "\n";
    readout += "Map size: " + state.replay.map_size[0] + "x" + state.replay.map_size[1] + "\n";
    readout += "Num agents: " + state.replay.num_agents + "\n";
    readout += "Max steps: " + state.replay.max_steps + "\n";

    var objectTypeCounts = new Map<string, number>();
    for (const gridObject of state.replay.grid_objects) {
      const type = getAttr(gridObject, "type");
      const typeName = state.replay.object_types[type];
      objectTypeCounts.set(typeName, (objectTypeCounts.get(typeName) || 0) + 1);
    }
    for (const [key, value] of objectTypeCounts.entries()) {
      readout += key + " count: " + value + "\n";
    }
  }
  let info = find("#info-panel .info")
  if (info !== null) {
    info.innerHTML = readout;
  }
}
