// Info panels are used to display information about the current state of objects.

// Lower level hover rules:
// * You need to hover over the object for 1 second for the info panel to show.
// * You can hover off the object, the panel will stay visible for 1 second.
// * If you hover over the panel, the panel will stay visible as long as mouse is over the panel.
// * If you drag the panel, it will detach and stay on screen.
//   * It will be only closed by clicking on the X.
//   * It will loose its hover stem on the bottom when it detached mode.

import { find, findIn, onEvent, removeChildren, findAttr } from "./htmlutils.js";
import { state, ui } from "./common.js";
import { getAttr } from "./replay.js";
import * as Common from "./common.js";
import { Vec2f } from "./vector_math.js";

/** An info panel. */
export class InfoPanel {
  public object: any;
  public div: HTMLElement;

  constructor(object: any) {
    this.object = object;
    this.div = document.createElement("div");
  }

  public update() {
    updateDom(this.div, this.object);
  }
}

onEvent("click", ".infopanel .close", (target: HTMLElement, e: Event) => {
  let panel = target.parentElement as HTMLElement;
  panel.remove();
  ui.infoPanels = ui.infoPanels.filter(p => p.div !== panel);
})

var infoPanelTemplate = find(".infopanel") as HTMLElement;
infoPanelTemplate.remove();

var hoverPanel = infoPanelTemplate.cloneNode(true) as HTMLElement;
document.body.appendChild(hoverPanel);
findIn(hoverPanel, ".actions").classList.add("hidden");
hoverPanel.classList.add("hidden");

hoverPanel.addEventListener("mousedown", (e: MouseEvent) => {
  // Create a new info panel.
  let panel = new InfoPanel(ui.delayedHoverObject);
  panel.div = infoPanelTemplate.cloneNode(true) as HTMLElement;
  panel.div.classList.add("draggable");
  let tip = findIn(panel.div, ".tip");
  tip.remove();
  document.body.appendChild(panel.div);
  updateDom(panel.div, panel.object);
  panel.div.style.top = hoverPanel.style.top;
  panel.div.style.left = hoverPanel.style.left;

  // Show the actions buttons (memory, etc.) if the object is an agent
  // and if the websocket is connected.
  var actions = findIn(panel.div, ".actions");
  if (state.ws != null && panel.object.hasOwnProperty("agent_id")) {
    actions.classList.remove("hidden");
  } else {
    actions.classList.add("hidden");
  }

  ui.dragHtml = panel.div;
  // Compute mouse position relative to the panel.
  let rect = panel.div.getBoundingClientRect();
  ui.dragOffset = new Vec2f(e.clientX - rect.left, e.clientY - rect.top);
  ui.dragging = "info-panel";
  ui.infoPanels.push(panel);

  // Hide the old hover panel.
  // THe new info panel should be identical to the old hover panel,
  // so that the user sees no difference.
  hoverPanel.classList.add("hidden");
  ui.hoverObject = null;
  ui.delayedHoverObject = null;
  e.stopPropagation();
});

/** Update the hover panel, visibility and position, and dom tree. */
export function updateHoverPanel(object: any) {
  if (object !== null && object !== undefined) {
    updateDom(hoverPanel, object);
    hoverPanel.classList.remove("hidden");

    let panelRect = hoverPanel.getBoundingClientRect();

    let x = getAttr(object, "c") * Common.TILE_SIZE;
    let y = getAttr(object, "r") * Common.TILE_SIZE;

    let uiPoint = ui.mapPanel.transformInner(new Vec2f(x, y - Common.TILE_SIZE / 2));

    // Put it in the center above the object.
    hoverPanel.style.left = uiPoint.x() - panelRect.width / 2 + "px";
    hoverPanel.style.top = uiPoint.y() - panelRect.height + "px";
  } else {
    hoverPanel.classList.add("hidden");
  }
  findIn(hoverPanel, ".close").classList.add("hidden");
}

/** Update the dom tree of the info panel. */
function updateDom(htmlPanel: HTMLElement, object: any) {
  // Update the readout.
  htmlPanel.setAttribute("data-object-id", getAttr(object, "id"));
  htmlPanel.setAttribute("data-agent-id", getAttr(object, "agent_id"));

  var params = findIn(htmlPanel, ".params");
  var paramTemplate = findIn(infoPanelTemplate, ".param");
  var inventory = findIn(htmlPanel, ".inventory");
  var itemTemplate = findIn(infoPanelTemplate, ".item");
  let actions = findIn(hoverPanel, ".actions");

  removeChildren(params);
  //top.appendChild(pin);
  removeChildren(inventory);

  for (const key in object) {
    let value = getAttr(object, key);
    if ((key.startsWith("inv:") || key.startsWith("agent:inv:")) && value > 0) {
      var item = itemTemplate.cloneNode(true) as HTMLElement;
      item.querySelector(".amount")!.textContent = value;
      let resource = key.replace("inv:", "").replace("agent:", "");
      item.querySelector(".icon")!.setAttribute("src", "data/resources/" + resource + ".png");
      inventory.appendChild(item)
    } else {
      if (key == "type") {
        value = state.replay.object_types[value];
      } else if (key == "agent:color" && value >= 0 && value < Common.COLORS.length) {
        value = Common.COLORS[value][0];
      } else if (["group", "total_reward", "agent_id"].includes(key)) {
        // if value is a float and not an integer, round it to 3 decimal places
        if (typeof value === "number" && !Number.isInteger(value)) {
          value = value.toFixed(3);
        }
      } else {
        continue;
      }
      var param = paramTemplate.cloneNode(true) as HTMLElement;
      param.querySelector(".name")!.textContent = key;
      param.querySelector(".value")!.textContent = value;
      params.appendChild(param);
    }
  }
}

/** Updates the readout of the selected object or replay info. */
export function updateReadout() {
  var readout = ""
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
  let info = find("#info-panel .info")
  if (info !== null) {
    info.innerHTML = readout;
  }
}
