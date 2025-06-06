// Info panels are used to display information about the current state of objects.
// They appear as hovers when you hover over an agent or click on a grid object.
// They can be pinned to stack on screen, a line is drawn to the object,
// when they are on pined on screen. They can also be moved around in this pinned mode.

// Lower level hover rules:
// * You need to hover over the object for 1 second for the info panel to show.
// * You can hover off the object, the panel will stay visible for 1 second.
// * If you hover over the panel, the panel will stay visible as long as mouse is over the panel.
// * If you drag the panel, it will detach and stay on screen.
//   * It will be only closed by clicking on the X.
//   * It will loose its hover stem on the bottom when it detached mode.

import { find, findIn, removeChildren } from "./htmlutils.js";
import { state, ui } from "./common.js";
import { getAttr } from "./replay.js";
import * as Common from "./common.js";
import { Vec2f } from "./vector_math.js";

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

var infoPanelTemplate = find(".infopanel") as HTMLElement;

infoPanelTemplate.remove();

var hoverPanel = infoPanelTemplate.cloneNode(true) as HTMLElement;
document.body.appendChild(hoverPanel);

hoverPanel.addEventListener("mousedown", (e: MouseEvent) => {
  console.log("Info panel clicked");

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

export function showInfoPanel(object: any) {
  // Show the info panel.
}

export function updateReadout(object: any) {
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
}

function updateDom(htmlPanel: HTMLElement, object: any) {
  // Update the readout.
  var top = findIn(htmlPanel, ".top");
  var paramTemplate = findIn(infoPanelTemplate, ".param");
  var inventory = findIn(htmlPanel, ".inventory");
  var itemTemplate = findIn(infoPanelTemplate, ".item");

  removeChildren(top);
  //top.appendChild(pin);
  removeChildren(inventory);

  for (const key in object) {
    let value = getAttr(object, key);
    if (key.startsWith("inv:") && value > 0) {
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
      top.appendChild(param);
    }
  }
}
