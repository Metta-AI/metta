// Info panels are used to display information about the current state of objects.
// They appear as hovers when you hover over an agent or click on a grid object.
// They can be pinned to stack on screen, a line is drawn to the object,
// when they are on pined on screen. They can also be moved around in this pinned mode.

import { find, findIn, removeChildren } from "./htmlutils.js";
import { state, ui } from "./common.js";
import { getAttr } from "./replay.js";
import * as Common from "./common.js";
import { Vec2f } from "./vector_math.js";

export class InfoPanel {
  private object: any;
  private panel: HTMLElement;

  constructor(object: any) {
    this.object = object;
    this.panel = document.createElement("div");
  }

}

var infoPanelTemplate = find(".infopanel") as HTMLElement;

infoPanelTemplate.remove();

var hoverPanel = infoPanelTemplate.cloneNode(true) as HTMLElement;
document.body.appendChild(hoverPanel);

export function showInfoPanel(object: any) {
  // Show the info panel.
}

export function updateReadout() {
  // Update the readout.
  if (state.selectedGridObject !== null) {

    var pin = findIn(infoPanelTemplate, ".pin-info-panel").cloneNode(true) as HTMLElement;
    var top = findIn(hoverPanel, ".top");
    var paramTemplate = findIn(infoPanelTemplate, ".param");
    var inventory = findIn(hoverPanel, ".inventory");
    var itemTemplate = findIn(infoPanelTemplate, ".item");

    removeChildren(top);
    top.appendChild(pin);
    removeChildren(inventory);

    for (const key in state.selectedGridObject) {
      let value = getAttr(state.selectedGridObject, key);
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

    let panelRect = hoverPanel.getBoundingClientRect();

    let x = getAttr(state.selectedGridObject, "c") * Common.TILE_SIZE;
    let y = getAttr(state.selectedGridObject, "r") * Common.TILE_SIZE;

    let uiPoint = ui.mapPanel.transformInner(new Vec2f(x, y - Common.TILE_SIZE / 2));

    // Put it in the center above the object.
    hoverPanel.style.left = uiPoint.x() - panelRect.width / 2 + "px";
    hoverPanel.style.top = uiPoint.y() - panelRect.height + "px";


  }
}
