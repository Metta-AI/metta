import { find, finds, removeChildren, walkUpAttribute } from "./htmlutils.js";
import { state, setFollowSelection } from "./common.js";
import { getAttr } from "./replay.js";
import { updateSelection } from "./main.js";

const agentPanel = find("#agent-panel");
const agentTable = find("#agent-table");
const header = find("#agent-table .header");
const fieldHeaderTemplate = finds("#agent-table .field-header")[0];
const rowTemplate = finds("#agent-table .row")[0];
const cellTemplate = finds("#agent-table .cell")[0];

export function initAgentTable() {
  removeChildren(rowTemplate);

  agentTable.addEventListener("click", (event) => {
    // Get the element that was actually clicked
    let target = event.target as HTMLElement;
    let agentId = walkUpAttribute(target, "data-agent-id");
    if (agentId != "") {
      for (let i = 0; i < state.replay.grid_objects.length; i++) {
        let gridObject = state.replay.grid_objects[i];
        if (getAttr(gridObject, "agent_id") == agentId) {
          updateSelection(gridObject, true)
          break;
        }
      }
    }

    let dataField = walkUpAttribute(target, "data-field");
    if (dataField != "") {
      if (dataField == mainSort) {
        sortDirection = -sortDirection;
      } else {
        mainSort = dataField;
        sortDirection = -1;
      }
      updateAgentTable();
    }
  });

  // agentPanel.addEventListener("mousedown", (event) => {
  //   if (event.target == agentPanel) {
  //     dragging = true;
  //     dragStartX = event.clientX;
  //     dragStartY = event.clientY;
  //     console.log("start dragging");
  //   }
  // });

  // agentPanel.addEventListener("mousemove", (event) => {
  //   if (dragging) {
  //     let x = event.clientX;
  //     let y = event.clientY;
  //     let left = x - dragStartX;
  //     let top = y - dragStartY;
  //     agentPanel.style.left = left + "px";
  //     agentPanel.style.top = top + "px";
  //     console.log("dragging");
  //   }
  // });

  // agentPanel.addEventListener("mouseup", (event) => {
  //   dragging = false;
  //   console.log("stop dragging");
  // });
}

// Capitalize the first letter of every word in a string.
// Example: "hello world" -> "Hello World"
function capitalize(str: string) {
  return str.split(" ").map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(" ");
}

var dragging = false;
var dragStartX = 0;
var dragStartY = 0;
var mainSort = "total_reward";
var sortDirection = -1;
var sortStepFinal = false;

export function updateAgentTable() {
  removeChildren(agentTable);
  agentTable.appendChild(header);

  removeChildren(header);
  let headers = [
    ["ID", "agent_id"],
    ["Reward", "total_reward"],
  ];

  for (let key of state.replay.all_keys) {
    if (key.startsWith("agent:") || key.startsWith("inv:")) {
      let name = capitalize(key.replace("agent:", "").replace("inv:", "").replace(".", " "));
      headers.push([name, key]);
    }
  }

  // Create the header cells.
  for (let i = 0; i < headers.length; i++) {
    let headerCell = fieldHeaderTemplate.cloneNode(true) as HTMLElement;
    headerCell.children[1].textContent = headers[i][0];
    if (headers[i][1] == mainSort) {
      (headerCell.children[0] as HTMLElement).style.opacity = "1";
      if (sortDirection == 1) {
        (headerCell.children[0] as HTMLElement).setAttribute("src", "data/ui/sort-down.png");
      } else {
        (headerCell.children[0] as HTMLElement).setAttribute("src", "data/ui/sort-up.png");
      }
    } else {
      (headerCell.children[0] as HTMLElement).style.opacity = "0";
    }
    headerCell.setAttribute("data-field", headers[i][1]);
    header.appendChild(headerCell);
  }

  let list = state.replay.agents.slice();
  let agents = list.sort((a: any, b: any) => {
    var aValue, bValue: number;
    if (sortStepFinal) {
      // Uses the final step for the sort.
      aValue = getAttr(a, mainSort, state.replay.max_step);
      bValue = getAttr(b, mainSort, state.replay.max_step);
    } else {
      // Uses the current step for the sort.
      aValue = getAttr(a, mainSort)
      bValue = getAttr(b, mainSort)
    }
    // Sort direction adjustment.
    if (sortDirection == 1) {
      return aValue - bValue;
    } else {
      return bValue - aValue;
    }
  });

  for (let i = 0; i < agents.length; i++) {
    let agent = agents[i];
    if (agent != null) {
      let row = rowTemplate.cloneNode(true) as HTMLElement;
      row.setAttribute("data-agent-id", getAttr(agent, "agent_id").toString());
      if (state.selectedGridObject != null && getAttr(agent, "agent_id") == getAttr(state.selectedGridObject, "agent_id")) {
        row.classList.add("selected");
      }
      for (let i = 0; i < headers.length; i++) {
        let cell = cellTemplate.cloneNode(true) as HTMLElement;
        let value = getAttr(agent, headers[i][1])
        let valueStr = value.toString();
        if (valueStr.includes(".")) {
          valueStr = value.toFixed(3);
        }
        cell.children[0].textContent = valueStr;
        row.appendChild(cell);
      }
      agentTable.appendChild(row);
    }
  }
}
