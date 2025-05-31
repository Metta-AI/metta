import { find, finds, removeChildren, walkUpAttribute, onEvent, showMenu, hideMenu } from "./htmlutils.js";
import { state, setFollowSelection, html } from "./common.js";
import { getAttr } from "./replay.js";
import { updateSelection } from "./main.js";

class ColumnDefinition {
  name: string;
  field: string;
  isFinal: boolean;
  sortDirection: number;

  constructor(name: string, field: string, isFinal: boolean, sortDirection: number = 0) {
    this.name = name;
    this.field = field;
    this.isFinal = isFinal;
    this.sortDirection = sortDirection;
  }
}

var columns = [
  new ColumnDefinition("Agent Id", "agent_id", false),
  new ColumnDefinition("Total Reward", "total_reward", false),
  new ColumnDefinition("Final Total Reward", "total_reward", true),

  new ColumnDefinition("Red Ore", "ore.red", false),
  new ColumnDefinition("Blue Ore", "ore.blue", false),
  new ColumnDefinition("Green Ore", "ore.green", false),

  new ColumnDefinition("Red Battery", "battery.red", false),
  new ColumnDefinition("Blue Battery", "battery.blue", false),
  new ColumnDefinition("Green Battery", "battery.green", false),
];
var mainSort: ColumnDefinition = columns[1];

// Capitalize the first letter of every word in a string.
// Example: "hello world" -> "Hello World"
function capitalize(str: string) {
  return str.split(" ").map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(" ");
}

const agentTable = find("#agent-panel .table");
const table = find("#agent-panel .table");
const columnTemplate = finds("#agent-panel .table .column")[0];
const headerCellTemplate = finds("#agent-panel .table .header-cell")[0];
const dataCellTemplate = finds("#agent-panel .table .data-cell")[0];
const newColumnTemplate = finds("#agent-panel .table .new-column")[0];
const newColumnHeaderCell = finds("#agent-panel .table .new-column .header-cell")[0];
const newColumnDataCell = finds("#agent-panel .table .new-column .data-cell")[0];
const columnMenu = find("#column-menu");
const newColumnDropdown = find("#new-column-dropdown");

export function initAgentTable() {
  // removeChildren(rowTemplate);

  columnMenu.classList.add("hidden");
  newColumnDropdown.classList.add("hidden");

  // agentTable.addEventListener("click", (event) => {
  //   // Get the element that was actually clicked
  //   let target = event.target as HTMLElement;
  //   let agentId = walkUpAttribute(target, "data-agent-id");
  //   if (agentId != "") {
  //     for (let i = 0; i < state.replay.grid_objects.length; i++) {
  //       let gridObject = state.replay.grid_objects[i];
  //       if (getAttr(gridObject, "agent_id") == agentId) {
  //         updateSelection(gridObject, true)
  //         break;
  //       }
  //     }
  //   }

  //   // let dataField = walkUpAttribute(target, "data-field");
  //   // if (dataField != "") {
  //   //   if (dataField == mainSort.field) {
  //   //     sortDirection = -sortDirection;
  //   //   } else {
  //   //     for (let header of headers) {
  //   //       if (header.field == dataField) {
  //   //         mainSort = header;
  //   //         break;
  //   //       }
  //   //     }
  //   //     sortDirection = -1;
  //   //   }
  //   //   updateAgentTable();
  //   // }
  // });



  onEvent("click", "#agent-panel .header-cell .dropdown", (target: HTMLElement, e: Event) => {
    // let event = e as MouseEvent;
    // console.log("Header cell clicked", event);
    // let columnMenu = find("#column-menu");
    // columnMenu.classList.remove("hidden");
    // // Get location of the target.
    // let rect = target.getBoundingClientRect();
    // columnMenu.style.left = rect.left + "px";
    // columnMenu.style.top = (rect.bottom + 2) + "px";
    // html.scrim.classList.remove("hidden");
    // html.scrimTarget = columnMenu;
    let columnMenu = find("#column-menu");
    showMenu(target, columnMenu);
  });

  onEvent("click", "#column-menu .sort-up", (target: HTMLElement, e: Event) => {
    console.log("Sort up clicked");
  });

  onEvent("click", "#column-menu .sort-down", (target: HTMLElement, e: Event) => {
    console.log("Sort up clicked");
  });

  onEvent("click", "#column-menu .move-left", (target: HTMLElement, e: Event) => {
    console.log("Move left clicked");
  });

  onEvent("click", "#column-menu .move-right", (target: HTMLElement, e: Event) => {
    console.log("Move right clicked");
  });

  onEvent("click", "#column-menu .hide-column", (target: HTMLElement, e: Event) => {
    console.log("Hide column clicked");
  });

  onEvent("click", "#agent-panel .data-cell", (target: HTMLElement, e: Event) => {
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
  });

  onEvent("click", "#new-column-input", (target: HTMLElement, e: Event) => {
    // let event = e as MouseEvent;
    // console.log("New column input clicked", event);
    // let newColumnDropdown = find("#new-column-dropdown");
    // newColumnDropdown.classList.remove("hidden");
    // // Get location of the target.
    // let rect = target.getBoundingClientRect();
    // newColumnDropdown.style.left = rect.left + "px";
    // newColumnDropdown.style.top = (rect.bottom + 2) + "px";
    // html.scrim.classList.remove("hidden");
    // html.scrimTarget = newColumnDropdown;
    let newColumnDropdown = find("#new-column-dropdown");
    showMenu(target, newColumnDropdown);
  });

  removeChildren(table);
  removeChildren(columnTemplate);
  removeChildren(newColumnTemplate);
}



export function updateAgentTable() {
  removeChildren(agentTable);



  // // for (let key of state.replay.all_keys) {
  // //   if (key.startsWith("agent:") || key.startsWith("inv:")) {
  // //     let name = capitalize(key.replace("agent:", "").replace("inv:", "").replace(".", " "));
  // //     headers.push([name, key]);
  // //   }
  // // }


  // // Create the header cells.
  // for (let i = 0; i < headers.length; i++) {
  //   let headerCell = fieldHeaderTemplate.cloneNode(true) as HTMLElement;
  //   headerCell.children[1].textContent = headers[i].name;
  //   if (headers[i].field == mainSort.field) {
  //     (headerCell.children[0] as HTMLElement).style.opacity = "1";
  //     if (sortDirection == 1) {
  //       (headerCell.children[0] as HTMLElement).setAttribute("src", "data/ui/sort-down.png");
  //     } else {
  //       (headerCell.children[0] as HTMLElement).setAttribute("src", "data/ui/sort-up.png");
  //     }
  //   } else {
  //     (headerCell.children[0] as HTMLElement).style.opacity = "0";
  //   }
  //   headerCell.setAttribute("data-field", headers[i].field);
  //   header.appendChild(headerCell);
  // }

  // header.appendChild(newFieldTemplate.cloneNode(true) as HTMLElement);

  let list = state.replay.agents.slice();
  let agents = list.sort((a: any, b: any) => {
    var aValue, bValue: number;
    if (mainSort.isFinal) {
      // Uses the final step for the sort.
      aValue = getAttr(a, mainSort.field, state.replay.max_step);
      bValue = getAttr(b, mainSort.field, state.replay.max_step);
    } else {
      // Uses the current step for the sort.
      aValue = getAttr(a, mainSort.field)
      bValue = getAttr(b, mainSort.field)
    }
    // Sort direction adjustment.
    if (mainSort.sortDirection == 1) {
      return aValue - bValue;
    } else {
      return bValue - aValue;
    }
  });

  // for (let i = 0; i < agents.length; i++) {
  //   let agent = agents[i];
  //   if (agent != null) {
  //     let row = rowTemplate.cloneNode(true) as HTMLElement;
  //     row.setAttribute("data-agent-id", getAttr(agent, "agent_id").toString());
  //     if (state.selectedGridObject != null && getAttr(agent, "agent_id") == getAttr(state.selectedGridObject, "agent_id")) {
  //       row.classList.add("selected");
  //     }
  //     for (let i = 0; i < headers.length; i++) {
  //       let cell = cellTemplate.cloneNode(true) as HTMLElement;
  //       var value: number;
  //       if (headers[i].isFinal) {
  //         value = getAttr(agent, headers[i].field, state.replay.max_steps - 1)
  //       } else {
  //         value = getAttr(agent, headers[i].field)
  //       }
  //       let valueStr = value.toString();
  //       if (valueStr.includes(".")) {
  //         valueStr = value.toFixed(3);
  //       }
  //       cell.children[0].textContent = valueStr;
  //       row.appendChild(cell);
  //     }
  //     agentTable.appendChild(row);
  //   }
  // }

  // Create the columns.
  for (let columnDef of columns) {

    let column = columnTemplate.cloneNode(true) as HTMLElement;
    let headerCell = headerCellTemplate.cloneNode(true) as HTMLElement;
    let name = headerCell.querySelectorAll(".name")[0];
    name.textContent = columnDef.name;
    let icon = headerCell.querySelectorAll(".icon")[0];
    icon.setAttribute("src", "data/resources/" + columnDef.field + ".png");
    column.appendChild(headerCell);

    // Create the data cells.
    for (let i = 0; i < agents.length; i++) {
      let agent = agents[i];
      if (agent != null) {
        let dataCell = dataCellTemplate.cloneNode(true) as HTMLElement;

        var value: number;
        if (columnDef.isFinal) {
          value = getAttr(agent, columnDef.field, state.replay.max_steps - 1)
        } else {
          value = getAttr(agent, columnDef.field)
        }
        let valueStr = value.toString();
        if (valueStr.includes(".")) {
          valueStr = value.toFixed(3);
        }

        dataCell.children[0].textContent = valueStr;
        let agentId = getAttr(agent, "agent_id");
        dataCell.setAttribute("data-agent-id", agentId.toString());
        if (state.selectedGridObject != null && agentId == getAttr(state.selectedGridObject, "agent_id")) {
          dataCell.classList.add("selected");
        }
        column.appendChild(dataCell);
      }
    }

    table.appendChild(column);
  }

  let newColumn = newColumnTemplate.cloneNode(true) as HTMLElement;
  let headerCell = newColumnHeaderCell.cloneNode(true) as HTMLElement;
  newColumn.appendChild(headerCell);
  for (let i = 0; i < agents.length; i++) {
    let dataCell = newColumnDataCell.cloneNode(true) as HTMLElement;
    let agent = agents[i];
    let agentId = getAttr(agent, "agent_id");
    dataCell.setAttribute("data-agent-id", agentId.toString());
    if (state.selectedGridObject != null && agentId == getAttr(state.selectedGridObject, "agent_id")) {
      dataCell.classList.add("selected");
    }
    newColumn.appendChild(dataCell);
  }
  table.appendChild(newColumn);

}
