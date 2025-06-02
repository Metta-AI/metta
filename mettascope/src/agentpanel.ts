import { find, finds, removeChildren, walkUpAttribute, onEvent, showMenu, hideMenu, showDropdown, hideDropdown } from "./htmlutils.js";
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

  new ColumnDefinition("Red Ore", "inv:ore.red", false),
  new ColumnDefinition("Blue Ore", "inv:ore.blue", false),
  new ColumnDefinition("Green Ore", "inv:ore.green", false),

  new ColumnDefinition("Red Battery", "inv:battery.red", false),
  new ColumnDefinition("Blue Battery", "inv:battery.blue", false),
  new ColumnDefinition("Green Battery", "inv:battery.green", false),
];
var mainSort: ColumnDefinition = columns[1];


// Capitalize the first letter of every word in a string.
// Example: "hello world" -> "Hello World"
function capitalize(str: string) {
  return str.split(" ").map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(" ");
}

// Swaps the element 1 position to the right.
function swapRight(list: any[], element: any) {
  let index = list.indexOf(element);
  if (index == -1) {
    return;
  }
  let tmp = list[index];
  list[index] = list[index + 1];
  list[index + 1] = tmp;
}

// Swaps the element 1 position to the left.
function swapLeft(list: any[], element: any) {
  let index = list.indexOf(element);
  if (index == -1) {
    return;
  }
  let tmp = list[index];
  list[index] = list[index - 1];
  list[index - 1] = tmp;
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
const columnOptions = find("#new-column-dropdown .column-options");
const columnOptionTemplate = find("#new-column-dropdown .column-option");

export function initAgentTable() {

  columnMenu.classList.add("hidden");
  newColumnDropdown.classList.add("hidden");

  // Clicking on the column menu button should show the column menu.
  onEvent("click", "#agent-panel .header-cell .dropdown", (target: HTMLElement, e: Event) => {
    let columnMenu = find("#column-menu");
    let columnName = walkUpAttribute(target, "data-column-name");
    columnMenu.setAttribute("data-column-name", columnName);
    showMenu(target, columnMenu);
  });

  // Clicking on the sort up button should sort the column in ascending order.
  onEvent("click", "#column-menu .sort-up", (target: HTMLElement, e: Event) => {
    console.log("Sort up clicked");
    let columnName = walkUpAttribute(target, "data-column-name");
    for (let i = 0; i < columns.length; i++) {
      if (columns[i].name == columnName) {
        columns[i].sortDirection = -1;
        mainSort = columns[i];
      } else {
        columns[i].sortDirection = 0;
      }
    }
    updateAgentTable();
    hideMenu();
  });

  // Clicking on the sort down button should sort the column in descending order.
  onEvent("click", "#column-menu .sort-down", (target: HTMLElement, e: Event) => {
    console.log("Sort up clicked");
    let columnName = walkUpAttribute(target, "data-column-name");
    for (let i = 0; i < columns.length; i++) {
      if (columns[i].name == columnName) {
        columns[i].sortDirection = 1;
        mainSort = columns[i];
      } else {
        columns[i].sortDirection = 0;
      }
    }
    updateAgentTable();
    hideMenu();
  });

  onEvent("click", "#column-menu .move-left", (target: HTMLElement, e: Event) => {
    console.log("Move left clicked");
    let columnName = walkUpAttribute(target, "data-column-name");
    let column = columns.find(column => column.name == columnName);
    if (column != null) {
      swapLeft(columns, column);
      updateAgentTable();
    }
    hideMenu();
  });

  // Clicking on the move right button should move the column to the right.
  onEvent("click", "#column-menu .move-right", (target: HTMLElement, e: Event) => {
    console.log("Move right clicked");
    let columnName = walkUpAttribute(target, "data-column-name");
    let column = columns.find(column => column.name == columnName);
    if (column != null) {
      swapRight(columns, column);
      updateAgentTable();
    }
    hideMenu();
  });

  // Clicking on the hide column button should remove the column from the columns array.
  onEvent("click", "#column-menu .hide-column", (target: HTMLElement, e: Event) => {
    console.log("Hide column clicked");
    hideMenu();
    // Remove this column from the columns array.
    let columnName = walkUpAttribute(target, "data-column-name");
    columns = columns.filter(column => column.name != columnName);
    updateAgentTable();
  });

  // Clicking on the table directly should cycle the sort direction.
  onEvent("click", "#agent-panel .header-cell", (target: HTMLElement, e: Event) => {
    let columnField = walkUpAttribute(target, "data-column-field");
    let columnIsFinal = walkUpAttribute(target, "data-column-is-final") == "true";
    if (columnField != "") {
      for (let column of columns) {
        if (column.field == columnField && column.isFinal == columnIsFinal) {
          if (mainSort == column) {
            if (column.sortDirection == 0) {
              column.sortDirection = 1
            } else {
              column.sortDirection = -column.sortDirection;
            }
          } else {
            column.sortDirection = 1;
          }
          mainSort = column;
        } else {
          column.sortDirection = 0;
        }
      }
      updateAgentTable();
    }
  });

  // Clicking on a data cell should select the agent.
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

  // Clicking on the new column input should show the new column dropdown and
  // allow you to type-ahead to select search for the column.
  onEvent("click", "#new-column-input", (target: HTMLElement, e: Event) => {
    let newColumnDropdown = find("#new-column-dropdown");
    updateAvailableColumns()
    showDropdown(target, newColumnDropdown);
  });

  onEvent("click", "#new-column-dropdown .step-check", (target: HTMLElement, e: Event) => {
    if (target.getAttribute("src") == "data/ui/check-on.png") {
      target.setAttribute("src", "data/ui/check-off.png");
    } else {
      target.setAttribute("src", "data/ui/check-on.png");
    }
  });

  onEvent("click", "#new-column-dropdown .final-check", (target: HTMLElement, e: Event) => {
    if (target.getAttribute("src") == "data/ui/check-on.png") {
      target.setAttribute("src", "data/ui/check-off.png");
    } else {
      target.setAttribute("src", "data/ui/check-on.png");
    }
  });

  removeChildren(table);
  removeChildren(columnTemplate);
  removeChildren(newColumnTemplate);
}

export function updateAvailableColumns() {
  // Replay format might change the available columns, in real time.

  var availableColumns: ColumnDefinition[] = [];
  // All agent keys:
  let agentKeys = new Set<string>();
  for (let agent of state.replay.agents) {
    for (let key in agent) {
      if (key.startsWith("agent:")) {
        agentKeys.add(key);
      }
    }
  }
  // All inventory keys:
  for (let key of agentKeys) {
    let name = capitalize(key.replace("agent:", "").replace("inv:", "").replace(".", " "));
    availableColumns.push(new ColumnDefinition(name, key, false));
  }

  removeChildren(columnOptions);

  for (let column of availableColumns) {
    let option = columnOptionTemplate.cloneNode(true) as HTMLElement;
    option.querySelector(".name")!.textContent = column.name;
    let src = "data/resources/" + column.field.replace("inv:", "").replace("agent:", "") + ".png";
    option.querySelector(".icon")!.setAttribute("src", src);
    option.setAttribute("title", column.field);
    option.setAttribute("data-column-name", column.name);
    columnOptions.appendChild(option);
  }

}

export function updateAgentTable() {
  removeChildren(agentTable);

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

  // Create the columns.
  for (let columnDef of columns) {

    let column = columnTemplate.cloneNode(true) as HTMLElement;
    column.setAttribute("data-column-field", columnDef.field);
    column.setAttribute("data-column-is-final", columnDef.isFinal.toString());
    let headerCell = headerCellTemplate.cloneNode(true) as HTMLElement;
    let name = headerCell.querySelectorAll(".name")[0];
    name.textContent = columnDef.name;
    let icon = headerCell.querySelectorAll(".icon")[0];
    let iconName = columnDef.field.replace("inv:", "").replace("agent:", "")
    icon.setAttribute("src", "data/resources/" + iconName + ".png");

    let sortIcon = headerCell.querySelector(".sort-icon") as HTMLElement;
    if (columnDef.sortDirection == 1) {
      sortIcon.setAttribute("src", "data/ui/sort-down.png");
    } else if (columnDef.sortDirection == -1) {
      sortIcon.setAttribute("src", "data/ui/sort-up.png");
    } else {
      sortIcon.classList.add("hidden");
    }
    let title = columnDef.field
    if (columnDef.isFinal) {
      title = "Final: " + title;
    }
    headerCell.setAttribute("title", title);
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
